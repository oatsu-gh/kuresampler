# Copyright (c) 2025 oatsu
"""Wavtool

wavtool に求められること
- 実行引数を解釈してプロパティにセットする
- envelope 文字列を解釈し、音量調節を行う。この際 ove も取得する。
- whd ファイルを読み書きする ※WORLD特徴量ベースでは使用しない
- dat ファイルを読み書きする ※WORLD特徴量ベースでは使用しない
- wav ファイルを読み書きする

wavtool の処理の流れ
- wav ファイルを読み込む
- stp, length に基づいて wav ファイルをクロップする
- envelope に基づいて音量調節を行う
- 既存の wav を読み込んでオーバーラップさせる
- wav ファイルを書き出す

## 注意すること
- 自身が 先頭ノート/中間ノート/最終ノート のいずれなのか不明なので、wav ファイルは常に出力必要。
- wav ファイルを出力する際、既存の wav ファイルがある場合は、オーバーラップ時間を考慮して重ねる。


UTAU から wavtool に渡されるコマンドの例
-----------------------
<tool> <output> <temp> <stp> <length> <envelope>
-----------------------
    tool   : wavtool.exe など実行ファイルのパス
    output : 出力wavファイルパス
    temp   : 入力wavファイルパス
    stp    : 入力wav使用開始位置[ms]
    length : ノート長さ[ms]
    env    : 音量エンベロープのパラメータ (複数可)


"""
import argparse
import logging
import sys
from copy import copy
from functools import partial
from pathlib import Path

import colored_traceback.auto  # noqa: F401
import numpy as np
import pyworld
import torch
from nnsvs.util import StandardScaler
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))  # for local import

from convert import (  # noqa: F401
    nnsvs_to_npzfile,
    nnsvs_to_world,
    npzfile_to_nnsvs,
    npzfile_to_world,
    waveform_to_wavfile,
    waveform_to_world,
    wavfile_to_waveform,
    world_to_nnsvs,
    world_to_nnsvs_to_waveform,
    world_to_npzfile,
    world_to_waveform,
)
from util import (
    get_device,
    load_vocoder_model,
    overlap_ap,
    overlap_f0,
    overlap_sp,
    round_by_frame,
    setup_logger,
    str2float,
)

DEFAULT_SAMPLE_RATE = 44100


def get_overlap(envelope: list[float]) -> float:
    """Envelope から ove だけを取得する。

    Args:
        envelope (list[float]): エンベロープ値のリスト

    """
    len_envelope = len(envelope)
    if len_envelope in [2, 7]:
        return 0.0
    if len_envelope in [8, 9, 10, 11]:
        return envelope[7]
    msg = (
        f'Invalid envelope length ({len_envelope}). '
        f'The length must be 2, 7, 8, 9, 10, or 11.: {envelope}'
    )
    raise ValueError(msg)


# MARK: NeuralNetworkWavTool
class NeuralNetworkWavTool:
    """WAV出力の代わりに WORLD の特徴量を出力するのに用いる。

    Args:
        output: 出力ファイルのパス(拡張子なし)
        export_wav: WAVファイルを出力するか否か
        export_features: WORLD特徴量を npz ファイルで出力するか否か
        frame_period: WORLD特徴量のフレーム周期 (ms)
        accumulated_features: メモリ上の累積特徴量 (f0, sp, ap) のタプル (オプション)

    ## PyWavTool.WavTool からの変更点
    - whd と dat を使用しない

    ### ファイル入出力について
    入力においては、npz がある場合は高速化のため npz を wav の代わりに優先的に読み込む。
    出力においては、npz を優先する場合でも常に wav 生成をしなければならない。
    NOTE: UTAU から呼び出される場合、実行中のノートが 最初/途中/最後 のどれか不明なので wav 出力は常に必要。

    ### キャッシュの取り扱い
    - WAVキャッシュを使用する場合、WORLD 特徴量に変換してから append する。
    - NPZキャッシュを使用する場合、NPZファイルから直接特徴量を読み込んで append する。
    - accumulated_features が渡された場合、メモリ上の特徴量を優先的に使用する。

    ### 内部データの取り扱い
    - self.dat は常に WORLD 特徴量を保持する。output のときだけ wav に変換する。
    - PyWavTool.WavTool._dat (waveform) の要素数はサンプルレートに応じた長さ (秒数*sample_rate) だったが、
      WorldFeatureWavTool.self._dat (WORLD特徴量) の要素数は frame_period に応じた長さ (秒数/frame_period*1000) になることに注意。

    TODO: 音量ノーマライズの際に WORLD 特徴量にノーマライズをかける方法を検討する。いったんWAVに変換して係数を算出する?


    Example:
        # 2つのWAV結合後の長さが想定通りであることを確認する。
        >>> wavtool = NeuralNetworkWavTool(
        ...     output_wav='test_nnwavtool_output.wav',
        ...     input_wav='test/sine_440Hz_sr44100.wav',
        ...     stp=0,
        ...     length=201.0,
        ...     # p1 p2 p3 v1 v2 v3 v4 ove p4
        ...     envelope=[6.0, 12.0, 12.0, 50, 100, 100, 50, 11.3, 6.0],
        ...     use_vocoder_model=False,
        ...     carryover_error=2.0,
        ... )
        >>> wavtool.overlap  # 誤差 = 11.3-10 = 1.3
        10
        >>> wavtool.length  # ノート長さ [ms] = carryover_error(2.0) + 元length(201.0) - overlap誤差(1.3) = 201.7 を5で丸め
        200
        >>> wavtool.envelope_p  # 音量エンベロープの時刻リスト
        [0, 5, 20, 180, 195, 200]
        >>> wavtool.envelope_v  # 音量エンベロープの音量リスト
        [0, 50, 100, 100, 50, 0]
        >>> wavtool.carryover_error  # {carryover_error(2.0) + 元length(201) - 元overlap(11.3)} - (丸めlength(200) - 丸めoverlap(10))
        1.6999999999999886

    """  # noqa: E501

    # 入出力パス
    input_wav: Path  # 入力wavのパス
    input_npz: Path  # 入力npzのパス
    output_wav: Path  # 出力wavのパス
    output_npz: Path  # 出力npzのパス
    stp: float  # 入力wavの先頭のオフセット [ms]
    length: float  # 追記したい音声長さ [ms]
    # WORLD特徴量
    frame_period: int  # WORLD特徴量のフレーム周期 [ms]
    f0: np.ndarray  # f0 (WORLD特徴量 F0)
    sp: np.ndarray  # sp (WORLD特徴量 Spectral envelope)
    ap: np.ndarray  # ap (WORLD特徴量 Aperiodicity)
    f0_appended: np.ndarray  # 追記後のf0 (WORLD特徴量 F0)
    sp_appended: np.ndarray  # 追記後のsp (WORLD特徴量 Spectral envelope)
    ap_appended: np.ndarray  # 追記後のap (WORLD特徴量 Aperiodicity)
    # サンプルレート関連
    internal_sample_rate: int  # 内部処理のサンプルレート [Hz]
    target_sample_rate: int  # 出力wavのサンプルレート [Hz]
    resample_type: str  # リサンプリングの種類
    # 出力用データ
    _waveform: np.ndarray | None  # 出力wavの波形データ
    # 音量エンベロープ関連
    envelope_p: list[float]  # 音量エンベロープの時刻のリスト [ms]
    envelope_v: list[float]  # 音量エンベロープの音量値のリスト(0-100-200) [-]
    overlap: float  # クロスフェード時間 [ms]
    # ボコーダー関連
    use_vocoder_model: bool = True  # Vocoder model を使用するか否か
    vocoder_model: torch.nn.Module | None = None  # Vocoder model
    vocoder_in_scaler: StandardScaler | None = None  # Vocoder input scaler
    vocoder_config: ListConfig | DictConfig | None = None  # Vocoder config
    vocoder_type: str
    vocoder_feature_type: str
    vocoder_vuv_threshold: float
    vocoder_frame_period: int
    device: torch.device
    # その他
    logger: logging.Logger
    carryover_error: float  # 丸め誤差 [ms]
    accumulated_features: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # MARK: __init__
    def __init__(
        self,
        output_wav: Path | str,
        input_wav: Path | str,
        stp: float,
        length: float,
        envelope: list[float],
        *,
        use_vocoder_model: bool,
        logger: logging.Logger | None = None,
        frame_period: int = 5,
        vocoder_model: torch.nn.Module | None = None,
        vocoder_in_scaler: StandardScaler | None = None,
        vocoder_config: DictConfig | ListConfig | None = None,
        vocoder_type: str = 'usfgan',
        vocoder_feature_type: str = 'world',
        vocoder_vuv_threshold: float = 0.5,
        vocoder_frame_period: int = 5,
        internal_sample_rate: int = 48000,
        target_sample_rate: int = 44100,
        resample_type: str = 'soxr_vhq',
        accumulated_features: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        feature_dtype: str = 'float64',
        carryover_error: float = 0.0,  # このノート以前の時刻丸め誤差
    ) -> None:
        """NeuralNetworkWavTool のコンストラクタ"""
        self.logger = logger or setup_logger(level=logging.INFO, name=self.__class__.__name__)
        self.input_wav = Path(input_wav)
        self.input_npz = Path(input_wav).with_suffix('.npz')
        self.output_wav = Path(output_wav)
        self.output_npz = Path(output_wav).with_suffix('.npz')
        self.frame_period = frame_period
        self.feature_dtype = feature_dtype
        # サンプルレート関連の初期化
        self.internal_sample_rate = internal_sample_rate
        self.target_sample_rate = target_sample_rate
        self.resample_type = resample_type
        # carryover_error を仮初期化。stp や length の初期化で更新される。
        self.carryover_error = carryover_error
        # length, envelope_p, envelope_v, overlap, carryover_error を初期化
        self._init_length_and_envelope(length, envelope)
        # stp を初期化
        self._init_stp(stp)
        # length と carryover_error を初期化
        # sample_rate, f0, sp, ap を初期化
        self._init_features()

        # デバイス設定
        self.device = get_device()
        # ボコーダー関連の設定
        self.use_vocoder_model = use_vocoder_model
        self.vocoder_type = vocoder_type
        self.vocoder_feature_type = vocoder_feature_type
        self.vocoder_vuv_threshold = vocoder_vuv_threshold
        self.vocoder_frame_period = vocoder_frame_period
        # その他
        self.accumulated_features = accumulated_features

        # frame_period と vocoder_frame_period が異なる場合は警告を出す
        if self.frame_period != self.vocoder_frame_period:
            msg = (
                f'frame_period ({self.frame_period} ms) '
                f'and vocoder_frame_period ({self.vocoder_frame_period} ms) do not match. '
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # use_vocoder_model が True の時はボコーダーモデルを代入する
        if use_vocoder_model:
            # vocoder model 関連の引数が全て揃っていることを確認
            if vocoder_model is None or vocoder_in_scaler is None or vocoder_config is None:
                msg = (
                    'When use_vocoder_model is True, '
                    'vocoder_model, vocoder_in_scaler, and vocoder_config must be provided.'
                )
                self.logger.error(msg)
                raise ValueError(msg)
            self.vocoder_model = vocoder_model
            self.vocoder_in_scaler = vocoder_in_scaler
            self.vocoder_config = vocoder_config
            self.logger.info(f'Using vocoder model: {self.use_vocoder_model}')
            # サンプルレートチェック
            if self.vocoder_config.data.sample_rate != self.internal_sample_rate:
                msg = (
                    f'Vocoder model sample rate ({self.vocoder_config.data.sample_rate} Hz) '
                    f'and internal sample rate ({self.internal_sample_rate} Hz) are different.'
                )
                self.logger.error(msg)
                raise ValueError(msg)
        # use_vocoder_model が False の場合
        else:
            # use_vocoder_model が False なのに vocoder が指定されているときは警告を出す
            if (vocoder_model, vocoder_in_scaler, vocoder_config) != (None, None, None):
                self.logger.warning(
                    'use_vocoder_model is False, '
                    'but [vocoder_model, vocoder_in_scaler, or vocoder_config] are provided. '
                    'They will be ignored.'
                )
                self.vocoder_model, self.vocoder_in_scaler, self.vocoder_config = None, None, None
            self.logger.info('Not using vocoder model.')

        # 出力フォルダが存在しなければ作成
        Path(output_wav).parent.mkdir(parents=True, exist_ok=True)

    @property
    def vocoder_sample_rate(self) -> int:
        """ボコーダーモデルのwav出力サンプリング周波数"""
        if self.vocoder_config is None:
            msg = 'vocoder_config is None. vocoder_model must be loaded first.'
            raise ValueError(msg)
        return self.vocoder_config.data.sample_rate

    @property
    def fft_size(self) -> int:
        """FFTサイズ"""
        return pyworld.get_cheaptrick_fft_size(self.internal_sample_rate)  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def waveform(self) -> np.ndarray:
        """出力wavの波形データ"""
        if self._waveform is None:
            msg = 'self._waveform is None. Run generate_waveform() first.'
            self.logger.error(msg)
            raise ValueError(msg)
        return self._waveform

    @waveform.setter
    def waveform(self, value: np.ndarray) -> None:
        """出力wavの波形データをセットする。"""
        self._waveform = value

    def _init_length_and_envelope(self, original_length: float, envelope: list[float]) -> None:
        """self.length, self.carryover_error, self.overlap, self.envelope_p, self.envelope_v, self.overlap を初期化する。

        Args:
            original_length     (float)        : 元の長さ [ms]
            envelope            (list[float])  : エンベロープの値のリスト
            initial_error     (float)        : 直前ノートまでの丸め誤差 [ms]

        Results:
            self.envelope_p      (list[float]) : 時刻のリスト(ms)。エンベロープが2点の場合空配列。
            self.envelope_v      (list[float]) : 音量値のリスト。0-200の範囲であることを想定。
            self.overlap         (float)       : クロスフェード時間(ms)。
            self.carryover_error (float)       : 次のノートに持ち越す丸め誤差(ms)

        ## エンベロープのパターン
        - 長さ2 : p1 p2
        - 長さ7 : p1 p2 p3 v1 v2 v3 v4
        - 長さ8 : p1 p2 p3 v1 v2 v3 v4 ove
        - 長さ9 : p1 p2 p3 v1 v2 v3 v4 ove p4
        - 長さ10: p1 p2 p3 v1 v2 v3 v4 ove p4 Unknown ※詳細不明
        - 長さ11: p1 p2 p3 v1 v2 v3 v4 ove p4 p5 v5
        p1, p2, p3, p4, p5, ove : float (ms)
        v1, v2, v3, v4, v5 : int (1-200)

        ## 各値の計算方法
        p1: ノート頭からの相対時刻(ms)
        p2: p1 からの相対時刻(ms)。
        p3: p4 がない場合は末端からの相対距離(ms)。 p4 がある場合は p4 からの相対距離(ms)。
        p4: 末端からの相対時刻(ms)
        p5: p2 からの相対時刻(ms)
        v1: そのまま
        v2: そのまま
        v3: そのまま
        v4: そのまま
        v5: そのまま

        """  # noqa: E501
        # 丸め関数を定義
        round_func = partial(round_by_frame, frame_period=self.frame_period)
        # 初期誤差を取得
        initial_error = self.carryover_error
        # オーバーラップの丸め誤差を計算
        original_overlap = get_overlap(envelope)
        rounded_overlap = round_func(original_overlap)
        # overlap が負の値のときにはクロスフェードができないので 0 に強制する
        if rounded_overlap < 0:
            msg = f'Negative overlap ({rounded_overlap} ms) is detected. Force 0 ms.'
            self.logger.warning(msg)
            rounded_overlap = 0
        overlap_error = (
            original_overlap - rounded_overlap
        )  # 正の場合はオーバーラップ時間が短くなったことを意味する -> 実質ノート長が長くなる

        # オーバーラップの丸め誤差を考慮して length を調整
        rounded_length = round_func(initial_error + original_length - overlap_error)
        # 次のノートに持ち越す丸め誤差を計算
        ## 本来の実質長さ - 丸め後の実質長さ
        final_error = (initial_error + original_length - original_overlap) - (rounded_length - rounded_overlap)  # noqa: E501 # fmt: skip

        # エンベロープを要素数に応じて展開 -------------------------
        len_envelope = len(envelope)
        rounded_p_list: list[float]
        v_list: list[float]

        ## 長さ2の時は空配列を返す
        if len_envelope == 2:
            rounded_p_list = []
            v_list = []
            rounded_overlap = 0

        ## 長さ7の時は overlap がない
        ## [p1, p2, p3, v1, v2, v3, v4]
        elif len_envelope == 7:
            p1, p2, p3 = envelope[0:3]
            v1, v2, v3, v4 = envelope[3:7]
            rounded_p_list = [
                0,
                round_func(p1),
                round_func(p1 + p2),
                rounded_length - round_func(p3),
                rounded_length,
            ]
            v_list = [0, v1, v2, v3, v4, 0]
            rounded_overlap = 0
        ## 長さ8の時は overlap が追加される
        ## [p1, p2, p3, v1, v2, v3, v4, ove]
        elif len_envelope == 8:
            p1, p2, p3 = envelope[0:3]
            v1, v2, v3, v4 = envelope[3:7]
            rounded_p_list = [
                0,
                round_func(p1),
                round_func(p1 + p2),
                rounded_length - round_func(p3),
                rounded_length,
            ]
            v_list = [0, v1, v2, v3, v4, 0]
            rounded_overlap = round_func(envelope[7])
        ## 長さ9 の時は p4 が追加される。
        ## [p1, p2, p3, v1, v2, v3, v4, ove, p4]
        ## 長さが10の時は何が追加されているかよくわからない(値は0)が、9と同じ処理をする
        ## [p1, p2, p3, v1, v2, v3, v4, ove, p4, Unknown]
        elif len_envelope in (9, 10):
            # [p1, p2, p3, v1, v2, v3, v4, ove, p4]
            p1, p2, p3 = envelope[0:3]
            v1, v2, v3, v4 = envelope[3:7]
            rounded_overlap = round_func(envelope[7])
            p4 = envelope[8]
            rounded_p_list = [
                0,
                round_func(p1),
                round_func(p1 + p2),
                rounded_length - round_func(p4 + p3),
                rounded_length - round_func(p4),
                rounded_length,
            ]
            v_list = [0, v1, v2, v3, v4, 0]
        ## 長さが11の時は p5, v5 が追加される
        ## [p1, p2, p3, v1, v2, v3, v4, ove, p4, p5, v5]
        elif len_envelope == 11:
            # [p1, p2, p3, v1, v2, v3, v4, ove, p4, p5, v5]
            p1, p2, p3 = envelope[0:3]
            v1, v2, v3, v4 = envelope[3:7]
            rounded_overlap = round_func(envelope[7])
            p4 = envelope[8]
            p5 = envelope[9]
            v5 = envelope[10]
            # NOTE: p5 の位置は p2 と p3 の間であることに注意!
            rounded_p_list = [
                0,
                round_func(p1),
                round_func(p1 + p2),
                round_func(p1 + p2 + p5),
                rounded_length - round_func(p4 + p3),
                rounded_length - round_func(p4),
                rounded_length,
            ]  # 絶対時刻
            v_list = [0, v1, v2, v5, v3, v4, 0]
        # それ以外の要素数はエラー
        else:
            msg = (
                f'Invalid envelope length ({len_envelope}). '
                f'The length must be 2, 7, 8, 9, or 11.: {envelope}'
            )
            raise ValueError(msg)
        # p_list が昇順になっていない場合はエラー
        if rounded_p_list != sorted(rounded_p_list):
            msg = f'p_list must be in ascending order, but got {rounded_p_list}.'
            raise ValueError(msg)

        # デバッグ出力
        self.logger.debug('Initialized length and envelope:')
        self.logger.debug('  initial_error     : %s ms', initial_error)
        self.logger.debug('  original_overlap  : %s ms', original_overlap)
        self.logger.debug('  rounded_overlap   : %s ms', rounded_overlap)
        self.logger.debug('  original_length   : %s ms', original_length)
        self.logger.debug('  rounded_length    : %s ms', rounded_length)
        self.logger.debug('  final_error       : %s ms', final_error)
        self.logger.debug('  envelope_p        : %s', rounded_p_list)
        self.logger.debug('  envelope_v        : %s', v_list)
        # 値をセット
        self.length = rounded_length
        self.envelope_p = rounded_p_list
        self.envelope_v = v_list
        self.overlap = rounded_overlap
        self.carryover_error = final_error

    def _init_stp(self, stp: float) -> None:
        """self.stp を初期化する。その際の丸め誤差を carryover_error に加算する。"""
        rounded_stp = round_by_frame(stp, frame_period=self.frame_period)
        stp_error = stp - rounded_stp
        self.stp = rounded_stp
        self.carryover_error -= stp_error

    def _init_features(self) -> None:
        """self.f0, self.sp, self.ap, self.sample_rate を初期化する。

        入力wavまたはnpzを読み込み、WORLD特徴量に変換して self.f0, self.sp, self.ap にセットする。
        npzが存在する場合はnpzを優先的に読み込む。
        特徴量が渡されている場合はそれを優先的に使用する。

        Args:
            accumulated_features (tuple[np.ndarray, np.ndarray, np.ndarray] | None):
                メモリ上の累積特徴量 (f0, sp, ap) のタプル

        Note:
            この関数は入力ファイル (input_wav/input_npz) の特徴量を読み込む。

        """
        # npz が存在する場合は npz から特徴量とサンプルレートを取得する。
        if self.input_npz.exists():
            self.logger.debug(f'Using WORLD features from NPZ: {self.input_npz}')
            f0, sp, ap, npz_sample_rate = npzfile_to_world(self.input_npz)
            # npz のサンプルレートが内部サンプルレートと異なる場合はエラー
            if npz_sample_rate != self.internal_sample_rate:
                msg = (
                    f"NPZ file's sample rate ({npz_sample_rate} Hz) and "
                    f'internal_sample_rate ({self.internal_sample_rate} Hz) do not match.'
                )
                self.logger.error(msg)
                raise ValueError(msg)

        # wav のみ存在する場合はサンプルレート変換したのちに特徴量抽出する。
        elif self.input_wav.exists():
            self.logger.debug(f'Using WORLD features from WAV: {self.input_wav}')
            waveform, _, _ = wavfile_to_waveform(
                self.input_wav,
                target_sample_rate=self.internal_sample_rate,
                resample_type=self.resample_type,
            )
            f0, sp, ap = waveform_to_world(
                waveform,
                sample_rate=self.internal_sample_rate,
                frame_period=self.frame_period,
            )
        # メモリ上の特徴量も npz も wav も存在しない場合は無音特徴量を使用する。
        else:
            msg = (
                f'Both WAV ({self.input_wav}) and NPZ ({self.input_npz}) do not exist. '
                'Using silent features.'
            )
            self.logger.info(msg)
            n_frames = round(self.length / self.frame_period)
            f0 = np.zeros((n_frames,), dtype=self.feature_dtype)
            sp = np.zeros((n_frames, self.fft_size // 2 + 1), dtype=self.feature_dtype)
            ap = np.zeros((n_frames, self.fft_size // 2 + 1), dtype=self.feature_dtype)

        # 読み取った特徴量の dtype を揃える
        self.f0 = f0.astype(self.feature_dtype)
        self.sp = sp.astype(self.feature_dtype)
        self.ap = ap.astype(self.feature_dtype)
        # デバッグ出力: f0, sp, ap のshape,min,maxを確認
        self.logger.debug('Initial features:')
        self.debug_features(f0=self.f0, sp=self.sp, ap=self.ap)

    def _apply_range(self) -> None:
        """self.f0, self.sp, self.ap に stp, length を適用する。

        stp, length に基づいて特徴量をクロップする。
        """
        length_by_frame = round(self.length / self.frame_period)
        # stp, length に基づいて特徴量をクロップする
        start_frame = round(self.stp / self.frame_period)
        end_frame = start_frame + length_by_frame
        self.f0 = self.f0[start_frame:end_frame]
        self.sp = self.sp[start_frame:end_frame, :]
        self.ap = self.ap[start_frame:end_frame, :]

    def _apply_envelope(self) -> None:
        """self.f0, self.sp, self.ap に音量エンベロープを適用する。

        TODO: 音量エンベロープの時刻と音量値に基づいて、spectrogram の音量加工を行う。
        """
        n_frames = self.sp.shape[0]
        self.logger.debug('envelope_p : %s', self.envelope_p)
        self.logger.debug('envelope_v : %s', self.envelope_v)
        self.logger.debug('sp.shape   : %s', self.sp.shape)
        self.logger.debug('n_frames   : %s', n_frames)
        # エンベロープが2点以下の場合は何もしない
        if len(self.envelope_p) < 2:
            return
        # エンベロープが3点以上の場合は音量エンベロープを適用する
        x = np.arange(n_frames) + 0.5
        # 時刻をフレーム単位に変換
        ## NOTE: 時刻(slice)でなくフレーム(index)で音量制御する都合上、
        ## NOTE: xp の最大値を n_frames-1 にしないと終端フレームが0にならないので時刻補正必要。
        ## TODO: 非常に短いノートではクロスフェード長がずれてしまう可能性があるため、フェードイン(p1,2,5)/アウト(p3,4)の時刻を別々に計算するなどの厳密な実装が必要。  # noqa: E501
        xp = [round(p / self.frame_period) for p in self.envelope_p]
        # 音量値(0-200)を0-2に正規化 (余った v は無視)
        fp = [v / 100.0 for v in self.envelope_v[: len(xp)]]
        # 音量エンベロープを計算
        volume_envelope = np.interp(x, xp, fp)
        # sp に音量エンベロープを適用する。音量を x 倍するには sp を x^2 倍する。
        self.sp *= volume_envelope[:, np.newaxis] ** 2
        self.logger.debug('x                    : %s', x)
        self.logger.debug('xp                   : %s', xp)
        self.logger.debug('fp                   : %s', fp)
        self.logger.debug('volume_envelope      : %s', volume_envelope)
        self.logger.debug('volume_envelope.shape: %s', volume_envelope.shape)

    def _apply_all(self) -> None:
        """self.f0, self.sp, self.ap に stp, length, envelope を適用する。

        - stp, length に基づいて特徴量をクロップする。
        - envelope に基づいて音量調整を行う。

        self.f0, self.sp, self.ap に音量エンベロープを適用する。
        音量エンベロープの時刻と音量値に基づいて、f0, sp, ap の各フレームに対して音量調整を行う。
        """
        length_by_frame = round(self.length / self.frame_period)
        if length_by_frame < 0:
            msg = f'Invalid length: {self.length} ms. Length must be non-negative.'
            self.logger.error(msg)
            raise ValueError(msg)
        # クロップする
        self._apply_range()
        # 音量エンベロープを適用する
        self._apply_envelope()

    def debug_features(self, **kwargs: np.ndarray) -> None:
        """self.f0, self.sp, self.ap の情報をログに出力する。"""
        # 空の ndarray をSkipする
        kwargs = {k: v for k, v in kwargs.items() if v.size > 0}
        # shape を出力
        for name, array in kwargs.items():
            self.logger.debug('  %s.shape: %s', name, array.shape)
        # min, max を出力
        for name, array in kwargs.items():
            self.logger.debug('  %s (min, max): (%s, %s)', name, array.min(), array.max())

    # MARK: append
    def append(self) -> None:
        """既存のnpzファイルを読み取って、それに書き込む。wav は全体を再計算して出力する。

        TODO: ノート数が多いほどWAV生成が重くなるので何とかしたい。
        """
        # 既存ファイルの特徴量を読み取る。なければ空の配列を取得する。
        # メモリ上の累積特徴量が渡されている場合はそれを使用
        if self.accumulated_features is not None:
            self.logger.info('Using features on memory')
            long_f0, long_sp, long_ap = self.accumulated_features
        # メモリ上の累積特徴量がない場合はファイルから読み込む
        elif self.output_npz.exists():
            self.logger.info('Loading existing features from: %s', self.output_npz)
            long_f0, long_sp, long_ap, npz_sample_rate = npzfile_to_world(self.output_npz)
            if npz_sample_rate != self.internal_sample_rate:
                msg = (
                    f"Existing NPZ file's sample rate ({npz_sample_rate} Hz) and "
                    f'internal_sample_rate ({self.internal_sample_rate} Hz) are different.'
                )
                self.logger.error(msg)
                raise ValueError(msg)
        else:
            self.logger.info('No existing features found. Starting fresh.')
            long_f0, long_sp, long_ap, _ = (
                np.array([]),
                np.array([[]]),
                np.array([[]]),
                self.internal_sample_rate,
            )

        # クロップしたのちエンベロープを適用する
        self._apply_all()

        # overlap をフレーム数に変換
        n_overlap_frames = round(self.overlap / self.frame_period)
        self.logger.debug('overlap_frames: %s', n_overlap_frames)

        # デバッグ出力 --------------------------
        self.logger.debug('Features before overlap:')
        self.debug_features(
            long_f0=long_f0,
            long_sp=long_sp,
            long_ap=long_ap,
            self_f0=self.f0,
            self_sp=self.sp,
            self_ap=self.ap,
        )
        # --------------------------------------

        # 先頭ノートの場合は何もせず代入
        if long_f0.size == 0:
            long_f0 = self.f0
            long_sp = self.sp
            long_ap = self.ap
        # 既存の特徴量がある場合はオーバーラップさせる
        else:
            # 既存特徴量に新規ノートの特徴量を結合する
            long_f0 = overlap_f0(long_f0, self.f0, n_overlap_frames, crossfade_shape='linear')
            long_sp = overlap_sp(long_sp, self.sp, n_overlap_frames, crossfade_shape=None)
            long_ap = overlap_ap(long_ap, self.ap, n_overlap_frames, crossfade_shape='linear')
        # 追記後の特徴量を保存
        self.f0_appended = long_f0.astype(self.feature_dtype)
        self.ap_appended = long_ap.astype(self.feature_dtype)
        self.sp_appended = long_sp.astype(self.feature_dtype)
        # デバッグ出力 --------------------------
        self.logger.debug('Features after overlap:')
        self.debug_features(
            long_f0=long_f0,
            long_sp=long_sp,
            long_ap=long_ap,
        )
        # --------------------------------------

    # MARK: synthesize
    def synthesize(self) -> None:
        """WORLD特徴量からwavを合成して出力する。

        Todo:
            WORLD 特徴量を.wav 拡張子で出力するオプションを追加する (.npz はUTAUが自動で消してくれないため)。
            もしくは、エンジン一括実行を行うツールで、レンダリング開始前に .npz を消す処理を追加する。

        """  # noqa: E501
        # append された特徴量が揃っていることを確認する
        if self.f0_appended is None or self.sp_appended is None or self.ap_appended is None:
            msg = 'f0_appended, sp_appended, or ap_appended is None. Call append() first.'
            self.logger.error(msg)
            raise ValueError(msg)

        # ボコーダーモデルを使用しない場合
        if self.use_vocoder_model is False:
            # wav 生成
            wav = world_to_waveform(
                self.f0_appended,
                self.sp_appended,
                self.ap_appended,
                sample_rate=self.internal_sample_rate,
                frame_period=self.frame_period,
            )  # internal_sample_rate

        # ボコーダーモデルを使用する場合
        elif self.use_vocoder_model is True:
            # vocoder model 関連の引数が全て揃っていることを確認
            if (
                self.vocoder_model is None
                or self.vocoder_in_scaler is None
                or self.vocoder_config is None
            ):
                msg = 'vocoder_model, vocoder_in_scaler, or vocoder_config is None.'
                raise ValueError(msg)
            # nnsvs のボコーダーモデルを使って wav 生成
            wav = world_to_nnsvs_to_waveform(
                device=self.device,
                f0=self.f0_appended,
                sp=self.sp_appended,
                ap=self.ap_appended,
                vocoder_model=self.vocoder_model,
                vocoder_config=self.vocoder_config,
                vocoder_in_scaler=self.vocoder_in_scaler,
                vocoder_frame_period=self.vocoder_frame_period,
                use_world_codec=True,
                feature_type=self.vocoder_feature_type,
                vocoder_type=self.vocoder_type,
                vuv_threshold=self.vocoder_vuv_threshold,
            )  # vocoder_sample_rate (= internal_sample_rate)

        else:
            msg = f'Invalid use_vocoder_model: {self.use_vocoder_model}. Must be True or False.'
            self.logger.error(msg)
            raise ValueError(msg)

        self.waveform = wav

    def save_npz(self) -> None:
        """WORLD特徴量をnpzファイルで保存する。"""
        # append された特徴量が揃っていることを確認する
        if self.f0_appended is None or self.sp_appended is None or self.ap_appended is None:
            msg = 'f0_appended, sp_appended, or ap_appended is None. Call append() first.'
            self.logger.error(msg)
            raise ValueError(msg)
        # npzファイルに保存
        world_to_npzfile(
            self.f0_appended,
            self.sp_appended,
            self.ap_appended,
            self.internal_sample_rate,
            self.output_npz,
        )
        self.logger.info('Saved WORLD features to: %s', self.output_npz)

    def save_wav(self) -> None:
        """wavファイルを保存する。"""
        # waveform が生成されていることを確認する
        if self.waveform is None:
            msg = 'waveform is None. Call synthesize() first.'
            self.logger.error(msg)
            raise ValueError(msg)
        wav: np.ndarray = copy(self.waveform)

        # wavform の長さを丸め誤差分だけ補正する ----------------------------------------------
        # self.carryover_error が正の場合、生成した波形が目標より短いので、ゼロパディング必要。
        # self.carryover_error が負の場合、生成した波形が目標より長いので、切り詰め必要。
        n_compensation_samples = round(self.carryover_error / 1000 * self.internal_sample_rate)
        self.logger.debug('n_compensation_samples: %d', n_compensation_samples)
        self.logger.debug('waveform.shape before compensation: %s', wav.shape)
        # wav が目標よりも短い場合はゼロパディングする。
        if n_compensation_samples > 0:
            wav = np.pad(wav, (0, n_compensation_samples))  # internal_sample_rate
        # wav が目標よりも長い場合は切り詰める。
        elif n_compensation_samples < 0:
            wav = wav[:n_compensation_samples]  # internal_sample_rate
        self.logger.debug('waveform.shape after compensation: %s', wav.shape)
        # -------------------------------------------------------------------------------------
        # wav ファイルに保存
        waveform_to_wavfile(
            wav,
            self.output_wav,
            original_sample_rate=self.internal_sample_rate,
            target_sample_rate=self.target_sample_rate,
            resample_type=self.resample_type,
            dtype=wav.dtype,
        )
        self.logger.info('Saved wav to: %s', self.output_wav)


# MARK: main_wavtool
def main_wavtool() -> None:
    """実行引数を展開して wavtool としてスタンドアロン動作させる"""
    logger = setup_logger()
    parser = argparse.ArgumentParser(description='UTAU wavtool crossfading WORLD features')
    parser.add_argument('output', help='output wav path', type=str)
    parser.add_argument('input', help='input wav path', type=str)
    parser.add_argument('stp', help='start offset of wav', type=float)
    parser.add_argument('length', help='append length(ms)', type=str)
    parser.add_argument(
        'envelope',
        nargs='*',
        type=float,
        help=(
            'envelope pattern '
            "'p1 p2' "
            "or 'p1 p2 p3 v1 v2 v3 v4 ove' "
            "or 'p1 p2 p3 v1 v2 v3 v4' "
            "or 'p1 p2 p3 v1 v2 v3 v4 ove p4' "
            "or 'p1 p2 p3 v1 v2 v3 v4 ove p4 p5 v5'"
        ),
    )
    # ボコーダーモデルを使用するか否か
    parser.add_argument(
        '--use_vocoder_model',
        help='Whether to use vocoder model for waveform synthesis. If False, use WORLD.',
        action='store_true',
        default=False,
    )

    # モデルを指定
    parser.add_argument(
        '--model_dir',
        help='Vocoder model directory (optional; required for neural network vocoder)',
        type=str,
        default=None,
    )
    # デバッグモード
    parser.add_argument(
        '--debug',
        help='Enable debug logging',
        action='store_true',
        default=False,
    )
    args = parser.parse_args()
    # デバッグモード
    if args.debug:
        logger.setLevel(logging.DEBUG)
    # 入出力wavパスをフルパスでデバッグ出力
    logger.debug('Output wav path: %s', Path(args.output).resolve())
    logger.debug('Input wav path: %s', Path(args.input).resolve())
    # length 文字列を float に変換
    length = str2float(args.length)
    # モデルロードを試みる
    if args.use_vocoder_model:
        if args.model_dir is None:
            msg = 'When --use_vocoder_model is specified, --model_dir must be provided.'
            logger.error(msg)
            raise ValueError(msg)
        vocoder_model, vocoder_in_scaler, vocoder_config = load_vocoder_model(args.model_dir)
    else:
        vocoder_model = None
        vocoder_in_scaler = None
        vocoder_config = None
    wavtool = NeuralNetworkWavTool(
        args.output,
        args.input,
        args.stp,
        length,
        args.envelope,
        logger=logger,
        use_vocoder_model=args.use_vocoder_model,
        vocoder_model=vocoder_model,
        vocoder_in_scaler=vocoder_in_scaler,
        vocoder_config=vocoder_config,
    )
    # wavtool で音声WORLD特徴量を結合
    wavtool.append()
    # wav データを生成
    wavtool.synthesize()
    # npz と wav ファイルを保存
    wavtool.save_npz()
    wavtool.save_wav()


if __name__ == '__main__':
    main_wavtool()

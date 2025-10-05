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
- 自身が 先頭ノート/中間ノート/最終ノート のいずれであるかはわからないため、wav ファイルは常に出力必要。
- wav ファイルを出力する際、既存の wav ファイルがある場合は、オーバーラップ時間を考慮して重ねる必要がある。

"""

import argparse
import logging
import sys
from functools import partial
from math import ceil
from pathlib import Path

import colored_traceback.auto  # noqa: F401
import numpy as np
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
    setup_logger,
)

DEFAULT_SAMPLE_RATE = 44100


def round_by_frame(x: float, frame_period: float) -> float:
    """frame_period に基づいて x を丸める。"""
    return round(x / frame_period) * frame_period


def extract_overlap(envelope: list[float]) -> float:
    """Envelope から ove だけを取得する。

    Args:
        envelope (list[float]): エンベロープ値のリスト

    """
    len_envelope = len(envelope)
    if len_envelope in [2, 7]:
        return 0.0
    if len_envelope in [8, 9, 11]:
        return envelope[7]
    msg = f'Invalid envelope length ({len_envelope}). The length must be 2, 7, 8, 9, or 11.'
    raise ValueError(msg)


def parse_envelope(
    envelope: list[float], rounded_length: float, frame_period: float
) -> tuple[list, list, float]:
    """Envelope のパターンを解析し、時刻のリストと音量のリストとoverlap時間を返す。

    Args:
        envelope (list[float]): エンベロープの値のリスト
        rounded_length (float): ノートの長さ(先行発声含む)(ms)。あらかじめ frame_period で丸めておく必要あり。
        frame_period (float)  : WORLD特徴量のフレーム周期(ms)

    Returns:
        tuple: (p, v, ove)
            p (list[float]): 音量制御の時刻のリスト(ms)。エンベロープが2点の場合空配列。
            v (list[int])  : 音量値のリスト。0-200の範囲であることを想定。
            ove (float)    : クロスフェード時間(ms)。エンベロープにoveがない場合は0を返す。

    ## エンベロープのパターン
    - 長さ2 : p1 p2
    - 長さ7 : p1 p2 p3 v1 v2 v3 v4
    - 長さ8 : p1 p2 p3 v1 v2 v3 v4 ove
    - 長さ9 : p1 p2 p3 v1 v2 v3 v4 ove p4
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

    """
    # エンベロープの要素数
    len_envelope = len(envelope)
    # エンベロープが2点のときは空配列を返す
    # TODO: 空配列でいいのか再検討(そのまま使って問題ない配列を返したい)
    if len_envelope == 2:
        return [], [], 0

    # 丸め関数を定義
    round_func = partial(round_by_frame, frame_period=frame_period)

    # エンベロープが2点以外で想定される点数のとき
    p_list: list[float]
    v_list: list[float]
    overlap: float
    # 長さ7の時は [p1, p2, p3, v1, v2, v3, v4] のみ
    if len_envelope == 7:
        p1, p2, p3 = envelope[0:3]
        v1, v2, v3, v4 = envelope[3:7]
        p_list = [
            0,
            round_func(p1),
            round_func(p1 + p2),
            rounded_length - round_func(p3),
            rounded_length,
        ]
        v_list = [0, v1, v2, v3, v4, 0]
        overlap = 0
    # 長さ8の時は overlap が追加される
    elif len_envelope == 8:
        # [p1, p2, p3, v1, v2, v3, v4, ove]
        p1, p2, p3 = envelope[0:3]
        v1, v2, v3, v4 = envelope[3:7]
        p_list = [
            0,
            round_func(p1),
            round_func(p1 + p2),
            rounded_length - round_func(p3),
            rounded_length,
        ]
        v_list = [0, v1, v2, v3, v4, 0]
        overlap = envelope[7]
    # 長さ9の時は p4 が追加される
    elif len_envelope == 9:
        # [p1, p2, p3, v1, v2, v3, v4, ove, p4]
        p1, p2, p3 = envelope[0:3]
        v1, v2, v3, v4 = envelope[3:7]
        overlap = envelope[7]
        p4 = envelope[8]
        p_list = [
            0,
            round_func(p1),
            round_func(p1 + p2),
            rounded_length - round_func(p4 + p3),
            rounded_length - round_func(p4),
            rounded_length,
        ]
        v_list = [0, v1, v2, v3, v4, 0]
    # 長さが11の時は p5, v5 が追加される
    elif len_envelope == 11:
        # [p1, p2, p3, v1, v2, v3, v4, ove, p4, p5, v5]
        p1, p2, p3 = envelope[0:3]
        v1, v2, v3, v4 = envelope[3:7]
        overlap = envelope[7]
        p4 = envelope[8]
        p5 = envelope[9]
        v5 = envelope[10]
        # NOTE: p5 の位置は p2 と p3 の間であることに注意!
        p_list = [
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
        msg = f'Invalid envelope length ({len_envelope}). The length must be 2, 7, 8, 9, or 11.'
        raise ValueError(msg)

    # p_list が昇順になっていない場合はエラー
    if p_list != sorted(p_list):
        msg = f'p_list must be in ascending order, but got {p_list}.'
        raise ValueError(msg)

    return p_list, v_list, overlap


def str2float(length: float | str) -> float:
    """UTAUのLength文字列をfloatに変換します。

    NOTE: Copied function from PyWavTool.PyWavTool.length_string.str2float by delta-kuro.

    Parameters
    ----------
    length :str
        length文字列

    Returns
    -------
    length :float


    Notes
    -----
    | lengthは以下のいずれかの形で与えられます。
    | tick@tempo
    | tick@tempo+delta
    | tick@tempo-delta

    | 戻り値の計算は以下の通りです。
    | 1拍あたりのms = 60*1000 / tempo
    | 1tickあたりのms = 1拍あたりのms / 480
    | length = 1tickあたりのms * tick +(-) delta

    """
    if isinstance(length, float):
        return length
    if isinstance(length, str):
        temp: list[str] = length.split('@')
        tempo: float
        delta: float = 0
        tick: int = int(temp[0])
        if '+' in temp[1]:
            tempo = float(temp[1].split('+')[0])
            delta = float(temp[1].split('+')[1])
        elif '-' in temp[1]:
            tempo = float(temp[1].split('-')[0])
            delta = -float(temp[1].split('-')[1])
        else:
            tempo = float(temp[1])
        return 60000 / tempo / 480 * tick + delta
    # float でも str でもない場合はエラー
    msg = f'length must be float or str, but got {type(length)}'
    raise TypeError(msg)


# MARK: NeuralNetworkWavTool
class NeuralNetworkWavTool:
    """WAV出力の代わりに WORLD の特徴量を出力するのに用いる。

    Args:
        output: 出力ファイルのパス(拡張子なし)
        export_wav: WAVファイルを出力するか否か
        export_features: WORLD特徴量を npz ファイルで出力するか否か
        frame_period: WORLD特徴量のフレーム周期 (ms)

    ## PyWavTool.WavTool からの変更点
    - whd と dat を使用しない

    ### ファイル入出力について
    入力においては、npz がある場合は高速化のため npz を wav の代わりに優先的に読み込む。
    出力においては、npz を優先する場合でも常に wav 生成をしなければならない。
    NOTE: UTAU から呼び出される場合は実行中のノートが 最初/途中/最後 のいずれであるか判別できないので成果物は常に wav が必要。

    ### キャッシュの取り扱い
    - WAVキャッシュを使用する場合、WORLD 特徴量に変換してから append する。
    - NPZキャッシュを使用する場合、NPZファイルから直接特徴量を読み込んで append する。

    ### 内部データの取り扱い
    - self.dat は常に WORLD 特徴量を保持する。output のときだけ wav に変換する。
    - PyWavTool.WavTool._dat (waveform) の要素数はサンプルレートに応じた長さ (秒数*sample_rate) だったが、
      WorldFeatureWavTool.self._dat (WORLD特徴量) の要素数は frame_period に応じた長さ (秒数/frame_period*1000) になることに注意。

    TODO: 音量ノーマライズの際に WORLD 特徴量にノーマライズをかける方法を検討する。いったんWAVに変換して係数を算出する?

    """

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
    original_sample_rate: int  # 入力wavのサンプルレート [Hz]
    internal_sample_rate: int  # 内部処理のサンプルレート [Hz]
    target_sample_rate: int  # 出力wavのサンプルレート [Hz]
    resample_type: str  # リサンプリングの種類
    # 音量エンベロープ関連
    envelope_p: list[float]  # 音量エンベロープの時刻のリスト [ms]
    envelope_v: list[int]  # 音量エンベロープの音量値のリスト(0-100-200) [-]
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
    _residual_error: float  # 丸め誤差 [ms]

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
        residual_error: float = 0.0,  # このノート以前の時刻丸め誤差
        vocoder_model: torch.nn.Module | None = None,
        vocoder_in_scaler: StandardScaler | None = None,
        vocoder_config: DictConfig | ListConfig | None = None,
        vocoder_type: str = 'usfgan',
        vocoder_feature_type: str = 'world',
        vocoder_vuv_threshold: float = 0.5,
        vocoder_frame_period: int = 5,
        target_sample_rate: int | None = None,
        resample_type: str = 'soxr_vhq',
    ) -> None:
        """NeuralNetworkWavTool のコンストラクタ"""
        self.logger = logger or setup_logger(level=logging.INFO)
        self.input_wav = Path(input_wav)
        self.input_npz = Path(input_wav).with_suffix('.npz')
        self.output_wav = Path(output_wav)
        self.output_npz = Path(output_wav).with_suffix('.npz')
        self.frame_period = frame_period
        self.stp = stp
        # length と _residual_error を初期化
        self.__init_length(length, extract_overlap(envelope), residual_error)
        # sample_rate, f0, sp, ap を初期化
        self.__init_features()

        # デバッグ出力: f0, sp, ap のshape,min,maxを確認
        self.logger.debug('Initial features:')
        self.debug_features(f0=self.f0, sp=self.sp, ap=self.ap)

        # envelope_p, envelope_v, overlap を初期化
        self.__init_envelope(envelope)
        # デバイス設定
        self.device = get_device()
        # ボコーダー関連の設定
        self.use_vocoder_model = use_vocoder_model
        self.vocoder_type = vocoder_type
        self.vocoder_feature_type = vocoder_feature_type
        self.vocoder_vuv_threshold = vocoder_vuv_threshold
        self.vocoder_frame_period = vocoder_frame_period
        self.resample_type = resample_type

        # frame_period と vocoder_frame_period が異なる場合は警告を出す
        if self.frame_period != self.vocoder_frame_period:
            self.logger.error(
                'frame_period (%d ms) and vocoder_frame_period (%d ms) are different. This may result in unexpected behavior.',
                self.frame_period,
                self.vocoder_frame_period,
            )
        # use_vocoder_model が True の時はボコーダーモデルを代入する
        if use_vocoder_model is True:
            # vocoder model 関連の引数が全て揃っていることを確認
            if vocoder_model is None or vocoder_in_scaler is None or vocoder_config is None:
                msg = 'When use_vocoder_model is True, vocoder_model, vocoder_in_scaler, and vocoder_config must be provided.'
                raise ValueError(msg)
            self.vocoder_model = vocoder_model
            self.vocoder_in_scaler = vocoder_in_scaler
            self.vocoder_config = vocoder_config
            self.logger.info('Using vocoder model: %s', self.use_vocoder_model)
            # サンプルレート初期化
            self.internal_sample_rate = self.vocoder_config.data.sample_rate
            self.target_sample_rate = target_sample_rate or self.internal_sample_rate
        # use_vocoder_model が False の場合
        else:
            # use_vocoder_model が False なのに vocoder が指定されているときは警告を出す
            if (vocoder_model, vocoder_in_scaler, vocoder_config) != (None, None, None):
                self.logger.warning(
                    'use_vocoder_model is False, but [vocoder_model, vocoder_in_scaler, or vocoder_config] are provided. They will be ignored.',
                )
            self.vocoder_model = None
            self.vocoder_in_scaler = None
            self.vocoder_config = None
            self.logger.info('Not using vocoder model.')
            # サンプルレート初期化
            self.internal_sample_rate = self.original_sample_rate
            self.target_sample_rate = target_sample_rate or self.internal_sample_rate

        # 出力フォルダが存在しなければ作成
        Path(output_wav).parent.mkdir(parents=True, exist_ok=True)

    @property
    def residual_error(self) -> float:
        """このノート以降の length の丸め誤差 [ms]"""
        return self._residual_error

    @property
    def vocoder_sample_rate(self) -> int:
        """ボコーダーモデルのwav出力サンプリング周波数"""
        if self.vocoder_config is None:
            msg = 'vocoder_config is None. vocoder_model must be loaded first.'
            raise ValueError(msg)
        return self.vocoder_config.data.sample_rate

    def __init_length(
        self, original_length: float, original_overlap: float, residual_error: float
    ):
        """self.length と self._residual_error を初期化する。

        Args:
            original_length (float): 元の長さ [ms]
            original_overlap (float): 元のオーバーラップ [ms]
            residual_error (float): このノート以前の丸め誤差 [ms]

        """
        rounded_overlap = round_by_frame(original_overlap, self.frame_period)
        overlap_error = original_overlap - rounded_overlap
        # 以前の丸め誤差を考慮して length を調整する
        adjusted_length = original_length + residual_error - overlap_error
        # frame_period に基づいて length を丸める
        rounded_length = round(adjusted_length / self.frame_period) * self.frame_period
        # 新しい丸め誤差を計算する
        new_residual_error = adjusted_length - rounded_length
        self.logger.debug('residual_error (before): %.3f [ms]', residual_error)
        self.logger.debug('original_overlap: %.3f [ms]', original_overlap)
        self.logger.debug('rounded_overlap: %.3f [ms]', rounded_overlap)
        self.logger.debug('overlap_error: %.3f [ms]', overlap_error)
        self.logger.debug('original_length: %.3f [ms]', original_length)
        self.logger.debug('adjusted_length: %.3f [ms]', adjusted_length)
        self.logger.debug('rounded_length: %.3f [ms]', rounded_length)
        self.logger.debug('new_residual_error (after): %.3f [ms]', new_residual_error)
        self.length = rounded_length
        self._residual_error = new_residual_error

    def __init_features(self, default_sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        """self.f0, self.sp, self.ap, self.sample_rate を初期化する。

        入力wavまたはnpzを読み込み、WORLD特徴量に変換して self.f0, self.sp, self.ap にセットする。
        npzが存在する場合はnpzを優先的に読み込む。
        """
        # wav と npz が両方存在する場合、wav からサンプルレートを取得し、npz から特徴量を取得する。
        if self.input_wav.exists() and self.input_npz.exists():
            waveform, sample_rate, _ = wavfile_to_waveform(self.input_wav)
            self.original_sample_rate = sample_rate
            self.f0, self.sp, self.ap = npzfile_to_world(self.input_npz)
        # wav のみ存在する場合、を読み込んでサンプルレートと waveform を取得する。sample_rate は 必須。
        elif self.input_wav.exists():
            waveform, sample_rate, _ = wavfile_to_waveform(self.input_wav)
            self.original_sample_rate = sample_rate
            self.f0, self.sp, self.ap = waveform_to_world(
                waveform,
                self.original_sample_rate,
                frame_period=self.frame_period,
            )
        # npz のみ存在する場合、npz から特徴量を取得する。sample_rate は default_sample_rate に設定する。
        elif self.input_npz.exists():
            self.original_sample_rate = default_sample_rate
            self.f0, self.sp, self.ap = npzfile_to_world(self.input_npz)
        # wav と npz が両方とも存在しない場合は無音特徴量を使用する。
        else:
            self.original_sample_rate = default_sample_rate
            msg = f'Input file not found: {self.input_wav} or {self.input_npz}. Using silent features.'
            self.logger.warning(msg, stacklevel=1)
            n_frames = ceil((self.length + self.stp) / self.frame_period)
            dtype = np.float64
            self.f0 = np.zeros((n_frames,), dtype=dtype)
            self.sp = np.zeros((n_frames, 1025), dtype=dtype)
            self.ap = np.ones((n_frames, 1025), dtype=dtype)

    def __init_envelope(self, envelope: list[float]) -> None:
        """Envelope を解析し、self.envelope_p, self.envelope_v, self.overlap を初期化する。

        Args:
            envelope (list[float]): エンベロープの値のリスト

        """
        p, v, ove = parse_envelope(envelope, self.length, self.frame_period)
        self.logger.debug('Parsed envelope:')
        self.logger.debug('  p  : %s', p)
        self.logger.debug('  v  : %s', v)
        self.logger.debug('  ove: %s', ove)
        self.envelope_p = p
        self.envelope_v = v
        self.overlap = ove

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

        TODO: 音量エンベロープの時刻と音量値に基づいて、spectrogram の各フレームに対して音量調整を行う。
        """
        self.logger.debug('envelope_p: %s', self.envelope_p)
        # エンベロープが2点以下の場合は何もしない
        if len(self.envelope_p) < 2:
            return
        # エンベロープが3点以上の場合は音量エンベロープを適用する
        n_frames = self.f0.shape[0]
        x = np.arange(n_frames)
        # 時刻をフレーム単位に変換
        xp = [round(p / self.frame_period) for p in self.envelope_p]
        # 音量値(0-200)を0-2に正規化 (余った v は無視)
        fp = [v / 100.0 for v in self.envelope_v[: len(xp)]]
        # 音量エンベロープを計算
        volume_envelope = np.interp(x, xp, fp)
        # 音量エンベロープを x 倍するには sp を x^2 倍する必要がある。
        # sp, ap に音量エンベロープを適用する。f0 は何もしない(appendのときにクロスフェード処理する)。
        self.sp *= volume_envelope[:, np.newaxis] ** 2

    def _apply_all(self) -> None:
        """self.f0, self.sp, self.ap に stp, length, envelope を適用する。

        - stp, length に基づいて特徴量をクロップする。
        - envelope に基づいて音量調整を行う。

        self.f0, self.sp, self.ap に音量エンベロープを適用する。
        音量エンベロープの時刻と音量値に基づいて、f0, sp, ap の各フレームに対して音量調整を行う。
        """
        length_by_frame = round(self.length / self.frame_period)
        if length_by_frame <= 0:
            msg = f'Invalid length: {self.length} ms. Length must be greater than 0 ms.'
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
        long_f0, long_sp, long_ap = (
            npzfile_to_world(self.output_npz)
            if self.output_npz.exists()
            else (np.array([]), np.array([[]]), np.array([[]]))
        )
        # クロップしたのちエンベロープを適用する
        self._apply_all()
        # overlap をフレーム数に変換
        overlap_frames = round(self.overlap / self.frame_period)
        self.logger.info('overlap_frames: %s', overlap_frames)

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
            long_f0 = overlap_f0(long_f0, self.f0, overlap_frames, crossfade_shape='linear')
            long_sp = overlap_sp(long_sp, self.sp, overlap_frames, crossfade_shape=None)
            long_ap = overlap_ap(long_ap, self.ap, overlap_frames, crossfade_shape='linear')
        # 追記後の特徴量を保存
        self.f0_appended = long_f0
        self.ap_appended = long_ap
        self.sp_appended = long_sp
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

        """
        # append された特徴量が揃っていることを確認する
        if self.f0_appended is None or self.sp_appended is None or self.ap_appended is None:
            msg = 'f0_appended, sp_appended, or ap_appended is None. Call append() first.'
            raise ValueError(msg)

        # npzファイルに書き出す
        world_to_npzfile(
            self.f0_appended,
            self.sp_appended,
            self.ap_appended,
            self.output_npz,
            compress=False,
        )

        # 入力ファイルのサンプルレートを取得
        input_sample_rate = self.original_sample_rate
        # ボコーダーモデルを使用しない場合
        if self.use_vocoder_model is False:
            output_sample_rate = input_sample_rate
            # wav 生成
            wav = world_to_waveform(
                self.f0_appended,
                self.sp_appended,
                self.ap_appended,
                input_sample_rate,
                frame_period=self.frame_period,
            )
        # ボコーダーモデルを使用する場合
        elif self.use_vocoder_model is True:
            output_sample_rate = self.vocoder_sample_rate
            # vocoder model 関連の引数が全て揃っていることを確認
            if (
                self.vocoder_model is None
                or self.vocoder_in_scaler is None
                or self.vocoder_config is None
            ):
                msg = 'vocoder_model, vocoder_in_scaler, or vocoder_config is None.'
                raise ValueError(msg)
            # wav 生成
            wav = world_to_nnsvs_to_waveform(
                device=self.device,
                f0=self.f0,
                sp=self.sp,
                ap=self.ap,
                target_sample_rate=output_sample_rate,
                vocoder_model=self.vocoder_model,
                vocoder_config=self.vocoder_config,
                vocoder_in_scaler=self.vocoder_in_scaler,
                vocoder_frame_period=self.vocoder_frame_period,
                use_world_codec=True,
                feature_type=self.vocoder_feature_type,
                vocoder_type=self.vocoder_type,
                vuv_threshold=self.vocoder_vuv_threshold,
                resample_type=self.resample_type,
            )
        else:
            msg = f'Invalid use_vocoder_model: {self.use_vocoder_model}. Must be True or False.'
            raise ValueError(msg)

        # wavform の長さを丸め誤差分だけ補正する
        n_compensation_samples = round(self._residual_error / 1000 * output_sample_rate)
        self.logger.debug('n_compensation_samples: %d', n_compensation_samples)
        self.logger.debug('waveform.shape before compensation: %s', wav.shape)
        # wav が目標よりも短い場合はゼロパディングする。
        if n_compensation_samples > 0:
            wav = np.pad(wav, (0, n_compensation_samples))
        # wav が目標よりも長い場合は切り詰める。
        elif n_compensation_samples < 0:
            wav = wav[:n_compensation_samples]
        self.logger.debug('waveform.shape after compensation: %s', wav.shape)

        # wavファイルに書き出す。この時点で既に output_sample_rate にリサンプリング済み。
        if self.use_vocoder_model:
            waveform_to_wavfile(wav, self.output_wav, output_sample_rate, output_sample_rate)
        else:
            waveform_to_wavfile(wav, self.output_wav, output_sample_rate, output_sample_rate)


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
    # length 文字列を float に変換
    length = str2float(args.length)
    # モデルロードを試みる
    if args.use_vocoder_model:
        if args.model_dir is None:
            msg = 'When --use_vocoder_model is specified, --model_dir must be provided.'
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
    # wav ファイルを生成
    wavtool.synthesize()


if __name__ == '__main__':
    main_wavtool()

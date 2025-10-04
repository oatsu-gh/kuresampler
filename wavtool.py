# Copyright (c) 2025 oatsu
"""
Resampler classes

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
- 自身が 先頭ノート/中間ノート/最終ノート のいずれであるかはわからないため、wav ファイルは常に出力する必要がある。
- wav ファイルを出力する際、既存の wav ファイルがある場合は、オーバーラップ時間を考慮して重ねる必要がある。

"""

import argparse
import logging
import sys
from pathlib import Path
from warnings import warn

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
    world_to_npzfile,
    world_to_waveform,
)
from util import (
    crossfade_world_feature,
    overlap_world_feature,
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
    envelope: list[float], length: float, frame_period: float
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
    # TODO: p_list が昇順になっていない場合は自動修正する。昇順になるように並べ替えるかクリッピングする。

    return p_list, v_list, overlap


def str2float(length: float | str) -> float:
    """
    UTAUのLength文字列をfloatに変換します。
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

    input_wav: Path  # 入力wavのパス
    input_npz: Path  # 入力npzのパス
    output_wav: Path  # 出力wavのパス
    output_npz: Path  # 出力npzのパス
    stp: float  # 入力wavの先頭のオフセット [ms]
    length: float  # 追記したい音声長さ [ms]
    frame_period: int  # WORLD特徴量のフレーム周期 [ms]
    sample_rate: int  # 入出力wavのサンプルレート [Hz]
    f0: np.ndarray  # f0 (WORLD特徴量 F0)
    sp: np.ndarray  # sp (WORLD特徴量 Spectrogram)
    ap: np.ndarray  # ap (WORLD特徴量 Aperiodicity)
    envelope_p: list[float]  # 音量エンベロープの時刻のリスト [ms]
    envelope_v: list[int]  # 音量エンベロープの音量値のリスト(0-100-200) [-]
    overlap: float  # クロスフェード時間 [ms]
    vocoder_model: torch.nn.Module | None = None  # Vocoder model
    vocoder_in_scaler: StandardScaler | None = None  # Vocoder input scaler
    vocoder_config: ListConfig | DictConfig | None = None  # Vocoder config
    logger: logging.Logger

    def __init__(
        self,
        output_wav: Path | str,
        input_wav: Path | str,
        stp: float,
        length: float,
        envelope: list[float],
        *,
        frame_period: int = 5,
        logger: logging.Logger | None = None,
    ) -> None:
        self.input_wav = Path(input_wav)
        self.input_npz = Path(input_wav).with_suffix('.npz')
        self.output_wav = Path(output_wav)
        self.output_npz = Path(output_wav).with_suffix('.npz')
        self.frame_period = frame_period
        self.stp = stp
        self.length = length
        self.logger = logger or setup_logger(level=logging.INFO)
        # sample_rate, f0, sp, ap を初期化
        self.__init_features()
        # envelope_p, envelope_v, overlap を初期化
        self.__init_envelope(envelope)
        # 出力フォルダが存在しなければ作成
        Path(output_wav).parent.mkdir(parents=True, exist_ok=True)

    def __init_features(self, default_sample_rate: int = 44100) -> None:
        """self.f0, self.sp, self.ap, self.sample_rate を初期化する。

        入力wavまたはnpzを読み込み、WORLD特徴量に変換して self.f0, self.sp, self.ap にセットする。
        npzが存在する場合はnpzを優先的に読み込む。
        """
        # まずは wav を読み込んでサンプルレートと waveform を取得する。sample_rate は 必須。
        if self.input_wav.exists():
            waveform, sample_rate, _ = wavfile_to_waveform(self.input_wav)
            self.sample_rate = sample_rate
        # wav が存在しない場合はサンプルレートを 44100 に設定する。
        else:
            self.sample_rate = default_sample_rate

        # npz が存在する場合は優先的に読み込んで特徴量を取得する
        if self.input_npz.exists():
            self.f0, self.sp, self.ap = npzfile_to_world(self.input_npz)
        # npz が存在しない場合は wav から特徴量を抽出する
        elif self.input_wav.exists():
            self.f0, self.sp, self.ap = waveform_to_world(
                waveform, self.sample_rate, frame_period=self.frame_period
            )
        # wav と npz が両方とも存在しない場合は無音特徴量を登録する。
        else:
            dtype = np.float64
            msg = f'Input file not found: {self.input_wav} or {self.input_npz}'
            warn(msg, stacklevel=1)
            n_frames = round(self.length / self.frame_period)
            self.f0 = np.zeros((n_frames,), dtype=np.float64)
            self.sp = np.full(
                (n_frames, 1025), np.finfo(dtype).tiny, dtype=dtype
            )  # 1025 はデフォルト次元数
            self.ap = np.ones((n_frames, 1025), dtype=dtype)

    def __init_envelope(self, envelope: list[float]) -> None:
        """envelope を解析し、self.envelope_p, self.envelope_v, self.overlap を初期化する。

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
        # 既存の特徴量が空の場合はそのまま追加
        if long_f0.size == 0:
            long_f0 = self.f0
            long_sp = self.sp
            long_ap = self.ap
        # 既存の特徴量がある場合はオーバーラップさせる
        else:
            self.logger.debug('Before crossfade:')
            self.logger.debug('  long_f0.shape: %s', long_f0.shape)
            self.logger.debug('  long_sp.shape: %s', long_sp.shape)
            self.logger.debug('  long_ap.shape: %s', long_ap.shape)
            self.logger.debug('  self.f0.shape: %s', self.f0.shape)
            self.logger.debug('  self.sp.shape: %s', self.f0.shape)
            self.logger.debug('  self.ap.shape: %s', self.f0.shape)
            long_f0 = crossfade_world_feature(
                long_f0.reshape(-1, 1),
                self.f0.reshape(-1, 1),
                overlap_frames,
                crossfade_shape='linear',
                calc_in_log=True,
            ).reshape(-1)
            long_sp = overlap_world_feature(long_sp, self.sp, overlap_frames)
            long_ap = crossfade_world_feature(
                long_ap, self.ap, overlap_frames, crossfade_shape='linear'
            )
            self.logger.debug('After crossfade:')
            self.logger.debug('  long_f0.shape: %s', long_f0.shape)
            self.logger.debug('  long_sp.shape: %s', long_sp.shape)
            self.logger.debug('  long_ap.shape: %s', long_ap.shape)
        # npzファイルに書き出す
        world_to_npzfile(long_f0, long_sp, long_ap, self.output_npz, compress=False)
        # wavファイルに書き出す
        # Use self.sample_rate for input (waveform) sample rate,
        # and self.output_sample_rate for output sample rate.
        input_sample_rate = self.sample_rate
        output_sample_rate = getattr(self, 'output_sample_rate', self.sample_rate)
        waveform = world_to_waveform(
            long_f0,
            long_sp,
            long_ap,
            input_sample_rate,
            frame_period=self.frame_period,
        )
        waveform_to_wavfile(waveform, self.output_wav, input_sample_rate, output_sample_rate)


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
    wavtool = NeuralNetworkWavTool(
        args.output, args.input, args.stp, length, args.envelope, logger=logger
    )
    # wavtool で音声WORLD特徴量を結合
    wavtool.append()


if __name__ == '__main__':
    main_wavtool()

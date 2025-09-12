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

from pathlib import Path
from warnings import warn

import colored_traceback.auto  # noqa: F401
import numpy as np
import torch
from nnsvs.util import StandardScaler
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

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
from util import (  # noqa: F401
    crossfade_world_feature,
    load_vocoder_model,
    overlap_world_feature,
    setup_logger,
)


def parse_envelope(
    envelope: list[float], length: float, frame_period: float
) -> tuple[list, list, float]:
    """envelope のパターンを解析し、時刻のリストと音量のリストとoverlap時間を返す。

    Args:
        envelope (list[float]): エンベロープの値のリスト
        length (float): ノートの長さ(先行発声含む)(ms)
    Returns:
        tuple: (p, v, ove)
            p (list[float]): 音量制御の時刻のリスト(ms)。エンベロープが2点の場合空配列。
            v (list[int]): 音量値のリスト。0-200の範囲であることを想定。
            ove (float): クロスフェード時間(ms)。エンベロープにoveがない場合は0を返す。

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

    def _round_by_frame(x: float) -> float:
        """frame_period に基づいて x を丸める。"""
        return round(x / frame_period) * frame_period

    # 各値を frame_period に基づいて丸める (フェードインとフェードアウト時間をそろえるため)
    envelope = list(map(_round_by_frame, envelope))
    # エンベロープが2点以外で想定される点数のとき
    p_list: list[float]
    v_list: list[float]
    overlap: float
    # 長さ7の時は [p1, p2, p3, v1, v2, v3, v4] のみ
    if len_envelope == 7:
        p1, p2, p3 = envelope[0:3]
        v1, v2, v3, v4 = envelope[3:7]
        p_list = [0, p1, p1 + p2, length - p3, length]
        v_list = [0, v1, v2, v3, v4, 0]
        overlap = 0
    # 長さ8の時は overlap が追加される
    elif len_envelope == 8:
        p1, p2, p3 = envelope[0:3]
        v1, v2, v3, v4 = envelope[3:7]
        p_list = [0, p1, p1 + p2, length - p3, length]
        v_list = [0, v1, v2, v3, v4, 0]
        overlap = envelope[7]
    # 長さ9の時は p4 が追加される
    elif len_envelope == 9:
        p1, p2, p3 = envelope[0:3]
        v1, v2, v3, v4 = envelope[3:7]
        overlap = envelope[7]
        p4 = envelope[8]
        p_list = [0, p1, p1 + p2, length - p4 - p3, length - p4, length]
        v_list = [0, v1, v2, v3, v4, 0]
    # 長さが11の時は p5, v5 が追加される
    elif len_envelope == 11:
        p1, p2, p3 = envelope[0:3]
        v1, v2, v3, v4 = envelope[3:7]
        overlap = envelope[7]
        p4 = envelope[8]
        p5 = envelope[9]
        v5 = envelope[10]
        # NOTE: p5 の位置は p2 と p3 の間であることに注意!
        p_list = [0, p1, p1 + p2, p1 + p2 + p5, length - p4 - p3, length - p4, length]  # 絶対時刻
        v_list = [0, v1, v2, v5, v3, v4, 0]
    # それ以外の要素数はエラー
    else:
        msg = f'Invalid envelope length (len_envelope={len_envelope}). The length must be 2, 7, 8, 9, or 11.'
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


# MARK: WorldFeatureWavTool
class WorldFeatureWavTool:
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

    _input_wav: Path  # 入力wavのパス
    _input_npz: Path  # 入力npzのパス
    _output_wav: Path  # 出力wavのパス
    _output_npz: Path  # 出力npzのパス
    _stp: float  # 入力wavの先頭のオフセット [ms]
    _length: float  # 追記したい音声長さ [ms]
    _frame_period: int  # WORLD特徴量のフレーム周期 [ms]
    _sample_rate: int  # 入出力wavのサンプルレート [Hz]
    _f0: np.ndarray  # f0 (WORLD特徴量 F0)
    _sp: np.ndarray  # sp (WORLD特徴量 Spectrogram)
    _ap: np.ndarray  # ap (WORLD特徴量 Aperiodicity)
    _envelope_p: list[float]  # 音量エンベロープの時刻のリスト [ms]
    _envelope_v: list[int]  # 音量エンベロープの音量値のリスト(0-100-200) [-]
    _overlap: float  # クロスフェード時間 [ms]
    _vocoder_model: torch.nn.Module | None = None  # Vocoder model
    _vocoder_in_scaler: StandardScaler | None = None  # Vocoder input scaler
    _vocoder_config: ListConfig | DictConfig | None = None  # Vocoder config

    def __init__(
        self,
        output_wav: Path | str,
        input_wav: Path | str,
        stp: float,
        length: float,
        envelope: list[float],
        *,
        frame_period: int = 5,
    ) -> None:
        self._input_wav = Path(input_wav)
        self._input_npz = Path(input_wav).with_suffix('.npz')
        self._output_wav = Path(output_wav)
        self._output_npz = Path(output_wav).with_suffix('.npz')
        self._frame_period = frame_period
        self._stp = stp
        self._length = length
        # sample_rate, f0, sp, ap を初期化
        self.__init_features()
        # envelope_p, envelope_v, overlap を初期化
        self.__init_envelope(envelope)
        # 出力フォルダが存在しなければ作成
        Path(output_wav).parent.mkdir(parents=True, exist_ok=True)

    def __init_features(self) -> None:
        """self._f0, self._sp, self._ap, self._sample_rate を初期化する。

        入力wavまたはnpzを読み込み、WORLD特徴量に変換して self._f0, self._sp, self._ap にセットする。
        npzが存在する場合はnpzを優先的に読み込む。
        """
        # まずは wav を読み込んで sample_rate と waveform を取得する。sample_rate は 必須。
        if self._input_wav.exists():
            waveform, sample_rate, _ = wavfile_to_waveform(self._input_wav)
            self._sample_rate = sample_rate
            # npz が存在する場合は優先的に読み込んで特徴量を取得する
            if self._input_npz.exists():
                self._f0, self._sp, self._ap = npzfile_to_world(self._input_npz)
            # npz が存在しない場合は wav から特徴量を抽出する
            elif self._input_wav.exists():
                self._f0, self._sp, self._ap = waveform_to_world(
                    waveform, self._sample_rate, frame_period=self._frame_period
                )
        # wav が存在しない場合は無音の特徴量で初期化する。
        else:
            msg = f'Input file not found: {self._input_wav} or {self._input_npz}'
            warn(msg, stacklevel=1)
            self._sample_rate = 44100  # デフォルトのサンプルレート
            num_frames = round(self._length / self._frame_period)
            self._f0 = np.zeros((num_frames,), dtype=np.float64)
            self._sp = np.full(
                (num_frames, 1025), 1e-100, dtype=np.float64
            )  # 1025 はデフォルトの次元数
            self._ap = np.ones((num_frames, 1025), dtype=np.float64)

    def __init_envelope(self, envelope: list[float]) -> None:
        """envelope を解析し、self._envelope_p, self._envelope_v, self._overlap を初期化する。

        Args:
            envelope (list[float]): エンベロープの値のリスト
        """
        p, v, ove = parse_envelope(envelope, self._length, self._frame_period)
        self._envelope_p = p
        self._envelope_v = v
        self._overlap = ove

    def _apply_range(self) -> None:
        """self._f0, self._sp, self._ap に stp, length を適用する。

        stp, length に基づいて特徴量をクロップする。
        """
        length_by_frame = round(self._length / self._frame_period)
        # stp, length に基づいて特徴量をクロップする
        start_frame = round(self._stp / self._frame_period)
        end_frame = start_frame + length_by_frame
        self._f0 = self._f0[start_frame:end_frame]
        self._sp = self._sp[start_frame:end_frame, :]
        self._ap = self._ap[start_frame:end_frame, :]

    def _apply_envelope(self) -> None:
        """self._f0, self._sp, self._ap に音量エンベロープを適用する。
        TODO: 音量エンベロープの時刻と音量値に基づいて、f0, sp, ap の各フレームに対して音量調整を行う。
        """
        # エンベロープが2点以下の場合は何もしない
        if len(self._envelope_p) < 2:
            return
        # エンベロープが3点以上の場合は音量エンベロープを適用する
        num_frames = self._f0.shape[0]
        x = np.arange(num_frames)
        # 時刻をフレーム単位に変換
        xp = [round(p / self._frame_period) for p in self._envelope_p]
        # 音量値を0-1に正規化 (余った v は無視)
        fp = [v / 100.0 for v in self._envelope_v[: len(xp)]]
        # 音量エンベロープを計算
        volume_envelope = np.interp(x, xp, fp)
        # sp, ap に音量エンベロープを適用する。f0 は何もしない(appendのときにクロスフェード処理する)。
        self._sp *= volume_envelope[:, np.newaxis]
        self._ap *= volume_envelope[:, np.newaxis]

    def _apply_all(self) -> None:
        """self._f0, self._sp, self._ap に stp, length, envelope を適用する。

        - stp, length に基づいて特徴量をクロップする。
        - envelope に基づいて音量調整を行う。

        self._f0, self._sp, self._ap に音量エンベロープを適用する。
        音量エンベロープの時刻と音量値に基づいて、f0, sp, ap の各フレームに対して音量調整を行う。
        """
        length_by_frame = round(self._length / self._frame_period)
        if length_by_frame <= 0:
            msg = f'Invalid length: {self._length} ms. Length must be greater than 0 ms.'
            raise ValueError(msg)
        # クロップする
        self._apply_range()
        # 音量エンベロープを適用する
        self._apply_envelope()

    def append(self):
        """既存のnpzファイルを読み取って、それに書き込む。wav は全体を再計算して出力する。

        ノート数が多いほどWAV生成が重くなるので何とかしたい。
        """
        # 既存ファイルの特徴量を読み取る。なければ空の配列を取得する。
        long_f0, long_sp, long_ap = (
            npzfile_to_world(self._output_npz)
            if self._output_npz.exists()
            else (np.array([]), np.array([[]]), np.array([[]]))
        )
        # クロップしたのちエンベロープを適用する
        self._apply_all()
        # overlap をフレーム数に変換
        overlap_frames = round(self._overlap / self._frame_period)
        print(f'self._overlap: {self._overlap}')
        print(f'overlap_frames: {overlap_frames}')
        # 既存の特徴量が空の場合はそのまま追加
        if long_f0.size == 0:
            long_f0 = self._f0
            long_sp = self._sp
            long_ap = self._ap
        # 既存の特徴量がある場合はオーバーラップさせる
        else:
            long_f0 = crossfade_world_feature(
                long_f0.reshape(-1, 1),
                self._f0.reshape(-1, 1),
                overlap_frames,
                shape='linear',
                calc_in_log=True,
            ).reshape(-1)
            long_sp = overlap_world_feature(long_sp, self._sp, overlap_frames)
            long_ap = overlap_world_feature(long_ap, self._ap, overlap_frames)
        # npzファイルに書き出す
        world_to_npzfile(long_f0, long_sp, long_ap, self._output_npz, compress=False)
        # wavファイルに書き出す
        sample_rate = self._sample_rate
        waveform = world_to_waveform(
            long_f0,
            long_sp,
            long_ap,
            sample_rate,
            frame_period=self._frame_period,
        )
        waveform_to_wavfile(waveform, self._output_wav, sample_rate, sample_rate)

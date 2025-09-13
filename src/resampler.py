# Copyright (c) 2025 oatsu
"""
Resampler classes

PyRwu.resamp.Resamp を継承し、
WAVファイルの代わりにWORLD特徴量をファイルに出力する
クラス WorldFeatureResamp を定義する。
"""

import argparse
from copy import copy
from logging import Logger
from pathlib import Path

import colored_traceback.auto  # noqa: F401
import librosa
import numpy as np
import PyRwu as pyrwu  # noqa: N813
import torch
from nnsvs.gen import predict_waveform
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
from util import denoise_spike, get_device, load_vocoder_model, setup_logger


# TODO: WorldFeatureResamp を NeuralNetworkRender に統合する。
# MARK: WorldFeatureResamp
class WorldFeatureResamp(pyrwu.Resamp):
    """WAVファイルの代わりにWORLDの特徴量をファイルに出力するResampler

    PyRwu.resamp.Resamp からの変更点
    - __init__ で logger を必須化
    - export_features を追加
    - export_wav を追加
    """

    _export_wav: bool
    _export_features: bool

    def __init__(
        self,
        input_path: str,
        output_path: str,
        target_tone: str,
        velocity: int,
        flag_value: str = '',
        offset: float = 0,
        target_ms: float = 0,
        fixed_ms: float = 0,
        end_ms: float = 0,
        volume: int = 100,
        modulation: int = 0,
        tempo: str = '!120',
        pitchbend: str = '',
        *,
        logger: Logger | None = None,
        export_wav: bool,
        export_features: bool,
    ) -> None:
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            target_tone=target_tone,
            velocity=velocity,
            flag_value=flag_value,
            offset=offset,
            target_ms=target_ms,
            fixed_ms=fixed_ms,
            end_ms=end_ms,
            volume=volume,
            modulation=modulation,
            tempo=tempo,
            pitchbend=pitchbend,
            logger=logger,
        )
        logger = setup_logger() if logger is None else logger
        # WAVファイルを出力するか否か
        self._export_wav = export_wav
        # WORLD特徴量をファイル出力するか否か
        self._export_features = export_features

    @property
    def export_wav(self) -> bool:
        return self._export_wav

    @export_wav.setter
    def export_wav(self, value: bool) -> None:
        self._export_wav = value

    @property
    def export_features(self) -> bool:
        return self._export_features

    @export_features.setter
    def export_features(self, value: bool) -> None:
        self._export_features = value

    def return_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """WORLD特徴量を返す。resamp() より後に実行する想定。"""
        return copy(self.f0), copy(self.sp), copy(self.ap)

    def return_waveform(self) -> np.ndarray:
        """waveform を返す。resamp() より後に実行する想定"""
        return copy(self._output_data)

    def denoise_f0(self) -> None:
        """f0 のスパイクノイズを除去する。"""
        if self._f0 is not None:
            self._f0 = denoise_spike(self._f0)

    def resamp(self) -> None:
        """WAVファイルの代わりにWORLDの特徴量をファイルに出力する。"""
        self.parseFlags()
        self.getInputData()
        self.stretch()
        self.pitchShift()
        self.applyPitch()  # FIXME: 最初のピッチ点のところで相対f0が0になる不具合を直す
        # パラメータ確認 ---------------------------------------
        self.logger.debug('  input_path  : %s', self.input_path)
        self.logger.debug('  output_path : %s', self.output_path)
        self.logger.debug('  framerate   : %s', self.framerate)
        self.logger.debug('  t.shape     : %s', self.t.shape)
        self.logger.debug('  f0.shape    : %s', self.f0.shape)
        self.logger.debug('  sp.shape    : %s', self.sp.shape)
        self.logger.debug('  ap.shape    : %s', self.ap.shape)
        # ------------------------------------------------------
        # f0 のスパイクノイズを除去
        self.denoise_f0()
        # WORLD特徴量にフラグを適用したのち wavform を更新する。
        self.synthesize()
        # UST の音量を waveform に反映する。wav 出力しない場合は無駄な処理なのでskip。
        if self.export_wav:
            self.adjustVolume()
        # WAVファイル出力は必須ではないがテスト用に出力可能。
        if self.export_wav:
            self.output()
            self.logger.debug('Exported WAV file: %s', self.output_path)
        # WORLD 特徴量を npz ファイル出力する。
        if self.export_features:
            npz_path = Path(self.output_path).with_suffix('.npz')
            np.savez(npz_path, f0=self.f0, spectrogram=self.sp, aperiodicity=self.ap)
            self.logger.debug('Exported WORLD features (f0, sp, ap): %s', npz_path)


# MARK: NeuralNetworkResamp
class NeuralNetworkResamp(WorldFeatureResamp):
    """Neural NetworkによるWORLD特徴量のリサンプリングを行う。

    Args:
        vocoder_model_dir: The directory containing the vocoder model files.

    """

    _vocoder_model: torch.nn.Module
    _vocoder_model_dir: Path | str
    _vocoder_in_scaler: StandardScaler
    _vocoder_config: DictConfig | ListConfig
    _vocoder_type: str
    _vocoder_feature_type: str
    _vocoder_vuv_threshold: float
    _vocoder_frame_period: int
    _device: torch.device
    _resample_type: str

    def __init__(
        self,
        *args,
        vocoder_model_dir: Path | str,
        vocoder_type: str = 'usfgan',
        vocoder_feature_type: str = 'world',
        vocoder_vuv_threshold: float = 0.5,
        vocoder_frame_period: int = 5,
        resample_type: str = 'soxr_vhq',
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._vocoder_model_dir = vocoder_model_dir
        self._device = get_device()
        self._vocoder_model, self._vocoder_in_scaler, self._vocoder_config = load_vocoder_model(
            vocoder_model_dir,
            device=self._device,
        )
        self._vocoder_type = vocoder_type
        self._vocoder_feature_type = vocoder_feature_type
        self._vocoder_vuv_threshold = vocoder_vuv_threshold
        self._vocoder_frame_period = vocoder_frame_period
        self._resample_type = resample_type

    @property
    def vocoder_model(self) -> torch.nn.Module:
        """ボコーダーモデル"""
        return self._vocoder_model

    @property
    def vocoder_sample_rate(self) -> int:
        """ボコーダーモデルのwav出力サンプリング周波数"""
        return self._vocoder_config.data.sample_rate

    def synthesize(self) -> None:
        """Pyworld の代わりに vocoder model を用いてWORLD特徴量からwaveformを生成し、self._output_dataに代入する。"""
        for effect in pyrwu.settings.F0_EFFECTS:
            self._f0 = effect.apply(self)

        for effect in pyrwu.settings.SP_EFFECTS:
            self._sp = effect.apply(self)

        for effect in pyrwu.settings.AP_EFFECTS:
            self._ap = effect.apply(self)

        for effect in pyrwu.settings.WORLD_EFFECTS:
            self._f0, self._sp, self._ap = effect.apply(self)

        assert self._vocoder_model is not None  # 念のため確認

        # WORLD 特徴量を NNSVS 用に変換
        mgc, lf0, vuv, bap = world_to_nnsvs(self.f0, self.sp, self.ap, self.vocoder_sample_rate)
        # モデルに渡す用に特徴量をまとめる
        multistream_features = (mgc, lf0, vuv, bap)
        # print(mgc.shape, lf0.shape, vuv.shape, bap.shape)
        # waveformを生成
        wav = predict_waveform(
            device=self._device,
            multistream_features=multistream_features,
            vocoder=self._vocoder_model,
            vocoder_config=self._vocoder_config,
            vocoder_in_scaler=self._vocoder_in_scaler,
            sample_rate=self.vocoder_sample_rate,
            frame_period=self._vocoder_frame_period,
            use_world_codec=True,
            feature_type=self._vocoder_feature_type,
            vocoder_type=self._vocoder_type,
            vuv_threshold=self._vocoder_vuv_threshold,  # vuv 閾値設定はするけど使われないはず
        )
        # サンプリング周波数が異なる場合、UTAUの原音と同じになるようにリサンプリングする。
        if self.vocoder_sample_rate != self.framerate:
            wav = librosa.resample(
                wav,
                orig_sr=self.vocoder_sample_rate,  # ボコーダモデルが出力するサンプルレート
                target_sr=self.framerate,  # UTAUの原音のサンプルレート
                res_type=self._resample_type,
            )
        # 生成した波形を _output_data に代入
        self._output_data = wav

    def output(self) -> None:
        """生成した波形を出力する。"""
        if self._output_data is not None:
            pyrwu.wave_io.write(
                self.output_path,
                self._output_data,
                self.vocoder_sample_rate,
                pyrwu.settings.OUTPUT_BITDEPTH // 8,
            )

    def resamp(self) -> None:
        """Neural Networkを用いてWORLD特徴量をリサンプリングする。"""
        self.logger.info(
            'Resample using vocoder model: %s (%s)',
            self._vocoder_model_dir,
            type(self._vocoder_model),
        )
        self.parseFlags()  # フラグを取得
        self.getInputData()  # 原音WAVからWORLD特徴量を抽出
        self.stretch()  # 時間伸縮
        self.pitchShift()  # ピッチシフト
        self.applyPitch()  # ピッチベンド適用

        # パラメータ確認 ---------------------------------------
        self.logger.debug('  input_path  : %s', self.input_path)
        self.logger.debug('  output_path : %s', self.output_path)
        self.logger.debug('  framerate   : %s', self.framerate)
        self.logger.debug('  t.shape     : %s', self.t.shape)
        self.logger.debug('  f0.shape    : %s', self.f0.shape)
        self.logger.debug('  sp.shape    : %s', self.sp.shape)
        self.logger.debug('  ap.shape    : %s', self.ap.shape)
        # ------------------------------------------------------
        # f0 のスパイクノイズを除去
        self.denoise_f0()
        # NOTE: synthesize はオーバーライドされているので nnsvs を使って waveform 生成していることに注意
        self.synthesize()
        # WAVファイル出力は必須ではないがテスト用に出力可能。
        if self.export_wav:
            # UST の音量を waveform に反映
            self.adjustVolume()  # TODO: npz にも反映できるようにする。
            # WAV ファイル出力
            self.output()
            self.logger.debug('Exported WAV file: %s', self.output_path)
        # WORLD 特徴量を npz ファイル出力する。
        if self.export_features:
            npz_path = Path(self.output_path).with_suffix('.npz')
            np.savez(npz_path, f0=self.f0, spectrogram=self.sp, aperiodicity=self.ap)
            self.logger.debug('Exported WORLD features (f0, sp, ap): %s', npz_path)


def main_resampler() -> None:
    """実行引数を展開して Resamp インスタンスを生成し、resamp() を実行する。

    想定される引数の形式
    resampler.exe <input wavfile> <output file> <pitch_percent> <velocity> [<flags> [<offset> <length_require> [<fixed length> [<end_blank> [<volume> [<modulation> [<pich bend>...]]]]]]]

    """
    logger = setup_logger()

    # 引数を展開
    parser = argparse.ArgumentParser(description='WORLD feature resampler')
    # Positional arguments are inherently required; remove invalid required=True
    parser.add_argument('input_path', help='原音のファイル名', type=str)
    parser.add_argument('output_path', help='wavファイルの出力先パス', type=str)
    parser.add_argument(
        'target_tone',
        help='音高名(A4=440Hz)。半角上げは#もしくは♯半角下げはbもしくは♭で与えられます。',
        type=str,
    )
    parser.add_argument('velocity', help='子音速度', type=int)
    parser.add_argument(
        'flags',
        help='フラグ(省略可 default:"")。詳細は--show-flags参照',
        nargs='?',
        default='',
    )
    parser.add_argument(
        'offset',
        help='入力ファイルの読み込み開始位置(ms)(省略可 default:0)',
        nargs='?',
        default=0,
    )
    parser.add_argument(
        'target_ms',
        help='出力ファイルの長さ(ms)(省略可 default:0)UTAUでは通常50ms単位に丸めた値が渡される。',
        nargs='?',
        default=0,
    )
    parser.add_argument(
        'fixed_ms', help='offsetからみて通常伸縮しない長さ(ms)', nargs='?', default=0
    )
    parser.add_argument(
        'end_ms',
        help='入力ファイルの読み込み終了位置(ms)(省略可 default:0)'
        '正の数の場合、ファイル末尾からの時間'
        '負の数の場合、offsetからの時間',
        nargs='?',
        default=0,
    )
    parser.add_argument('volume', help='音量。0～200(省略可 default:100)', nargs='?', default=100)
    parser.add_argument(
        'modulation',
        help='モジュレーション。0～200(省略可 default:0)',
        nargs='?',
        default=0,
    )
    parser.add_argument(
        'tempo',
        help='ピッチのテンポ。数字の頭に!がついた文字列(省略可 default:"!120")',
        nargs='?',
        default='!120',
    )
    parser.add_argument(
        'pitchbend',
        help='ピッチベンド。(省略可 default:"")'
        '-2048～2047までの12bitの2進数をbase64で2文字の文字列に変換し、'
        '同じ数字が続く場合ランレングス圧縮したもの',
        nargs='?',
        default='',
    )
    args = parser.parse_args()

    # WorldFeatureResamp インスタンスを生成
    resamp = WorldFeatureResamp(
        input_path=args.input_path,
        output_path=args.output_path,
        target_tone=args.target_tone,
        velocity=args.velocity,
        flag_value=args.flags,
        offset=args.offset,
        target_ms=args.target_ms,
        fixed_ms=args.fixed_ms,
        end_ms=args.end_ms,
        volume=args.volume,
        modulation=args.modulation,
        pitchbend=','.join(args.pitchbend),
        logger=logger,
        export_wav=True,
        export_features=True,
    )
    # リサンプリングを実行
    resamp.resamp()


if __name__ == '__main__':
    main_resampler()

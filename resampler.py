# Copyright (c) 2025 oatsu
"""
Resampler classes

PyRwu.resamp.Resamp を継承し、
WAVファイルの代わりにWORLD特徴量をファイルに出力する
クラス WorldFeatureResamp を定義する。
"""

import argparse
import sys
from copy import copy
from logging import Logger
from pathlib import Path

import colored_traceback.auto  # noqa: F401
import librosa
import numpy as np
import PyRwu as pyrwu  # noqa: N813
import pyworld
import torch
from nnsvs.gen import predict_waveform
from nnsvs.util import StandardScaler
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))  # for local import

from convert import world_to_nnsvs
from util import denoise_spike, get_device, load_vocoder_model, setup_logger


# MARK: NeuralNetworkResamp
class NeuralNetworkResamp(pyrwu.Resamp):
    """Neural NetworkによるWORLD特徴量のリサンプリングを行う。

    Args:
        vocoder_model_dir: The directory containing the vocoder model files.
        use_vocoder_model: Whether to use vocoder model for waveform synthesis. If False, use WORLD.

    """

    _export_wav: bool
    _export_features: bool
    _use_vocoder_model: bool
    _vocoder_model: torch.nn.Module | None
    _vocoder_model_dir: Path | str | None
    _vocoder_in_scaler: StandardScaler | None
    _vocoder_config: DictConfig | ListConfig | None
    _vocoder_type: str
    _vocoder_feature_type: str
    _vocoder_vuv_threshold: float
    _vocoder_frame_period: int
    _device: torch.device
    _resample_type: str

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
        tempo: str | None = None,
        pitchbend: str = '',
        *,
        logger: Logger | None = None,
        export_wav: bool,
        export_features: bool,
        use_vocoder_model: bool = True,
        vocoder_model_dir: Path | str | None = None,
        vocoder_type: str = 'usfgan',
        vocoder_feature_type: str = 'world',
        vocoder_vuv_threshold: float = 0.5,
        vocoder_frame_period: int = 5,
        resample_type: str = 'soxr_vhq',
    ) -> None:
        self.logger = setup_logger() if logger is None else logger
        if tempo is None:
            self.logger.warning('Tempo is None, set to "!120" by default')
            tempo = '!120'

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
        # WAVファイルを出力するか否か
        self._export_wav = export_wav
        # WORLD特徴量をファイル出力するか否か
        self._export_features = export_features
        # ボコーダーモデルを使用するか否か
        self._use_vocoder_model = use_vocoder_model
        # フラグに 'e' (stretch) を追加して、原音WAVの伸縮をストレッチ式に強制する。
        self.__force_stretch()

        # デバイス設定
        self._device = get_device()

        # ボコーダー関連の設定
        self._vocoder_model_dir = vocoder_model_dir
        self._vocoder_type = vocoder_type
        self._vocoder_feature_type = vocoder_feature_type
        self._vocoder_vuv_threshold = vocoder_vuv_threshold
        self._vocoder_frame_period = vocoder_frame_period
        self._resample_type = resample_type

        # use_vocoder_model が True の時のみボコーダーモデルを読み込む
        if self._use_vocoder_model:
            if vocoder_model_dir is None:
                msg = 'vocoder_model_dir is required when use_vocoder_model is True'
                raise ValueError(msg)
            self._vocoder_model, self._vocoder_in_scaler, self._vocoder_config = (
                load_vocoder_model(
                    vocoder_model_dir,
                    device=self._device,
                )
            )
        else:
            self._vocoder_model = None
            self._vocoder_in_scaler = None
            self._vocoder_config = None

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

    @property
    def use_vocoder_model(self) -> bool:
        return self._use_vocoder_model

    @use_vocoder_model.setter
    def use_vocoder_model(self, value: bool) -> None:
        self._use_vocoder_model = value

    def __force_stretch(self) -> None:
        """原音WAVの伸縮をストレッチ式に強制する。

        note.flags に `e` を追加する。
        ただし、note.flags に `e` が既に存在する場合や、`l` (loop) が明示的に指定されている場合は skip。
        """
        if 'e' not in self._flag_value and 'l' not in self._flag_value:
            self._flag_value += 'e'

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

    @property
    def vocoder_model(self) -> torch.nn.Module | None:
        """ボコーダーモデル"""
        return self._vocoder_model

    @property
    def vocoder_sample_rate(self) -> int:
        """ボコーダーモデルのwav出力サンプリング周波数"""
        if self._vocoder_config is None:
            msg = 'vocoder_config is None. vocoder_model must be loaded first.'
            raise ValueError(msg)
        return self._vocoder_config.data.sample_rate

    def synthesize(self) -> None:
        """Pyworld または vocoder model を用いてWORLD特徴量からwaveformを生成し、self._output_dataに代入する。"""
        # DEBUG: --------------------------------------------------
        self.logger.debug('WORLD features before applying F0_EFFECTS-----------------------')
        self.logger.debug('  f0 (min, max): (%s, %s)', self.f0.min(), self.f0.max())
        self.logger.debug('  sp (min, max): (%s, %s)', self.sp.min(), self.sp.max())
        self.logger.debug('  ap (min, max): (%s, %s)', self.ap.min(), self.ap.max())
        for effect in pyrwu.settings.F0_EFFECTS:
            self._f0 = effect.apply(self)

        self.logger.debug('WORLD features after applying F0_EFFECTS-----------------------')
        self.logger.debug('  f0 (min, max): (%s, %s)', self.f0.min(), self.f0.max())
        self.logger.debug('  sp (min, max): (%s, %s)', self.sp.min(), self.sp.max())
        self.logger.debug('  ap (min, max): (%s, %s)', self.ap.min(), self.ap.max())

        for effect in pyrwu.settings.SP_EFFECTS:
            self._sp = effect.apply(self)
        self.logger.debug('WORLD features after applying SP_EFFECTS-----------------------')
        self.logger.debug('  f0 (min, max): (%s, %s)', self.f0.min(), self.f0.max())
        self.logger.debug('  sp (min, max): (%s, %s)', self.sp.min(), self.sp.max())
        self.logger.debug('  ap (min, max): (%s, %s)', self.ap.min(), self.ap.max())

        for effect in pyrwu.settings.AP_EFFECTS:
            self._ap = effect.apply(self)
        self.logger.debug('WORLD features after applying AP_EFFECTS-----------------------')
        self.logger.debug('  f0 (min, max): (%s, %s)', self.f0.min(), self.f0.max())
        self.logger.debug('  sp (min, max): (%s, %s)', self.sp.min(), self.sp.max())
        self.logger.debug('  ap (min, max): (%s, %s)', self.ap.min(), self.ap.max())

        for effect in pyrwu.settings.WORLD_EFFECTS:
            self._f0, self._sp, self._ap = effect.apply(self)

        self.logger.debug('WORLD features after applying WORLD_EFFECTS--------------------')
        self.logger.debug('  f0 (min, max): (%s, %s)', self.f0.min(), self.f0.max())
        self.logger.debug('  sp (min, max): (%s, %s)', self.sp.min(), self.sp.max())
        self.logger.debug('  ap (min, max): (%s, %s)', self.ap.min(), self.ap.max())

        if self._use_vocoder_model:
            # Neural Network Vocoderを使用してwaveformを生成
            self._synthesize_with_vocoder_model()
        else:
            # WORLDを使用してwaveformを生成
            self._synthesize_with_world()

    def _synthesize_with_world(self) -> None:
        """WORLDを用いてWORLD特徴量からwaveformを生成する。"""
        # WORLDを使って直接waveformを合成
        clipped_ap = np.clip(self.ap.astype(np.float64), np.finfo(np.float64).tiny, 1.0)
        wav = pyworld.synthesize(  # pyright: ignore[reportAttributeAccessIssue]
            self.f0.astype(np.float64),
            self.sp.astype(np.float64),
            clipped_ap,
            self.framerate,
            frame_period=pyrwu.settings.PYWORLD_PERIOD,
        )

        # 生成した波形を _output_data に代入
        self._output_data = wav.astype(np.float32)

    def _synthesize_with_vocoder_model(self) -> None:
        """Vocoder modelを用いてWORLD特徴量からwaveformを生成する。"""
        assert self._vocoder_model is not None, 'vocoder_model is None'
        assert self._vocoder_config is not None, 'vocoder_config is None'
        assert self._vocoder_in_scaler is not None, 'vocoder_in_scaler is None'
        # WORLD 特徴量を NNSVS 用に変換
        # sp, ap はもとの wav のサンプリング周波数に基づいて抽出されているので、
        # nnsvs 向け特徴量への変換時はフレームレートは原音 wav のそれを渡す。
        # ap に 0 が含まれていると bap の計算で nan になるので、最小値を 1e-10 にする
        # DEBUG: --------------------------------------------------
        # モデルに渡す用に特徴量を変換する
        mgc, lf0, vuv, bap = world_to_nnsvs(self.f0, self.sp, self.ap, self.framerate)
        multistream_features = (mgc, lf0, vuv, bap)
        # DEBUG: --------------------------------------------------
        self.logger.debug('NNSVS features before waveform prediction -----------------------')
        self.logger.debug('  mgc (min, max): (%s, %s)', mgc.min(), mgc.max())
        self.logger.debug('  lf0 (min, max): (%s, %s)', lf0.min(), lf0.max())
        self.logger.debug('  vuv (min, max): (%s, %s)', vuv.min(), vuv.max())
        self.logger.debug('  bap (min, max): (%s, %s)', bap.min(), bap.max())
        # DEBUG: --------------------------------------------------
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
        # DEBUG: --------------------------------------------------
        self.logger.debug('wav: %s', wav)
        # DEBUG: --------------------------------------------------
        if self.vocoder_sample_rate != self.framerate:
            wav = librosa.resample(
                wav,
                orig_sr=self.vocoder_sample_rate,  # ボコーダモデルが出力するサンプルレート
                target_sr=self.framerate,  # UTAUの原音のサンプルレート
                res_type=self._resample_type,
            )
        # 生成した波形を _output_data に代入
        self._output_data = wav

    def resamp(self) -> None:
        """Neural Networkまたは WORLD を用いてWORLD特徴量をリサンプリングする。"""
        if self._use_vocoder_model:
            self.logger.info(
                'Resample with WORLD and vocoder model: %s (%s)',
                self._vocoder_model_dir,
                type(self._vocoder_model),
            )
        else:
            self.logger.info('Resample with WORLD')

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
        # synthesize はオーバーライドされているので vocoder または world を使って waveform 生成
        self.synthesize()
        # UST の音量を waveform に反映
        self.adjustVolume()  # TODO: npz にも反映できるようにする。

        # WAV ファイル出力
        if self.export_wav:
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
        type=str,
        nargs='?',
        default='',
    )
    parser.add_argument(
        'offset',
        help='入力ファイルの読み込み開始位置(ms)(省略可 default:0)',
        type=float,
        nargs='?',
        default=0,
    )
    parser.add_argument(
        'target_ms',
        help='出力ファイルの長さ(ms)(省略可 default:0)UTAUでは通常50ms単位に丸めた値が渡される。',
        type=float,
        nargs='?',
        default=0,
    )
    parser.add_argument(
        'fixed_ms',
        help='offsetからみて通常伸縮しない長さ(ms)',
        type=float,
        nargs='?',
        default=0,
    )
    parser.add_argument(
        'end_ms',
        help='入力ファイルの読み込み終了位置(ms)(省略可 default:0)'
        '正の数の場合、ファイル末尾からの時間'
        '負の数の場合、offsetからの時間',
        type=float,
        nargs='?',
        default=0,
    )
    parser.add_argument('volume', help='音量。0～200(省略可 default:100)', nargs='?', default=100)
    parser.add_argument(
        'modulation',
        help='モジュレーション。0～200(省略可 default:0)',
        type=int,
        nargs='?',
        default=0,
    )
    parser.add_argument(
        'tempo',
        help='ピッチのテンポ。数字の頭に!がついた文字列(省略可 default:"!120")',
        type=str,
        nargs='?',
        default='!120',
    )
    parser.add_argument(
        'pitchbend',
        help='ピッチベンド。(省略可 default:"")'
        '-2048～2047までの12bitの2進数をbase64で2文字の文字列に変換し、'
        '同じ数字が続く場合ランレングス圧縮したもの',
        type=str,
        nargs='?',
        default='',
    )
    # モデルを指定
    parser.add_argument(
        '--model_dir',
        help='Vocoder model directory (optional; required for neural network vocoder)',
        type=str,
        default=None,
    )
    # ボコーダーモデルを使用するか否か
    parser.add_argument(
        '--use_vocoder_model',
        help='Whether to use vocoder model for waveform synthesis. If False, use WORLD.',
        action='store_true',
        default=False,
    )

    # デバッグモード
    parser.add_argument(
        '--debug',
        help='Enable debug mode logging',
        action='store_true',
        default=False,
    )
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(40)  # logging.DEBUG
        logger.debug('Debug mode enabled')

    # NeuralNetworkResamp インスタンスを生成
    resamp = NeuralNetworkResamp(
        input_path=str(args.input_path),
        output_path=str(args.output_path),
        target_tone=str(args.target_tone),
        velocity=int(args.velocity),
        flag_value=str(args.flags),
        offset=float(args.offset),
        target_ms=float(args.target_ms),
        fixed_ms=float(args.fixed_ms),
        end_ms=float(args.end_ms),
        volume=int(args.volume),
        modulation=int(args.modulation),
        tempo=str(args.tempo),
        pitchbend=str(args.pitchbend),
        logger=logger,
        export_wav=True,
        export_features=False,
        use_vocoder_model=args.use_vocoder_model,
        vocoder_type='usfgan',
        vocoder_model_dir=args.model_dir,
    )
    # リサンプリングを実行
    resamp.resamp()


if __name__ == '__main__':
    main_resampler()

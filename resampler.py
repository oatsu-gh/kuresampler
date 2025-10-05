# Copyright (c) 2025 oatsu
"""Resampler classes

PyRwu.resamp.Resamp を継承し、
WAVファイルの代わりにWORLD特徴量をファイルに出力する
クラス WorldFeatureResamp を定義する。
"""

import argparse
import copy
import logging
import sys
import warnings
from logging import Logger
from pathlib import Path

import colored_traceback.auto  # noqa: F401
import librosa
import PyRwu as pyrwu  # noqa: N813
import pyworld
import torch
from nnsvs.util import StandardScaler
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))  # for local import

from convert import (
    waveform_to_wavfile,
    world_to_nnsvs_to_waveform,
    world_to_npzfile,
    world_to_waveform,
)
from util import denoise_spike, get_device, load_vocoder_model, setup_logger


# MARK: NeuralNetworkResamp
class NeuralNetworkResamp(pyrwu.Resamp):
    """Neural NetworkによるWORLD特徴量のリサンプリングを行う。

    Args:
        vocoder_model_dir: The directory containing the vocoder model files.
        use_vocoder_model: Whether to use vocoder model for waveform synthesis. If False, use WORLD.

    """

    # MARK: init
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
        use_vocoder_model: bool,
        logger: Logger | None = None,
        export_features: bool,
        vocoder_model: torch.nn.Module | None = None,
        vocoder_in_scaler: StandardScaler | None = None,
        vocoder_config: DictConfig | ListConfig | None = None,
        vocoder_type: str = 'usfgan',
        vocoder_feature_type: str = 'world',
        vocoder_vuv_threshold: float = 0.5,
        vocoder_frame_period: int = 5,
        resample_type: str = 'soxr_vhq',
        target_sample_rate: int | None = None,
    ) -> None:
        """Initialize NeuralNetworkResamp."""
        # logger 設定(先頭固定)======================================
        self.logger = setup_logger() if logger is None else logger
        # None が渡される可能性がある必須パラメータの処理============
        # tempo が None の場合は '!120' に設定
        if tempo is None:
            self.logger.warning('Tempo is None, set to "!120" by default')
            tempo = '!120'
        ## クラス変数への代入========================================
        self._input_path = input_path
        self._output_path = output_path
        self._target_tone = target_tone
        self._velocity = velocity
        self._flags = copy.deepcopy(pyrwu.settings.FLAGS)
        self._flag_value = flag_value
        self._offset = offset
        self._target_ms = target_ms
        self._fixed_ms = fixed_ms
        self._end_ms = end_ms
        self._volume = volume
        self._modulation = modulation
        self._tempo = tempo
        self._pitchbend = pitchbend
        # サンプリング周波数関連=========================
        self._original_sample_rate: int  # getInputData() で初期化予定
        self._target_sample_rate: int | None = target_sample_rate
        self.resample_type: str = resample_type
        ## インスタンス変数への代入==================================
        # WORLD特徴量をファイル出力するか否か
        self.export_features: bool = export_features
        # ボコーダーモデルを使用するか否か
        self.use_vocoder_model: bool = use_vocoder_model
        # フラグに 'e' (stretch) を追加して、原音WAVの伸縮をストレッチ式に強制する。
        self.__force_stretch()
        # デバイス設定
        self.device: torch.device = get_device()
        # ボコーダー関連の設定
        self.vocoder_type: str = vocoder_type
        self.vocoder_feature_type: str = vocoder_feature_type
        self.vocoder_vuv_threshold: float = vocoder_vuv_threshold
        self.vocoder_frame_period: int = vocoder_frame_period
        self.vocoder_model: torch.nn.Module | None = vocoder_model
        self.vocoder_in_scaler: StandardScaler | None = vocoder_in_scaler
        self.vocoder_config: DictConfig | ListConfig | None = vocoder_config

        # use_vocoder_model が True なのにボコーダー関連の引数が一つでも None の場合はエラー
        if use_vocoder_model is True and any(
            x is None for x in (vocoder_model, vocoder_in_scaler, vocoder_config)
        ):
            msg = 'When use_vocoder_model is True, vocoder_model, vocoder_in_scaler, and vocoder_config must be provided.'
            raise ValueError(msg)

        # use_vocoder_model が False なのにボコーダー関連の引数が指定されている場合は警告を出力し、None に強制
        if use_vocoder_model is False and any(
            x is not None for x in (vocoder_model, vocoder_in_scaler, vocoder_config)
        ):
            self._vocoder_model = None
            self._vocoder_in_scaler = None
            self._vocoder_config = None
            msg = 'vocoder_model, vocoder_in_scaler, and vocoder_config are ignored when use_vocoder_model is False.'
            self.logger.warning(msg)

    # MARK: properties
    @property
    def framerate(self) -> int:
        """内部処理のサンプリング周波数"""
        warnings.warn(
            DeprecationWarning('framerate is deprecated, use internal_sample_rate instead'),
            stacklevel=2,
        )
        return self._framerate

    @property
    def sample_rate(self) -> int:
        """内部処理のサンプリング周波数"""
        warnings.warn(
            DeprecationWarning('sample_rate is deprecated, use internal_sample_rate instead'),
            stacklevel=2,
        )
        return self._framerate

    @property
    def internal_sample_rate(self) -> int:
        """内部処理で使用するサンプリング周波数"""
        return self._framerate

    @internal_sample_rate.setter
    def internal_sample_rate(self, value: int) -> None:
        self._framerate = value

    @property
    def target_sample_rate(self) -> int:
        """出力wavのサンプリング周波数"""
        if self._target_sample_rate is None:
            msg = 'Target_sample_rate is None. It will be set to internal_sample_rate.'
            raise ValueError(msg)
        return self._target_sample_rate

    @target_sample_rate.setter
    def target_sample_rate(self, value: int) -> None:
        self._target_sample_rate = value

    @property
    def vocoder_sample_rate(self) -> int:
        """ボコーダーモデルのwav出力サンプリング周波数"""
        if self.vocoder_config is None:
            msg = 'vocoder_config is None. vocoder_model must be loaded first.'
            raise ValueError(msg)
        return self.vocoder_config.data.sample_rate

    def __force_stretch(self) -> None:
        """原音WAVの伸縮をストレッチ式に強制する。

        note.flags に `e` を追加する。
        ただし、note.flags に `e` が既に存在する場合や、`l` (loop) が明示的に指定されている場合は skip。
        """
        if 'e' not in self._flag_value and 'l' not in self._flag_value:
            self._flag_value += 'e'

    # MARK: getInputData
    def getInputData(
        self,
        f0_floor: float = pyrwu.settings.PYWORLD_F0_FLOOR,
        f0_ceil: float = pyrwu.settings.PYWORLD_F0_CEIL,
        frame_period: float = pyrwu.settings.PYWORLD_PERIOD,
        q1: float = pyrwu.settings.PYWORLD_Q1,
        threshold: float = pyrwu.settings.PYWORLD_THRESHOLD,
    ) -> None:
        """入力された音声データからworldパラメータを取得し、self._input_data, self._framerate, self._f0, self._sp, self._apを更新します。

        Args:
        f0_floor: float, default settings.PYWORLD_F0_FLOOR
            | worldでの分析するf0の下限
            | デフォルトでは71.0

        f0_ceil: float, default settings.PYWORLD_F0_CEIL
            | worldでの分析するf0の上限
            | デフォルトでは800.0

        frame_period: float, default settings.PYWORLD_PERIOD
            | worldデータの1フレーム当たりの時間(ms)
            | 初期設定では5.0

        q1: float, default settings.PYWORLD_Q1
            | worldでスペクトル包絡抽出時の補正値
            | 通常は変更不要
            | 初期設定では-15.0

        threshold: float, default settings.PYWORLD_THRESHOLD
            | worldで非周期性指標抽出時に、有声/無声を決定する閾値(0 ～ 1)
            | 値が0の場合、音声のあるフレームを全て有声と判定します。
            | 値が0超の場合、一部のフレームを無声音として判断します。
            | 初期値0.85はharvestと組み合わせる前提で調整されています。

        Notes:
            - 音声データの取得方法を変更したい場合、このメソッドをオーバーライドしてください。
            - オーバーライドする際、self._input_dataはこれ以降の処理で使用しないため、更新しなくても問題ありません。

        PyRwy.resamp.Resamp.getInputData() からの変更点:
            - PyWorldのキャッシュファイルを常に使用しない。
            - _getAp() を使わず pyworld.d4c() を直接使用する。
            - 原音 wav を self._input_data に代入する前に librosa.resample() でリサンプリングする。

        """
        wav_path = Path(self._input_path)
        frq_path = wav_path.with_name(wav_path.stem + '_wav.frq')
        # 原音のWAVファイルを切り出して、データとフレームレートを取得
        original_waveform, original_sample_rate = pyrwu.wave_io.read(
            self._input_path,
            self._offset,
            self._end_ms,
        )
        # 周波数表FRQファイルが無い場合は新規作成する
        if not frq_path.exists():
            input_data, original_sample_rate = pyrwu.wave_io.read(self._input_path, 0, 0)
            pyrwu.frq_io.write(input_data, str(frq_path), original_sample_rate)

        # 周波数表FRQファイルが存在する場合は読み込む
        if frq_path.exists():
            f0, t = pyrwu.frq_io.read(
                str(frq_path),
                self._offset,
                self._end_ms,
                original_sample_rate,
                frame_period,
            )
        # 周波数表FRQファイルが(なぜか)存在しない場合は新規にf0抽出する
        else:
            f0, t = pyworld.harvest(  # pyright: ignore[reportAttributeAccessIssue]
                original_waveform,
                original_sample_rate,
                f0_floor=f0_floor,
                f0_ceil=f0_ceil,
                frame_period=frame_period,
            )
        # f0をstonemaskで補正
        f0 = pyworld.stonemask(  # pyright: ignore[reportAttributeAccessIssue]
            original_waveform,
            f0,
            t,
            original_sample_rate,
        )
        ## リサンプル
        # ボコーダーモデルを使用する場合はそのモデルのサンプルレートにする
        # ボコーダーモデルを使用しない場合は原音のサンプルレートを維持
        internal_sample_rate = (
            self.vocoder_sample_rate if self.use_vocoder_model else original_sample_rate
        )
        resampled_wav = librosa.resample(
            original_waveform,
            orig_sr=original_sample_rate,
            target_sr=internal_sample_rate,
            res_type=self.resample_type,
        )
        # スペクトル包絡(sp)抽出
        sp = pyworld.cheaptrick(  # pyright: ignore[reportAttributeAccessIssue]
            resampled_wav,
            f0,
            t,
            internal_sample_rate,
            q1=q1,
            f0_floor=f0_floor,
        )
        # 非周期性指標(ap)抽出
        ap = pyworld.d4c(  # pyright: ignore[reportAttributeAccessIssue]
            resampled_wav,
            f0,
            t,
            internal_sample_rate,
            threshold=threshold,
        )
        # インスタンス変数に代入
        self._input_data = resampled_wav
        self._t = t
        self._f0 = f0
        self._sp = sp
        self._ap = ap
        self.original_sample_rate = original_sample_rate
        self.internal_sample_rate = internal_sample_rate
        self.target_sample_rate = self._target_sample_rate or internal_sample_rate

    def denoise_f0(self) -> None:
        """f0 のスパイクノイズを除去する。"""
        if self._f0 is not None:
            self._f0 = denoise_spike(self._f0, logger=self.logger)

    # MARK: synthesize
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

        if self.use_vocoder_model:
            # Neural Network Vocoderを使用してwaveformを生成
            self._synthesize_with_vocoder_model()
        else:
            # WORLDを使用してwaveformを生成
            self._synthesize_with_world()

    def _synthesize_with_world(self) -> None:
        """WORLDを用いてWORLD特徴量からwaveformを生成する。"""
        wav = world_to_waveform(
            self.f0,
            self.sp,
            self.ap,
            self.internal_sample_rate,
            frame_period=pyrwu.settings.PYWORLD_PERIOD,
        )
        # 生成した波形を _output_data に代入
        self._output_data = wav

    def _synthesize_with_vocoder_model(self) -> None:
        """Vocoder modelを用いてWORLD特徴量からwaveformを生成する。"""
        if self.vocoder_model is None:
            msg = 'vocoder_model is None'
            raise ValueError(msg)
        if self.vocoder_config is None:
            msg = 'vocoder_config is None'
            raise ValueError(msg)
        if self.vocoder_in_scaler is None:
            msg = 'vocoder_in_scaler is None'
            raise ValueError(msg)
        # nnsvs を使って waveform を合成
        wav = world_to_nnsvs_to_waveform(
            device=self.device,
            f0=self.f0,
            sp=self.sp,
            ap=self.ap,
            vocoder_model=self.vocoder_model,
            vocoder_config=self.vocoder_config,
            vocoder_in_scaler=self.vocoder_in_scaler,
            vocoder_frame_period=self.vocoder_frame_period,
            use_world_codec=True,
            feature_type=self.vocoder_feature_type,
            vocoder_type=self.vocoder_type,
            vuv_threshold=self.vocoder_vuv_threshold,
            target_sample_rate=self.internal_sample_rate,
            resample_type=self.resample_type,
        )
        # 生成した波形を _output_data に代入
        self._output_data = wav

    # MARK: resamp
    def resamp(self) -> None:
        """Neural Networkまたは WORLD を用いてWORLD特徴量をリサンプリングする。"""
        self.parseFlags()  # フラグ解析
        self.getInputData()  # WORLD特徴量を抽出
        self.stretch()  # 時間伸縮
        self.pitchShift()  # ピッチシフト
        self.applyPitch()  # ピッチベンド適用

        # パラメータ確認 ---------------------------------------
        self.logger.debug('  input_path  : %s', self.input_path)
        self.logger.debug('  output_path : %s', self.output_path)
        self.logger.debug('  t.shape     : %s', self.t.shape)
        self.logger.debug('  f0.shape    : %s', self.f0.shape)
        self.logger.debug('  sp.shape    : %s', self.sp.shape)
        self.logger.debug('  ap.shape    : %s', self.ap.shape)
        self.logger.debug('  original_sample_rate : %s', self.original_sample_rate)
        self.logger.debug('  internal_sample_rate : %s', self.internal_sample_rate)
        self.logger.debug('  target_sample_rate   : %s', self.target_sample_rate)
        # ------------------------------------------------------
        # f0 のスパイクノイズを除去
        self.denoise_f0()
        # モデル情報とサンプルレートをログ出力
        # ボコーダーモデルを使用する場合はそのモデルのサンプルレートでwav出力する
        if self.use_vocoder_model:
            self.logger.info(
                'Synthesize WAV using Neural Vocoder (%s)',
                type(self.vocoder_model),
            )
        # ボコーダーモデルを使用しない場合は原音のサンプルレートでwav出力する
        else:
            self.logger.info('Synthesize WAV using WORLD Vocoder')

        # synthesize はオーバーライドされているので vocoder または world を使って waveform 生成
        self.synthesize()
        # UST の音量を waveform に反映
        self.adjustVolume()  # TODO: npz にも反映できるようにする。

        # WAV ファイル出力
        waveform_to_wavfile(
            self._output_data,
            self.output_path,
            in_sample_rate=self.internal_sample_rate,
            out_sample_rate=self.target_sample_rate,
        )
        self.logger.debug('Exported WAV file: %s', self.output_path)

        # WORLD 特徴量を npz ファイル出力する。
        if self.export_features:
            npz_path = Path(self.output_path).with_suffix('.npz')
            world_to_npzfile(self.f0, self.sp, self.ap, npz_path)
            self.logger.debug('Exported NPZ (f0, spectral_envelope, aperiodicity): %s', npz_path)


def main_resampler(
    arg_list: list | None = None,
    *,
    vocoder_model: torch.nn.Module | None = None,
    vocoder_in_scaler: StandardScaler | None = None,
    vocoder_config: DictConfig | ListConfig | None = None,
) -> None:
    """実行引数を展開して Resamp インスタンスを生成し、resamp() を実行する。

    想定される sys.argv の形式
    resampler.py <input_wavfile> <output_file> <pitch_percent> <velocity> <flags> <offset> <length_require> <fixed_length> <end_blank> <volume> <modulation> <pitch_bend>

    Args:
        arg_list          : 引数リスト。None の場合は sys.argv[1:] を使用する。
        vocoder_model     : ボコーダーモデル。None の場合は WORLD を使用する。
        vocoder_in_scaler : ボコーダーモデルの入力スケーラー。
        vocoder_config    : ボコーダーモデルの設定。

    Returns:
        None

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
    parser.add_argument(
        'volume',
        help='音量。0～200(省略可 default:100)',
        nargs='?',
        default=100,
    )
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

    # arg_list が None の場合は sys.argv[1:] を使用する
    if arg_list is None:
        arg_list = sys.argv[1:]
        if '--debug' in arg_list:
            logger.setLevel(logging.DEBUG)
            logger.debug('Debug mode enabled')
        logger.debug('Arguments are set from sys.argv: %s', arg_list)
    else:
        if '--debug' in arg_list:
            logger.setLevel(logging.DEBUG)
            logger.debug('Debug mode enabled')
        logger.debug('Arguments are set from caller: %s', arg_list)

    # 引数を解析
    args = parser.parse_args(arg_list)

    # model_dir が渡されている場合はモデルを読み込む
    if args.model_dir is not None:
        if not args.use_vocoder_model:
            logger.warning(
                '--model_dir is specified but --use_vocoder_model is not set. The model will be ignored.'
            )
        else:
            # vocoder model 関連のファイルを読み込む
            model_dir = Path(args.model_dir)
            vocoder_model, vocoder_in_scaler, vocoder_config = load_vocoder_model(model_dir)

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
        export_features=False,
        use_vocoder_model=args.use_vocoder_model,
        vocoder_model=vocoder_model,
        vocoder_in_scaler=vocoder_in_scaler,
        vocoder_config=vocoder_config,
        vocoder_type='usfgan',
    )
    # リサンプリングを実行
    resamp.resamp()


if __name__ == '__main__':
    main_resampler()

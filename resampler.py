# Copyright (c) 2025 oatsu
"""
Resampler classes

PyRwu.resamp.Resamp を継承し、
WAVファイルの代わりにWORLD特徴量をファイルに出力する
クラス WorldFeatureResamp を定義する。
"""

import argparse
import re
import sys
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

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))  # for local import

from convert import (
    world_to_nnsvs,
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
        tempo: str | None = None,
        pitchbend: str = '',
        *,
        logger: Logger | None = None,
        export_wav: bool,
        export_features: bool,
    ) -> None:
        logger = setup_logger() if logger is None else logger
        if tempo is None:
            logger.warning('Tempo is None, set to "!120" by default')
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
        # フラグに 'e' (stretch) を追加して、原音WAVの伸縮をストレッチ式に強制する。
        self.__force_stretch()

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

    def resamp(self) -> None:
        """WAVファイルの代わりにWORLDの特徴量をファイルに出力する。"""
        self.parseFlags()
        self.getInputData()
        self.stretch()
        self.pitchShift()
        self.applyPitch()
        self.denoise_f0()
        self.synthesize()
        self.adjustVolume()
        if self.export_wav:
            self.output()
            self.logger.debug('Exported WAV file: %s', self.output_path)
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
        # effects 適用前の ap 値を記録
        ap_before_effects = copy(self._ap) if self._ap is not None else None
        ap_nonzero_count_before = np.count_nonzero(self._ap) if self._ap is not None else 0

        self.logger.debug(
            'Before effects - ap shape: %s, nonzero count: %d',
            self._ap.shape if self._ap is not None else None,
            ap_nonzero_count_before,
        )

        for effect in pyrwu.settings.F0_EFFECTS:
            self._f0 = effect.apply(self)

        for effect in pyrwu.settings.SP_EFFECTS:
            self._sp = effect.apply(self)

        for effect in pyrwu.settings.AP_EFFECTS:
            ap_before_this_effect = copy(self._ap) if self._ap is not None else None
            self._ap = effect.apply(self)

            # AP_EFFECTS の処理で ap が全て 0 になった場合の検出
            if self._ap is not None and np.all(self._ap == 0):
                effect_name = type(effect).__name__
                self.logger.warning(
                    'AP effect %s set all ap values to zero! Effect: %s', effect_name, effect
                )

                # B flag が 0 の場合は意図的に全て 0 にするので復元しない
                # それ以外の場合は予期しない動作として復元する
                b_flag_value = getattr(self, '_b_flag', None) if hasattr(self, '_b_flag') else None

                # フラグパーサーから B フラグの値を確認
                is_b_zero_intended = False
                if hasattr(self, '_flags') and self._flags:
                    # B フラグが明示的に 0 に設定されているかチェック
                    flag_str = str(getattr(self, '_flag_value', ''))
                    # より正確な B0 検出（B01, B02 等を除外）
                    b_zero_pattern = r'(?:^|[^0-9])B0(?:[^0-9]|$)'
                    if re.search(b_zero_pattern, flag_str):
                        is_b_zero_intended = True
                        self.logger.debug('B0 flag detected - ap=0 is intended behavior')

                if not is_b_zero_intended and ap_before_this_effect is not None:
                    self.logger.warning('Restoring previous ap values (not B0 case)')
                    self._ap = ap_before_this_effect
                elif is_b_zero_intended:
                    self.logger.debug('B0 flag: keeping ap=0 as intended')
                else:
                    self.logger.error('Cannot restore ap values - no previous values available')

        for effect in pyrwu.settings.WORLD_EFFECTS:
            f0_before, sp_before, ap_before = copy(self._f0), copy(self._sp), copy(self._ap)
            self._f0, self._sp, self._ap = effect.apply(self)

            # WORLD_EFFECTS の処理で ap が全て 0 になった場合の検出
            if self._ap is not None and np.all(self._ap == 0):
                effect_name = type(effect).__name__
                self.logger.warning(
                    'WORLD effect %s set all ap values to zero! Effect: %s', effect_name, effect
                )

                # B flag が 0 の場合以外は復元
                flag_str = str(getattr(self, '_flag_value', ''))
                # より正確な B0 検出（B01, B02 等を除外）
                b_zero_pattern = r'(?:^|[^0-9])B0(?:[^0-9]|$)'
                is_b_zero_intended = re.search(b_zero_pattern, flag_str) is not None

                if not is_b_zero_intended and ap_before is not None:
                    self.logger.warning('Restoring previous ap values (not B0 case)')
                    self._ap = ap_before
                elif is_b_zero_intended:
                    self.logger.debug('B0 flag: keeping ap=0 as intended')
                else:
                    self.logger.error('Cannot restore ap values - no previous values available')

        # effects 適用後の ap 値を記録
        ap_nonzero_count_after = np.count_nonzero(self._ap) if self._ap is not None else 0
        self.logger.debug(
            'After effects - ap shape: %s, nonzero count: %d',
            self._ap.shape if self._ap is not None else None,
            ap_nonzero_count_after,
        )

        # ap が全て 0 になった場合の最終チェックと復元（B0 フラグ以外）
        flag_str = str(getattr(self, '_flag_value', ''))
        # より正確な B0 検出（B01, B02 等を除外）
        b_zero_pattern = r'(?:^|[^0-9])B0(?:[^0-9]|$)'
        is_b_zero_intended = re.search(b_zero_pattern, flag_str) is not None

        if (
            self._ap is not None
            and np.all(self._ap == 0)
            and ap_before_effects is not None
            and not np.all(ap_before_effects == 0)
            and not is_b_zero_intended
        ):
            self.logger.warning(
                'All ap values are zero after effects processing (not B0)! Restoring original ap values.'
            )
            self._ap = ap_before_effects

        assert self._vocoder_model is not None  # 念のため確認

        # WORLD 特徴量を NNSVS 用に変換
        mgc, lf0, vuv, bap = world_to_nnsvs(self.f0, self.sp, self.ap, self.vocoder_sample_rate)
        # モデルに渡す用に特徴量をまとめる
        multistream_features = (mgc, lf0, vuv, bap)
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

    def resamp(self) -> None:
        """Neural Networkを用いてWORLD特徴量をリサンプリングする。"""
        self.logger.info(
            'Resample using vocoder model: %s (%s)',
            self._vocoder_model_dir,
            type(self._vocoder_model),
        )
        self.parseFlags()  # フラグを取得

        # フラグの情報をログ出力
        self.logger.debug('Parsed flags: %s', getattr(self, '_flag_value', 'Unknown'))

        self.getInputData()  # 原音WAVからWORLD特徴量を抽出

        # getInputData後のap状態をチェック
        ap_nonzero_after_input = np.count_nonzero(self._ap) if self._ap is not None else 0
        self.logger.debug('After getInputData - ap nonzero count: %d', ap_nonzero_after_input)

        self.stretch()  # 時間伸縮

        # stretch後のap状態をチェック
        ap_nonzero_after_stretch = np.count_nonzero(self._ap) if self._ap is not None else 0
        self.logger.debug('After stretch - ap nonzero count: %d', ap_nonzero_after_stretch)

        self.pitchShift()  # ピッチシフト

        # pitchShift後のap状態をチェック
        ap_nonzero_after_pitch_shift = np.count_nonzero(self._ap) if self._ap is not None else 0
        self.logger.debug('After pitchShift - ap nonzero count: %d', ap_nonzero_after_pitch_shift)

        self.applyPitch()  # ピッチベンド適用

        # applyPitch後のap状態をチェック
        ap_nonzero_after_apply_pitch = np.count_nonzero(self._ap) if self._ap is not None else 0
        self.logger.debug('After applyPitch - ap nonzero count: %d', ap_nonzero_after_apply_pitch)

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
    args = parser.parse_args()

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
        export_features=True,
        vocoder_type='usfgan',
        vocoder_model_dir=args.model_dir,
    )
    # リサンプリングを実行
    resamp.resamp()


if __name__ == '__main__':
    main_resampler()

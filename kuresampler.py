#!/usr/bin/env python3
# Copyright (c) 2025 oatsu
"""
UTAU engine for smooth crossfades

# 方針
- UTAU の resampler として、各ノートの WORLD 特徴量を生成する。
- UTAU の wavtool として、各ノートの WORLD 特徴量をクロスフェード結合する。
- 結合した WORLD 特徴量をニューラルボコーダーに入力し、WAV を出力する。

# 作る順番
- PyRwu で WORLD の特徴量をファイル出力するモジュールを作る。
- PyWavTool で クロスフェードする。
- NNSVS を使って WORLD 特徴量から WAV を生成する。
- WORLD 特徴量の IO 形式をそろえる。

"""

import logging
import os
import sys
from argparse import ArgumentParser
from copy import copy
from logging import Logger
from os.path import splitext
from pathlib import Path

import colored_traceback.auto  # noqa: F401
import librosa
import numpy as np
import PyRwu as pyrwu
import PyWavTool as pywavtool
import torch
from nnsvs.gen import predict_waveform
from nnsvs.usfgan import USFGANWrapper
from nnsvs.util import StandardScaler
from nnsvs.util import load_vocoder as nnsvs_load_vocoder
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from PyUtauCli.projects.Render import Render
from PyUtauCli.projects.Ust import Ust
from tqdm.auto import tqdm

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


def setup_logger() -> Logger:
    """Loggerを作成する。"""
    # my_package.my_moduleのみに絞ってsys.stderrにlogを出す
    logging.basicConfig(
        stream=sys.stdout,
        format='[%(filename)s][%(levelname)s] %(message)s',
        level=logging.INFO,
    )
    return logging.getLogger(__name__)


def get_device() -> torch.device:
    """PyTorch デバイスを取得する。"""
    device = torch.accelerator.current_accelerator(check_available=True) or torch.device('cpu')
    return device


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
        logger = logging.getLogger(__name__) if logger is None else logger
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

    def resamp(self) -> None:
        """
        WAVファイルの代わりにWORLDの特徴量をファイルに出力する。
        """
        self.parseFlags()
        self.getInputData()
        self.stretch()
        self.pitchShift()
        self.applyPitch()
        # パラメータ確認 ---------------------------------------
        self.logger.debug(f'  input_path  : {self.input_path}')
        self.logger.debug(f'  output_path : {self.output_path}')
        self.logger.debug(f'  framerate   : {self.framerate}')
        self.logger.debug(f'  t.shape     : {self.t.shape}')
        self.logger.debug(f'  f0.shape    : {self.f0.shape}')
        self.logger.debug(f'  sp.shape    : {self.sp.shape}')
        self.logger.debug(f'  ap.shape    : {self.ap.shape}')
        # ------------------------------------------------------
        # WORLD特徴量にフラグを適用したのち wavform を更新する。
        self.synthesize()
        # UST の音量を waveform に反映する。wav 出力しない場合は無駄な処理なのでskip。
        if self.export_wav:
            self.adjustVolume()
        # WAVファイル出力は必須ではないがテスト用に出力可能。
        if self.export_wav:
            self.output()
            self.logger.info(f'Exported WAV file: {self.output_path}')
        # WORLD 特徴量を npz ファイル出力する。
        if self.export_features:
            npz_path = splitext(self.output_path)[0] + '.npz'
            np.savez(npz_path, f0=self.f0, spectrogram=self.sp, aperiodicity=self.ap)
            self.logger.info(f'Exported WORLD features (f0, sp, ap): {npz_path}')


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
        """pyworld の代わりに vocoder model を用いてWORLD特徴量からwaveformを生成し、self._output_dataに代入する。"""

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
        """
        Neural Networkを用いてWORLD特徴量をリサンプリングする。
        """
        self.logger.info(
            f'Resample using vocoder model: {self._vocoder_model_dir} ({type(self._vocoder_model)})'
        )
        self.parseFlags()  # フラグを取得
        self.getInputData()  # 原音WAVからWORLD特徴量を抽出
        self.stretch()  # 時間伸縮
        self.pitchShift()  # ピッチシフト
        self.applyPitch()  # ピッチベンド適用

        # パラメータ確認 ---------------------------------------
        self.logger.debug(f'  input_path  : {self.input_path}')
        self.logger.debug(f'  output_path : {self.output_path}')
        self.logger.debug(f'  framerate   : {self.framerate}')
        self.logger.debug(f'  t.shape     : {self.t.shape}')
        self.logger.debug(f'  f0.shape    : {self.f0.shape}')
        self.logger.debug(f'  sp.shape    : {self.sp.shape}')
        self.logger.debug(f'  ap.shape    : {self.ap.shape}')
        # ------------------------------------------------------
        # NOTE: synthesize はオーバーライドされていえるので nnsvs を使って waveform 生成していることに注意
        self.synthesize()
        # UST の音量を waveform に反映する。wav 出力しない場合は無駄な処理なのでskip。
        if self.export_wav:
            self.adjustVolume()
        # WAVファイル出力は必須ではないがテスト用に出力可能。
        if self.export_wav:
            self.output()
            self.logger.info(f'Exported WAV file: {self.output_path}')
        # WORLD 特徴量を npz ファイル出力する。
        if self.export_features:
            npz_path = splitext(self.output_path)[0] + '.npz'
            np.savez(npz_path, f0=self.f0, spectrogram=self.sp, aperiodicity=self.ap)
            self.logger.info(f'Exported WORLD features (f0, sp, ap): {npz_path}')


class WorldFeatureWavTool(pywavtool.WavTool):
    """WAV出力の代わりに WORLD の特徴量を出力するのに用いる。"""

    _export_wav: bool
    _export_features: bool

    def __init__(self, output: str):
        """
        Parameters
        ----------
        output : str
            出力するwavのパス
        """
        self._error = False
        if os.path.split(output)[0] != '':
            os.makedirs(os.path.split(output)[0], exist_ok=True)
        self._header = pywavtool.whd.Whd(output + '.whd')
        self._dat = pywavtool.dat.Dat(output + '.dat')
        self._output = output

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


class WorldFeatureRender(Render):
    """
    WAV出力の代わりに WORLD の特徴量ファイルを出力するのに用いる。
    """

    _export_wav: bool
    _export_features: bool

    def __init__(
        self,
        ust: Ust,
        *,
        voice_dir: str = '',
        cache_dir: str = '',
        output_file: str = '',
        logger: Logger | None = None,
        export_wav: bool = False,
        export_features: bool = False,
    ) -> None:
        super().__init__(
            ust, voice_dir=voice_dir, cache_dir=cache_dir, output_file=output_file, logger=logger
        )
        self._export_wav = export_wav
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

    def resamp(self, *, force: bool = False) -> None:
        """
        Resampの代わりにWorldFeatureResampを用いる。

        PyRwu.Resampを使用してキャッシュファイルを生成する。
        Parameters
        ----------
        force: bool, default False
            Trueの場合、キャッシュファイルがあっても生成する。
        """
        os.makedirs(self._cache_dir, exist_ok=True)
        for note in tqdm(self.notes, colour='cyan'):
            self.logger.debug('------------------------------------------------')
            if not note.require_resamp:
                continue
            if force or not os.path.isfile(note.cache_path):
                self.logger.info(
                    '{} {} {} {} {} {} {} {} {} {} {} {} {}'.format(  # noqa: UP032
                        note.input_path,
                        note.cache_path,
                        note.target_tone,
                        note.velocity,
                        note.flags,
                        note.offset,
                        note.target_ms,
                        note.fixed_ms,
                        note.end_ms,
                        note.intensity,
                        note.modulation,
                        note.tempo,
                        note.pitchbend,
                    )
                )
                resamp = WorldFeatureResamp(
                    input_path=note.input_path,
                    output_path=note.cache_path,
                    target_tone=note.target_tone,
                    velocity=note.velocity,
                    flag_value=note.flags,
                    offset=note.offset,
                    target_ms=note.target_ms,
                    fixed_ms=note.fixed_ms,
                    end_ms=note.end_ms,
                    volume=note.intensity,
                    modulation=note.modulation,
                    tempo=note.tempo,
                    pitchbend=note.pitchbend,
                    logger=self.logger,
                    export_wav=self.export_wav,
                    export_features=self.export_features,
                )
                resamp.resamp()
            else:
                self.logger.info(f'Using cache ({note.cache_path})')
        self.logger.debug('------------------------------------------------')


class NeuralNetworkRender(WorldFeatureRender):
    """ニューラルボコーダーを使ってWAVレンダリングする。

    # TODO: resampler 部分と wavtool 部分それぞれ vocoder model を使うか選択できるようにクラス指定できるようにする。
    # TODO: model_dir ではなく load 済みのモデルを渡しても良いようにする。高速化のため。
    """

    def __init__(
        self,
        *args,
        vocoder_model_dir: Path | str,
        vocoder_type: str = 'usfgan',
        vocoder_feature_type: str = 'world',
        vocoder_vuv_threshold: float = 0.5,
        vocoder_frame_period: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._vocoder_model_dir = vocoder_model_dir
        self._vocoder_type = vocoder_type
        self._vocoder_feature_type = vocoder_feature_type
        self._vocoder_vuv_threshold = vocoder_vuv_threshold
        self._vocoder_frame_period = vocoder_frame_period

    def resamp(self, *, force: bool = False) -> None:
        """
        PyRwu.Resamp の代わりに NeuralNetworkResamp を用いる。

        PyRwu.Resampを使用してキャッシュファイルを生成する。
        Parameters
        ----------
        force: bool, default False
            Trueの場合、キャッシュファイルがあっても生成する。
        """
        os.makedirs(self._cache_dir, exist_ok=True)
        for note in tqdm(self.notes, colour='cyan'):
            self.logger.debug('------------------------------------------------')
            if not note.require_resamp:
                continue
            if force or not os.path.isfile(note.cache_path):
                self.logger.info(
                    '{} {} {} {} {} {} {} {} {} {} {} {} {}'.format(  # noqa: UP032
                        note.input_path,
                        note.cache_path,
                        note.target_tone,
                        note.velocity,
                        note.flags,
                        note.offset,
                        note.target_ms,
                        note.fixed_ms,
                        note.end_ms,
                        note.intensity,
                        note.modulation,
                        note.tempo,
                        note.pitchbend,
                    )
                )
                resamp = NeuralNetworkResamp(
                    input_path=note.input_path,
                    output_path=note.cache_path,
                    target_tone=note.target_tone,
                    velocity=note.velocity,
                    flag_value=note.flags,
                    offset=note.offset,
                    target_ms=note.target_ms,
                    fixed_ms=note.fixed_ms,
                    end_ms=note.end_ms,
                    volume=note.intensity,
                    modulation=note.modulation,
                    tempo=note.tempo,
                    pitchbend=note.pitchbend,
                    logger=self.logger,
                    export_wav=self.export_wav,
                    export_features=self.export_features,
                    vocoder_model_dir=self._vocoder_model_dir,
                )
                resamp.resamp()
            else:
                self.logger.info(f'Using cache ({note.cache_path})')
        self.logger.debug('------------------------------------------------')


def load_vocoder_model(
    model_dir: Path | str, device: torch.device | None = None
) -> tuple[USFGANWrapper, StandardScaler, ListConfig | DictConfig]:
    """Load NNSVS vocoder model

    Args:
        model_dir (Path|str): Path to the model directory
        device (torch.device | None): Device to load the model on
    """
    model_dir = Path(model_dir)
    # get device
    if device is None:
        device = get_device()
    # load configs
    vocoder_model_path = model_dir / 'vocoder_model.pth'
    # vocoder_config_path = model_dir / 'vocoder_model.yaml'
    acoustic_config_path = model_dir / 'acoustic_model.yaml'
    # vocoder_config = OmegaConf.load(vocoder_config_path)
    acoustic_config = OmegaConf.load(acoustic_config_path)

    vocoder, vocoder_in_scaler, vocoder_config = nnsvs_load_vocoder(
        vocoder_model_path, device, acoustic_config
    )
    return vocoder, vocoder_in_scaler, vocoder_config


def nnsvs_to_waveform(
    mgc: np.ndarray,
    lf0: np.ndarray,
    vuv: np.ndarray,
    bap: np.ndarray,
    vocoder_model_dir: Path | str,
    *,
    vocoder_type: str = 'usfgan',
    feature_type: str = 'world',
    vuv_threshold: float = 0.5,
    frame_period: int = 5,
    logger: Logger | None = None,
) -> np.ndarray:
    """world (original) の特徴量から wav を生成する。
    2025-08-25 時点の nnsvs.gen.predict_waveform は vocoder_type は world, pwg, usfgan に対応している。

    Args:
        f0 (np.ndarray): continuous F0
        spectrogram (np.ndarray):
        aperiodicity (np.ndarray): band aperiodicity
        vocoder (nn.Module): Vocoder model
        vocoder_config (dict): Vocoder config
        vocoder_in_scaler (StandardScaler): Vocoder input scaler
        vocoder_type (str): Vocoder type. `world` or `pwg` or `usfgan`

    Returns:
        wav (np.ndarray): wavform. shape: (n_samples,)
    """
    logger = logging.getLogger(__name__) if logger is None else logger
    # モデルに渡す用に特徴量をまとめる
    multistream_features = (mgc, lf0, vuv, bap)
    # モデルを読み込む
    vocoder_model, vocoder_in_scaler, vocoder_config = load_vocoder_model(vocoder_model_dir)
    # サンプリング周波数を自動取得
    sample_rate = vocoder_config.data.sample_rate
    # waveform を生成
    device = get_device()
    wav = predict_waveform(
        device=device,
        multistream_features=multistream_features,
        vocoder=vocoder_model,
        vocoder_config=vocoder_config,
        vocoder_in_scaler=vocoder_in_scaler,
        sample_rate=sample_rate,
        frame_period=frame_period,
        use_world_codec=True,
        feature_type=feature_type,
        vocoder_type=vocoder_type,
        vuv_threshold=vuv_threshold,  # vuv 閾値設定はするけど使われないはず
    )
    # 生成した waveform を返す
    return wav


def main_as_resampler() -> None:
    """resampler (伸縮器) として各ノートの wav 加工を行う。

    Args
    """
    logger = setup_logger()

    parser = ArgumentParser(description='This module is Resampler for UTAU powered by world')
    parser.add_argument('input_path', help='原音のファイル名', type=str)
    parser.add_argument('output_path', help='wavファイルの出力先パス', type=str)
    parser.add_argument(
        'target_tone',
        help='音高名(A4=440Hz)。'
        + '半角上げは#もしくは♯'
        + '半角下げはbもしくは♭で与えられます。',
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
        help='出力ファイルの長さ(ms)(省略可 default:0)'
        + 'UTAUでは通常50ms単位に丸めた値が渡される。',
        nargs='?',
        default=0,
    )
    parser.add_argument(
        'fixed_ms', help='offsetからみて通常伸縮しない長さ(ms)', nargs='?', default=0
    )
    parser.add_argument(
        'end_ms',
        help='入力ファイルの読み込み終了位置(ms)(省略可 default:0)'
        + '正の数の場合、ファイル末尾からの時間'
        + '負の数の場合、offsetからの時間',
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
        + '-2048～2047までの12bitの2進数をbase64で2文字の文字列に変換し、'
        + '同じ数字が続く場合ランレングス圧縮したもの',
        nargs='?',
        default='',
    )
    parser.add_argument('--show-flag', action=pyrwu.ShowFlagAction)
    args = parser.parse_args()

    if args.pitchbend == '':  # flagsに値がないとき、引数がずれてしまうので補正する。
        args.pitchbend = args.tempo
        args.tempo = args.modulation
        args.modulation = args.volume
        args.volume = args.end_ms
        args.end_ms = args.fixed_ms
        args.fixed_ms = args.target_ms
        args.target_ms = args.offset
        args.offset = args.flags
        args.flags = ''

    WorldFeatureResamp(
        args.input_path,
        args.output_path,
        args.target_tone,
        args.velocity,
        args.flags,
        float(args.offset),
        int(args.target_ms),
        float(args.fixed_ms),
        float(args.end_ms),
        int(args.volume),
        int(args.modulation),
        args.tempo,
        args.pitchbend,
        logger=logger,
        export_features=True,
        export_wav=True,
    ).resamp()


def main_as_integrated_wavtool(path_ust: str, path_wav: str) -> None:
    """WavTool1 (Append) と WavTool2 (Resamp) を統合的に実行する。

    長所:
    - 特徴量でクロスフェードしたあとにボコーダーを通すので接続が滑らかだが、

    短所:
    - 特徴量でクロスフェードする必要があるので、クロスフェード計算を独自実装する必要あり。
    - エンベロープおよびゲイン反映を独自実装する必要あり。
    - 音量ノーマライズを独自実装する必要あり。いっそ world で wav を内部生成して音量係数を取得してしまう？
    - 一度にボコーダーに渡すサイズが大きいので WAV 生成に時間がかかり、VRAM 消費も激しい。
    """
    # TODO: 実装
    pass


if __name__ == '__main__':
    pass

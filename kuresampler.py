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
from logging import Logger
from os.path import dirname, join, splitext
from pathlib import Path
from tempfile import TemporaryDirectory

import colored_traceback.auto  # noqa: F401
import numpy as np
import PyRwu as pyrwu
import PyWavTool as pywavetool
import torch
import utaupy
from nnsvs.gen import predict_waveform
from nnsvs.usfgan import USFGANWrapper
from nnsvs.util import StandardScaler
from nnsvs.util import load_vocoder as nnsvs_load_vocoder
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
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
        level=logging.DEBUG,
    )
    return logging.getLogger(__name__)


def get_device() -> torch.device:
    """PyTorch デバイスを取得する。"""
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device('cpu')


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

    def resamp(self) -> tuple:
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
        # WORLD 特徴量を返す。
        return self.f0, self.sp, self.ap


class WorldFeatureWavTool(pywavetool.WavTool):
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
        self._header = pywavetool.whd.Whd(output + '.whd')
        self._dat = pywavetool.dat.Dat(output + '.dat')
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


def load_vocoder_model(
    model_dir: Path | str, device: torch.device | None = None
) -> tuple[USFGANWrapper, StandardScaler, DictConfig]:
    """Load NNSVS vocoder model

    Supports only packed models.
    # NOTE:
    # If you want to load non-packed ParallelWaveGAN models or non-packed uSFGAN models, please refer `nnsvs.util.load_vocoder()`.
    # Simple process step is prioritized in this function.

    Args:
        model_dir (Path|str): Path to the model directory
        device (torch.device | None): Device to load the model on
    """
    model_dir = Path(model_dir)
    # get device
    if device is None:
        device = get_device()
    # load configs
    model_dir = Path(model_dir)
    vocoder_model_path = model_dir / 'vocoder_model.pth'
    vocoder_config_path = model_dir / 'vocoder_model.yaml'
    acoustic_config_path = model_dir / 'acoustic_model.yaml'
    _vocoder_config = OmegaConf.load(vocoder_config_path)
    acoustic_config = OmegaConf.load(acoustic_config_path)

    vocoder, vocoder_in_scaler, vocoder_config = nnsvs_load_vocoder(
        vocoder_model_path, device, acoustic_config
    )
    return vocoder, vocoder_in_scaler, vocoder_config


def nnsvs_world_to_waveform(
    f0: np.ndarray,
    spectrogram: np.ndarray,
    aperiodicity: np.ndarray,
    sample_rate: int,
    vocoder: torch.nn.Module | None,
    vocoder_config: dict | None,
    vocoder_in_scaler: StandardScaler | None,
    vocoder_type: str,
    *,
    logger: Logger | None = None,
) -> np.ndarray:
    """world (original) の特徴量から wav を生成する。
    2025-08-25 時点の nnsvs.gen.predict_waveform は vocoder_type は world, pwg, usfgan に対応している。

    Args:
        f0 (np.ndarray): continuous F0
        spectrogram (np.ndarray):
        aperiodicity (np.ndarray): band aperiodicity
        sample_rate (int): sample rate
        vocoder (nn.Module): Vocoder model
        vocoder_config (dict): Vocoder config
        vocoder_in_scaler (StandardScaler): Vocoder input scaler
        vocoder_type (str): Vocoder type. `world` or `pwg` or `usfgan`

    Returns:
        wav (np.ndarray): wavform. shape: (n_samples,)
    """
    logger = logging.getLogger(__name__) if logger is None else logger

    # WORLD特徴量をNNSVS用に前処理する
    mgc, lf0, vuv, bap = world_to_nnsvs(f0, spectrogram, aperiodicity)
    # 関数に渡すために形式を整える
    multistream_features = (mgc, lf0, vuv, bap)

    # auto detect device
    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available()
        else torch.device('cpu')
    )
    logger.debug(f'Using device: {device}')

    # predict waveform with nnsvs-usfgan model
    wav = predict_waveform(
        device=device,
        multistream_features=multistream_features,
        vocoder=vocoder,
        vocoder_config=vocoder_config,
        vocoder_in_scaler=vocoder_in_scaler,
        sample_rate=sample_rate,
        frame_period=5,
        use_world_codec=True,
        feature_type='world',
        vocoder_type=vocoder_type,
        vuv_threshold=0.5,  # vuv 閾値設定はするけど使われないはず
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
    logger = setup_logger()

    # TODO: 特徴量取得を実装
    f0 = None
    spectrogram = None
    aperiodicity = None
    # TODO: モデルとconfig読み取りを実装
    sample_rate = 48000
    vocoder = None
    vocoder_config = None
    vocoder_in_scaler = None
    vocoder_type = 'usfgan'

    # WORLD特徴量をNNSVS用に前処理する
    mgc, lf0, vuv, bap = world_to_nnsvs(f0, spectrogram, aperiodicity)
    # 関数に渡すために形式を整える
    multistream_features = (mgc, lf0, vuv, bap)

    # Auto detect device
    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available()
        else torch.device('cpu')
    )

    # Predict waveform with nnsvs-usfgan model
    wav = predict_waveform(
        device=device,
        multistream_features=multistream_features,
        vocoder=vocoder,
        vocoder_config=vocoder_config,
        vocoder_in_scaler=vocoder_in_scaler,
        sample_rate=sample_rate,
        frame_period=5,
        use_world_codec=True,
        feature_type='world',
        vocoder_type=vocoder_type,
        vuv_threshold=0.5,  # vuv 閾値設定はするけど使われないはず
    )
    # 生成した waveform を返す
    return wav


def main_as_standalone(path_ust_in, path_wav_out) -> None:
    """全体の処理を行う。"""
    logger = setup_logger()
    # utaupyでUSTを読み取る
    ust_utaupy = utaupy.ust.load(path_ust_in)
    voice_dir = ust_utaupy.voicedir
    # ust_path = ust.setting.get('Project')  # noqa: F841
    cache_dir = ust_utaupy.setting.get('CacheDir', join(dirname(__file__), 'kuresampler.cache'))
    # path_wav_out = ust.setting.get('OutFile', 'output.wav')  # noqa: F841

    # 一時フォルダにustを出力してPyUtauCliで読み直す
    with TemporaryDirectory() as temp_dir:
        # utaupyでプラグインをustファイルとして保存する
        path_temp_ust = join(temp_dir, 'temp.ust')
        if isinstance(ust_utaupy, utaupy.utauplugin.UtauPlugin):
            ust_utaupy.as_ust().write(path_temp_ust)
        else:
            ust_utaupy.write(path_temp_ust)
        # pyutaucliでustを読み込みなおす
        ust = Ust(path_temp_ust)
        ust.load()

    # PyUtauCli でレンダリング
    render = WorldFeatureRender(
        ust,
        logger=logger,
        voice_dir=str(voice_dir),
        cache_dir=str(cache_dir),
        output_file=str(path_wav_out),
        export_wav=True,
        export_features=True,
    )
    render.clean()
    render.resamp(force=True)

    # render.append()


if __name__ == '__main__':
    main_as_standalone(
        join(dirname(__file__), 'test.ust'),
        join(dirname(__file__), 'output.wav'),
    )

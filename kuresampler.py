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
from logging import Logger
from os.path import dirname, join, splitext
from pathlib import Path
from tempfile import TemporaryDirectory

import colored_traceback.auto  # noqa: F401
import numpy as np
import pyworld
import torch
import utaupy
from nnsvs.gen import predict_waveform
from PyRwu.resamp import Resamp
from PyUtauCli.projects.Render import Render
from PyUtauCli.projects.Ust import Ust
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


def setup_logger() -> Logger:
    """Loggerを作成する。"""
    # my_package.my_moduleのみに絞ってsys.stderrにlogを出す
    logging.basicConfig(
        stream=sys.stdout,
        format='[%(filename)s][%(levelname)s] %(message)s',
        level=logging.DEBUG,
    )
    return logging.getLogger(__name__)


class WorldFeatureResamp(Resamp):
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
        export_wav: bool = False,
        export_features: bool = False,
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

    def resamp(self, *args, force: bool = False) -> None:
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


def world_original_to_world_nnsvs(
    f0: np.ndarray,
    spectrogram: np.ndarray,
    aperiodicity: np.ndarray,
    sample_rate: int,
    number_of_mgc_dimensions: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    world 特徴量を nnsvs 対応の形式に変換する。

    Args:
        f0           (np.ndarray): F0
        spectrogram  (np.ndarray): spectrogram
        aperiodicity (np.ndarray): aperiodicity

    Returns:
        mgc (np.ndarray): mel-generalized cepstral coefficients
        lf0 (np.ndarray): log F0
        vuv (np.ndarray): voiced / unvoiced flag
        bap (np.ndarray): band aperiodicity
    """
    # spectrogram -> mgc
    mgc = pyworld.code_spectral_envelope(spectrogram, sample_rate, number_of_mgc_dimensions)
    # f0 -> lf0
    lf0 = np.where(f0 > 0, np.log(f0), 0)
    # vuv を作成
    vuv = (f0 > 0).astype(np.float32)
    # aperiodicity -> bap
    bap = pyworld.code_aperiodicity(aperiodicity, sample_rate)

    # nnsvs 向けの world 特徴量を返す
    return mgc, lf0, vuv, bap


def npz2wav(
    npz_path: Path | str,
    sample_rate: int,
    vocoder: torch.nn.Module | None,
    vocoder_config: dict | None,
    vocoder_in_scaler: StandardScaler | None,
    vocoder_type: str,
    *,
    logger: Logger | None = None,
) -> np.ndarray:
    """WORLD特徴量 (f0, spectrogram, aperiodicity) のファイルを読み取って waveform を生成する。

    Args:
        npz_file (Path | str): archive of WORLD features (f0, spectrogram, aperiodicity)
        sample_rate (int): sample rate for the output waveform
        vocoder (torch.nn.Module | None): vocoder model for waveform generation
        vocoder_config (dict | None): configuration for the vocoder
        vocoder_in_scaler (StandardScaler | None): scaler for vocoder input features
        vocoder_type (str): type of vocoder (e.g., "world", "pwg", "usfgan")
        logger (Logger | None, optional): logger instance. Defaults to None.

    Returns:
        np.ndarray: waveform
    """
    logger = logging.getLogger(__name__) if logger is None else logger

    # npzファイルを読み込む
    npz_data = np.load(npz_path)
    # 特徴量をそれぞれ取り出す
    f0 = npz_data['f0']
    spectrogram = npz_data['spectrogram']
    aperiodicity = npz_data['aperiodicity']

    # WORLD特徴量からwavを生成する
    wav = world2wav(
        f0=f0,
        spectrogram=spectrogram,
        aperiodicity=aperiodicity,
        sample_rate=sample_rate,
        vocoder=vocoder,
        vocoder_config=vocoder_config,
        vocoder_in_scaler=vocoder_in_scaler,
        vocoder_type=vocoder_type,
        logger=logger,
    )
    return wav


def world2wav(
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
    mgc, lf0, vuv, bap = world_original_to_world_nnsvs(f0, spectrogram, aperiodicity)
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


def render_wav_from_ust(path_ust_in, path_wav_out) -> None:
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


def main_as_normal_resampler(path_ust_in, path_wav_out) -> None:
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
    # キャッシュ削除
    render.clean()
    # 各ノートのキャッシュファイルを生成
    render.resamp(force=True)
    # キャッシュファイルを結合して全体wavを出力
    render.append()


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
    # WORLD特徴量をNNSVS用に前処理する
    mgc, lf0, vuv, bap = world_original_to_world_nnsvs(f0, spectrogram, aperiodicity)
    # 関数に渡すために形式を整える
    multistream_features = (mgc, lf0, vuv, bap)

    # auto detect device
    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available()
        else torch.device('cpu')
    )

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


if __name__ == '__main__':
    main_as_normal_resampler(
        join(dirname(__file__), 'test.ust'),
        join(dirname(__file__), 'output.wav'),
    )

#!/usr/bin/env python3
# Copyright (c) 2025 oatsu
"""UTAU engine for smooth crossfades

# 方針
- UTAU の resampler として、各ノートの WORLD 特徴量を生成する。
- UTAU の wavtool として、各ノートの WORLD 特徴量をクロスフェード結合する。
- 結合した WORLD 特徴量をニューラルボコーダーに入力し、WAV を出力する。

"""

import logging
import sys
from argparse import ArgumentParser
from copy import copy
from logging import Logger
from pathlib import Path
from shutil import rmtree
from typing import Any

import colored_traceback.auto  # noqa: F401
import librosa
import numpy as np
import PyRwu as pyrwu  # noqa: N813
import PyWavTool as pywavtool  # noqa: N813
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


# MARK: Utility functions
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


def load_vocoder_model(
    model_dir: Path | str,
    device: torch.device | None = None,
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
        vocoder_model_path,
        device,
        acoustic_config,
    )
    return vocoder, vocoder_in_scaler, vocoder_config


def overlap_world_features(
    features_a: np.ndarray,
    features_b: np.ndarray,
    overlap_samples: int,
) -> np.ndarray:
    """WORLD特徴量をオーバーラップさせる。

    Args:
        features_a (np.ndarray): The first set of WORLD features.
        features_b (np.ndarray): The second set of WORLD features.
        fade_samples (int): The number of samples to fade.

    Returns:
        np.ndarray: The crossfaded WORLD features.
    """
    # Create a linear fade-in and fade-out envelope
    # クロスフェード部分の重ね合わせを行う
    overlap_area = features_a[:-overlap_samples] + features_b[:overlap_samples]
    # 両端を結合して返す
    result = np.concatenate(
        [features_a[:-overlap_samples], overlap_area, features_b[overlap_samples:]]
    )
    print(
        'a.shape:',
        features_a.shape,
        'b.shape:',
        features_b.shape,
        'overlap_samples:',
        overlap_samples,
        'result.shape:',
        result.shape,
    )
    return result


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
    """World (original) の特徴量から wav を生成する。
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
        """WAVファイルの代わりにWORLDの特徴量をファイルに出力する。"""
        self.parseFlags()
        self.getInputData()
        self.stretch()
        self.pitchShift()
        self.applyPitch()
        # パラメータ確認 ---------------------------------------
        self.logger.debug('  input_path  : %s', self.input_path)
        self.logger.debug('  output_path : %s', self.output_path)
        self.logger.debug('  framerate   : %s', self.framerate)
        self.logger.debug('  t.shape     : %s', self.t.shape)
        self.logger.debug('  f0.shape    : %s', self.f0.shape)
        self.logger.debug('  sp.shape    : %s', self.sp.shape)
        self.logger.debug('  ap.shape    : %s', self.ap.shape)
        # ------------------------------------------------------
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


# MARK: NeuralNetworkWavTool
class NeuralNetworkWavTool(pywavtool.WavTool):
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

    _error: bool = False
    _output: str
    _input: str
    _stp: float
    _length: float
    _data: np.ndarray
    _range_data: np.ndarray
    _apply_data: np.ndarray
    _header: Any  # もとは pywavtool.Whd だが、使用しないのでAny
    _dat: Any  # もとは pywavtool.Dat だが、使用しないのでAny
    _export_wav: bool
    _export_features: bool
    _frame_period: int
    _feature_ext: str
    _f0: np.ndarray
    _sp: np.ndarray
    _ap: np.ndarray
    _range_f0: np.ndarray
    _range_sp: np.ndarray
    _range_ap: np.ndarray
    _apply_f0: np.ndarray
    _apply_sp: np.ndarray
    _apply_ap: np.ndarray
    _sample_rate: int
    _npz_path: Path

    def __init__(
        self,
        output: str,
        *,
        export_wav: bool = True,
        export_features: bool = True,
        frame_period: int = 5,
        feature_extension: str = '.npz',
        sample_rate: int = 44100,
    ) -> None:
        self._error = False
        self._output = output
        self._feature_ext = feature_extension
        self._export_wav = export_wav
        self._export_features = export_features
        self._frame_period = frame_period
        self._f0 = np.array([])
        self._sp = np.array([[]])
        self._ap = np.array([[]])
        self._sample_rate = sample_rate
        self._npz_path = Path(output).with_suffix(self._feature_ext)
        # 出力フォルダを作成
        Path(output).parent.mkdir(parents=True, exist_ok=True)

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
    def features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """WORLD特徴量 (f0, sp, ap) を返す。"""
        return self._f0, self._sp, self._ap

    @features.setter
    def features(self, features: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        self._f0, self._sp, self._ap = features

    @property
    def range_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """クロップされたWORLD特徴量 (f0, sp, ap) を返す。"""
        return self._range_f0, self._range_sp, self._range_ap

    @range_features.setter
    def range_features(self, features: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        self._range_f0, self._range_sp, self._range_ap = features

    @property
    def apply_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """エフェクト適用後のWORLD特徴量 (f0, sp, ap) を返す。"""
        return self._apply_f0, self._apply_sp, self._apply_ap

    @apply_features.setter
    def apply_features(self, features: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        self._apply_f0, self._apply_sp, self._apply_ap = features

    @property
    def overlap(self) -> float:
        """WAV先頭のオーバーラップ時間を返す。"""
        return self._envelope[7] if len(self._envelope) >= 8 else 0

    def applyData(self, stp: float, length_ms: float):
        """
        Parameters
        ----------
        stp : float
            入力wavの先頭のオフセットをmsで指定する。
        length : float
            datに追加する長さ(ms)
        """
        # 音量エンベロープの最後の値をクロスフェードに使用する。
        # NOTE: _ove が使用されてないけど大丈夫？
        _ove: float = self._envelope[7] if len(self._envelope) >= 8 else 0
        # 休符ではないとき
        if not self._error:
            self._applyRange(stp, length_ms)
            p, v = self._getEnvelopes(length_ms)
            self._applyEnvelope(p, v)
        # 休符のときは f0, sp, ap に0を代入
        else:
            n_frames = round(length_ms / self._frame_period)
            self._apply_f0 = np.zeros((n_frames,), dtype=np.float64)
            self._apply_sp = np.zeros((n_frames, self._sp.shape[1]), dtype=np.float64)
            self._apply_ap = np.zeros((n_frames, self._ap.shape[1]), dtype=np.float64)
        # TODO: 既存のnpzを読み込んで継ぎ足して、wavを出力する。
        overlap_samples = round(self.overlap / self._frame_period)
        if self._npz_path.exists():
            existing_f0, existing_sp, existing_ap = npzfile_to_world(self._npz_path)
            new_f0 = crossfade_world_features(existing_f0, self._apply_f0, overlap_samples)
            new_sp = crossfade_world_features(existing_sp, self._apply_sp, overlap_samples)
            new_ap = crossfade_world_features(existing_ap, self._apply_ap, overlap_samples)
        # npzが存在しない場合(先頭ノートの場合)は新規作成
        else:
            new_f0 = self._apply_f0
            new_sp = self._apply_sp
            new_ap = self._apply_ap
        # WORLD特徴量を npz ファイルに保存
        world_to_npzfile(new_f0, new_sp, new_ap, self._npz_path)
        # WAVファイルを出力する
        if self._export_wav:
            waveform = world_to_waveform(
                new_f0, new_sp, new_ap, self._sample_rate, frame_period=self._frame_period
            )
            waveform_to_wavfile(waveform, self._output, self._sample_rate, self._sample_rate)

    def inputCheck(self, input_wav_path: str | Path):
        """
        | 入力値が正しいかチェックします。
        | 正常値の場合、self._dataにwavの中身を最大1に正規化したfloatに変換して代入します。
        | 異常値の場合、self._errorをTrueにします。

        Parameters
        ----------
        input_ : str
            入力するwavのパス

        Notes (WavToolからの変更点)
        -----
        - npzファイルが存在する場合、npzからWORLD特徴量を読み込みます。
        - npzファイルが存在せずwavファイルだけ存在する場合、wavからWORLD特徴量を抽出します。

        TODO: 音量正規化の処理を追加する。
        """
        self._error = False
        input_wav_path = Path(input_wav_path)

        # 入力パスがwavファイルでない場合はエラー
        if input_wav_path.suffix.lower() != '.wav':
            self._error = True
            return
        # wavファイルとnpzファイルが存在するか確認
        input_npz_path = input_wav_path.with_suffix(self._feature_ext)
        input_wav_exists = input_wav_path.exists()
        input_npz_exists = input_npz_path.exists()

        waveform = np.array([])
        # npzファイルが存在する場合、npzからWORLD特徴量を読み込む
        if input_npz_exists:
            f0, sp, ap = npzfile_to_world(input_npz_path)
        # npzファイルが存在しない場合、wavからWORLD特徴量を抽出する
        elif input_wav_exists:
            waveform, sample_rate, _ = wavfile_to_waveform(input_wav_path)
            f0, sp, ap = waveform_to_world(
                waveform,
                sample_rate,
                frame_period=self._frame_period,
            )
            self._sample_rate = sample_rate
        # どちらのファイルも存在しない場合はエラー
        else:
            print(f'File not found: {input_wav_path} or {input_npz_path}')
            self._error = True
            return
        # WORLD特徴量をself._dataに代入
        self._f0 = f0
        self._sp = sp
        self._ap = ap

    def _applyRange(self, stp: float, length_ms: float) -> None:
        """stpを特徴量に反映して、self._range_features に代入します。

        Args:
            stp (float)     : 入力wavの先頭のオフセット(ms)
            length (float)  : 入力wavクロップ後、datに追加する長さ(ms)

        NOTE:
        frame 単位で計算するため本来必要な長さからずれる可能性あり。
        例えば、frame_period=5ms のとき、stp=7ms, length=12ms と指定しても、
        実際には stp=5ms, length=10ms になる。
        1frameぶん長めに取得すべきか検討必要。
        結合(append)後に長さが適切になるか検証必要。

        """
        start_frame = round(stp / self._frame_period)
        n_frames = round(length_ms / self._frame_period)
        # クロップした範囲をself._range_dataに代入
        self._range_f0 = self._f0[start_frame : start_frame + n_frames]
        self._range_sp = self._sp[start_frame : start_frame + n_frames, :]
        self._range_ap = self._ap[start_frame : start_frame + n_frames, :]

    def _applyEnvelope(self, p: list[float], v: list[int]):
        """times, volumesを特徴量に反映して、self._apply_features に代入します。

        Args:
            p (list): ボリューム制御の時刻のリスト
            v (list): ボリュームエンベロープ。ノート頭からms順に並べたポルタメントの音量値。エンベロープが2点の場合空配列。
        NOTE:
        frame 単位で計算するため本来必要な長さからずれる可能性あり。
        例えば、frame_period=5ms のとき、p=[0, 1, 0.5] と指定しても、
        実際には p=[0, 1, 0.5, 0, 0] になる。
        1frameぶん長めに取得すべきか検討必要。
        結合(append)後に長さが適切になるか検証必要。

        """
        # 休符などの例外では無音を生成する
        if len(p) == 0:
            self._apply_f0 = np.zeros_like(self._range_f0)
            self._apply_sp = np.zeros_like(self._range_sp)
            self._apply_ap = np.zeros_like(self._range_ap)
            return
        # p のほうが要素数が少ない場合は余ったvは無視する
        v = v[: len(p)]
        # 音量制御点の時刻をフレーム数に変換する
        p_by_frame = [round(t_ms / self._frame_period) for t_ms in p]
        # 特徴量のフレーム数
        n_frames = self._range_f0.shape[0]
        # 音量をもとに各フレームの倍率を算出する
        x = np.arange(n_frames)
        frame_weights = np.interp(x, p_by_frame, v) / 100.0
        print(
            f'f0.shape: {self._range_f0.shape}, sp.shape: {self._range_sp.shape}, ap.shape: {self._range_ap.shape}'
        )
        print(f'frame_weights.shape: {frame_weights.shape}')
        print('frame_weights:', frame_weights.tolist())
        frame_weights = frame_weights.reshape(-1, 1)  # sp, ap に合わせて2次元に変換
        # 各特徴量に倍率を適用する
        self._apply_f0 = self._range_f0 * frame_weights
        self._apply_sp = self._range_sp * frame_weights
        self._apply_ap = self._range_ap * frame_weights

    def _getEnvelopes(self, length: float) -> tuple[list, list]:
        """
        | エンベロープをノート頭からのms順に並べ、pとvのリストを返します。
        | エンベロープがパターンにマッチすることを事前に確認するのが条件です。
        Parameters
        ----------
        length : float
            datに追加する長さ(ms)

        Returns
        -------
        p :list of int
            Pstart P1 P2 P3 (P5) P4 Pendの順に並べたポルタメント。エンベロープが2点の場合空配列
        v: list of int
            ノート頭からms順に並べたポルタメントの音量値。エンベロープが2点の場合空配列

        ## PyWavTool.WavTool の _getEnvelopes からの変更点
        - p: フレームレートを無視して ms のまま返す
        - v: 変更なし

        Notes:
        ## エンベロープのパターン
            長さ2 : p1 p2
            長さ7 : p1 p2 p3 v1 v2 v3 v4
            長さ8 : p1 p2 p3 v1 v2 v3 v4 ove
            長さ9 : p1 p2 p3 v1 v2 v3 v4 ove p4
            長さ11: p1 p2 p3 v1 v2 v3 v4 ove p4 p5 v5
        p1,p2,p3,p4,p5,ove : float (ms)
        v1,v2,v3,v4,v5 : int (1-200)

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

        NOTE: p5 の位置は p2 と p3 の間であることに注意!
        """
        len_envelope = len(self._envelope)
        # エンベロープが2点のときは空配列を返す
        envelope = list(map(float, self._envelope))
        if len_envelope == 2:
            return [], []

        # エンベロープが2点以外で想定される点数のとき
        p: list[float]
        v: list[float]
        # 長さ7の時は [p1, p2, p3, v1, v2, v3, v4] のみ
        if len_envelope == 7:
            p1, p2, p3 = envelope[0:3]
            v1, v2, v3, v4 = envelope[3:7]
            p = [0, p1, p1 + p2, length - p3, length]  # 絶対時刻
            v = [0, v1, v2, v3, v4, 0]  # [0, v1, v2, v3, v4, 0]
        # 長さ8の時は ove が追加される
        elif len_envelope == 8:
            p1, p2, p3 = envelope[0:3]
            v1, v2, v3, v4 = envelope[3:7]
            p = [0, p1, p1 + p2, length - p3, length]  # 絶対時刻
            v = [0, v1, v2, v3, v4, 0]  # [0, v1, v2, v3, v4, 0]
            _ove = envelope[7]  # ove は使用しない
        # 長さ9の時は p4 が追加される
        elif len_envelope == 9:
            p1, p2, p3 = envelope[0:3]
            v1, v2, v3, v4 = envelope[3:7]
            _ove = envelope[7]
            p4 = envelope[8]
            p = [0, p1, p1 + p2, length - p4 - p3, length - p4, length]  # 絶対時刻
            v = [0, v1, v2, v3, v4, 0]
        # 長さが11の時は p5, v5 が追加される
        elif len_envelope == 11:
            p1, p2, p3 = envelope[0:3]
            v1, v2, v3, v4 = envelope[3:7]
            _ove = envelope[7]
            p4 = envelope[8]
            p5 = envelope[9]
            v5 = envelope[10]
            # NOTE: p5 の位置は p2 と p3 の間であることに注意!
            p = [0, p1, p1 + p2, p1 + p2 + p5, length - p4 - p3, length - p4, length]  # 絶対時刻
            # NOTE: p5 の位置は p2 と p3 の間であることに注意!
            v = [0, v1, v2, v5, v3, v4, 0]
        # それ以外の長さはエラー
        else:
            msg = f'Invalid envelope length (len_envelope={len_envelope}). The length must be 2, 7, 8, 9, or 11.'
            raise ValueError(msg)
        print('p:', p)
        print('v:', v)
        return p, v

    def write(self) -> None:
        """f0, sp, ap から WAV ファイルと NPZ ファイルを出力する。"""
        wav_out_path = Path(self._output)
        npz_out_path = Path(self._output).with_suffix(self._feature_ext)
        # サンプルレート
        sample_rate = self._sample_rate
        # WAVファイルを出力
        # WORLD特徴量をwavに変換
        waveform = world_to_waveform(
            self._apply_f0,
            self._apply_sp,
            self._apply_ap,
            frame_period=self._frame_period,
            sample_rate=sample_rate,
        )
        # wavをファイルに出力
        waveform_to_wavfile(
            waveform,
            wav_out_path,
            in_sample_rate=sample_rate,
            out_sample_rate=sample_rate,
        )
        # WORLD特徴量をnpzファイルに出力
        world_to_npzfile(
            self._apply_f0,
            self._apply_sp,
            self._apply_ap,
            npz_out_path,
        )


# MARK: NeuralNetworkRender
class NeuralNetworkRender(Render):
    """WAV出力の代わりに WORLD の特徴量ファイルを出力するのに用いる。"""

    _export_wav: bool
    _export_features: bool
    _use_neural_resampler: bool
    _use_neural_wavtool: bool
    _vocoder_model_dir: Path | str | None
    _vocoder_type: str
    _vocoder_feature_type: str
    _vocoder_vuv_threshold: float
    _vocoder_frame_period: int
    _force_wav_crossfade: bool

    def __init__(
        self,
        ust: Ust,
        *,
        voice_dir: str = '',
        cache_dir: str = '',
        output_file: str = '',
        logger: Logger | None = None,
        export_wav: bool,
        export_features: bool,
        vocoder_model_dir: Path | str | None = None,
        vocoder_type: str = 'usfgan',
        vocoder_feature_type: str = 'world',
        vocoder_vuv_threshold: float = 0.5,
        vocoder_frame_period: int = 5,
        use_neural_resampler: bool = True,
        use_neural_wavtool: bool = True,
        force_wav_crossfade: bool = False,
    ) -> None:
        # loggerを作成
        logger = setup_logger() if logger is None else logger
        super().__init__(
            ust,
            voice_dir=voice_dir,
            cache_dir=cache_dir,
            output_file=output_file,
            logger=logger,
        )
        self._export_wav = export_wav
        self._export_features = export_features
        self._use_neural_resampler = use_neural_resampler
        self._use_neural_wavtool = use_neural_wavtool
        self._vocoder_model_dir = vocoder_model_dir
        self._vocoder_type = vocoder_type
        self._vocoder_feature_type = vocoder_feature_type
        self._vocoder_vuv_threshold = vocoder_vuv_threshold
        self._vocoder_frame_period = vocoder_frame_period
        self._force_wav_crossfade = force_wav_crossfade

        # 引数の整合性をチェック
        if self._export_wav is False and self._export_features is False:
            msg = 'At least one of export_wav or export_features must be True.'
            raise ValueError(msg)
        if self._use_neural_resampler and self._vocoder_model_dir is None:
            msg = 'vocoder_model_dir must be specified when use_neural_resampler is True.'
            raise ValueError(msg)
        if self._use_neural_wavtool and self._vocoder_model_dir is None:
            msg = 'vocoder_model_dir must be specified when use_neural_wavtool is True.'
            raise ValueError(msg)
        if self._force_wav_crossfade and self._export_wav is False:
            msg = (
                'force_wav_crossfade=True かつ export_wav=False は実施不可です。'
                'export_wav=True に強制設定して処理を続行します。'
            )
            logger.warning(msg)
            self._export_wav = True
        if self._use_neural_resampler is True and self._use_neural_wavtool is True:
            msg = (
                '非推奨の組み合わせが検出されました。'
                'use_neural_resampler=True かつ use_neural_wavtool=True は非推奨です。'
                'use_neural_resampler=False に強制設定して処理を続行します (高速化のため)。'
            )
            logger.warning(msg)
            self._use_neural_resampler = False
        if self._use_neural_wavtool and self._export_features is False:
            msg = (
                '非推奨の組み合わせが検出されました。'
                'use_neural_wavtool=True かつ export_features=False は非推奨です。'
                'export_features=True に強制設定して処理を続行します (クロスフェード品質最大化のため)。'
            )
            logger.warning(msg)
            self._export_features = True

    def resamp(self, *, force: bool = False) -> None:
        """
        NeuralNetworkResampを使用してキャッシュファイルを生成する。

        Args:
            force: Trueの場合、キャッシュファイルがあっても生成する。

        """
        Path(self._cache_dir).mkdir(parents=True, exist_ok=True)
        for note in tqdm(self.notes, colour='cyan', desc='Resample', unit='note'):
            self.logger.debug('------------------------------------------------')
            if not note.require_resamp:
                continue
            if force or not Path(note.cache_path).is_file():
                self.logger.debug(
                    '%s %s %s %s %s %s %s %s %s %s %s %s %s',
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
                # Resampler で vocoder モデルを使わない場合
                if self._use_neural_resampler is False:
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
                        export_wav=self._export_wav,
                        export_features=self._export_features,
                    )
                # Resampler で vocoder モデルを使う場合
                elif self._use_neural_resampler is True:
                    assert self._vocoder_model_dir is not None  # 念のため型チェック
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
                        export_wav=self._export_wav,
                        export_features=self._export_features,
                        vocoder_model_dir=self._vocoder_model_dir,
                        vocoder_type=self._vocoder_type,
                        vocoder_feature_type=self._vocoder_feature_type,
                        vocoder_vuv_threshold=self._vocoder_vuv_threshold,
                        vocoder_frame_period=self._vocoder_frame_period,
                    )
                # _use_neural_resampler が True でも False でもない場合はエラー
                else:
                    error_msg = (
                        f'Invalid _use_neural_resampler value: {self._use_neural_resampler}'
                    )
                    raise ValueError(error_msg)
                resamp.resamp()
            else:
                self.logger.debug('Using cache (%s)', note.cache_path)
        self.logger.debug('------------------------------------------------')

    def append(self):
        """WorldFeatureWavToolを用いて各ノートのキャッシュWAVまたはWORLD特徴量を連結し、WAV出力する。"""
        if self._force_wav_crossfade is True:
            self.logger.info('WAVでクロスフェードします (force_wav_crossfade=True)')
            super().append()
            return

        out_dir = Path(self._output_file).parent
        if out_dir != Path():
            out_dir.mkdir(exist_ok=True, parents=True)
        # wav, npz, whd, dat ファイルがすでに存在する場合は削除
        out_wav_path = Path(self._output_file)
        out_wav_path.unlink(missing_ok=True)
        out_wav_path.with_suffix('.npz').unlink(missing_ok=True)
        out_wav_path.with_suffix('.wav.whd').unlink(missing_ok=True)
        out_wav_path.with_suffix('.wav.dat').unlink(missing_ok=True)

        # WavTool で vocoder モデルを使わない場合、かつ特徴量でクロスフェードしたい場合
        wavtool = NeuralNetworkWavTool(
            self._output_file,
            export_wav=True,
            export_features=True,
        )
        # 特徴量クロスフェードを実施
        for note in tqdm(self.notes, colour='magenta', desc='Append', unit='note'):
            if note.direct:
                self.logger.debug(
                    '%s %s %s %s',
                    note.input_path,
                    note.envelope,
                    note.stp + note.offset,
                    note.output_ms,
                )
                wavtool.inputCheck(note.input_path)
                wavtool.setEnvelope([float(item) for item in note.envelope.split(' ')])
                wavtool.applyData(note.stp + note.offset, note.output_ms)
            else:
                self.logger.debug(
                    '%s %s %s %s', note.cache_path, note.envelope, note.stp, note.output_ms
                )
                wavtool.inputCheck(note.cache_path)
                wavtool.setEnvelope([float(item) for item in note.envelope.split(' ')])
                wavtool.applyData(note.stp, note.output_ms)
        # ファイル出力
        wavtool.write()

    def clean(self) -> None:
        """キャッシュディレクトリと出力ファイルを削除する。"""
        if Path(self._cache_dir).is_dir():
            rmtree(self._cache_dir)
        Path(self._output_file).unlink(missing_ok=True)
        Path(self._output_file).with_suffix('.npz').unlink(missing_ok=True)


# MARK: Main functions
def main_as_resampler() -> None:
    """Resampler (伸縮器) として各ノートの wav 加工を行う。

    Args:

    """
    logger = setup_logger()

    parser = ArgumentParser(description='This module is Resampler for UTAU powered by world')
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
        'fixed_ms',
        help='offsetからみて通常伸縮しない長さ(ms)',
        nargs='?',
        default=0,
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


if __name__ == '__main__':
    pass

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
from logging import Logger
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory

import colored_traceback.auto  # noqa: F401
import numpy as np
import torch
import utaupy
from nnsvs.gen import predict_waveform
from PyUtauCli.projects.Render import Render
from PyUtauCli.projects.Ust import Ust
from tqdm.auto import tqdm

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
from resampler import NeuralNetworkResamp, WorldFeatureResamp
from resampler import main_resampler as _main_resampler
from util import get_device, load_vocoder_model, setup_logger
from wavtool import WorldFeatureWavTool
from wavtool import main_wavtool as _main_wavtool

# MARK: Utility functions


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
    
    # Check for non-finite values in the generated waveform and clean them
    if not np.isfinite(wav).all():
        logger.warning(
            'Generated waveform contains non-finite values (NaN/inf). '
            'Replacing with zeros to prevent audio errors.'
        )
        wav = np.where(np.isfinite(wav), wav, 0.0)
    
    # 生成した waveform を返す
    return wav


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
        if self._use_neural_resampler is True and self._vocoder_model_dir is None:
            msg = 'vocoder_model_dir must be specified when use_neural_resampler is True.'
            raise ValueError(msg)
        if self._use_neural_wavtool is True and self._vocoder_model_dir is None:
            msg = 'vocoder_model_dir must be specified when use_neural_wavtool is True.'
            raise ValueError(msg)
        # 不成立の組み合わせを修正
        if self._force_wav_crossfade is True and self._export_wav is False:
            msg = (
                'force_wav_crossfade=True かつ export_wav=False は実施不可です。'
                'export_wav=True に強制設定して処理を続行します。'
            )
            logger.warning(msg)
            self._export_wav = True
        # 非推奨の組み合わせを修正
        if self._use_neural_resampler is True and self._use_neural_wavtool is True:
            msg = (
                '非推奨の組み合わせが検出されました。'
                'use_neural_resampler=True かつ use_neural_wavtool=True は非推奨です。'
                'use_neural_resampler=False に強制設定して処理を続行します (レンダリング時間短縮のため)。'
            )
            logger.warning(msg)
            self._use_neural_resampler = False
        # TODO: デバッグが終わったらコメントアウトを解除する。(ここから)--------------------------------------
        # if self._use_neural_wavtool and self._export_features is False:
        #     msg = (
        #         '非推奨の組み合わせが検出されました。'
        #         'use_neural_wavtool=True かつ export_features=False は非推奨です。'
        #         'export_features=True に強制設定して処理を続行します (クロスフェード品質最大化のため)。'
        #     )
        #     logger.warning(msg)
        #     self._export_features = True
        # TODO: デバッグが終わったらコメントアウトを解除する。(ここまで)--------------------------------------
        # vocoder モデルを読み込む
        if self._use_neural_resampler is True or self._use_neural_wavtool is True:
            if self._vocoder_model_dir is None:
                msg = 'vocoder_model_dir must be specified when use_neural_resampler or use_neural_wavtool is True.'
                raise ValueError(msg)
            logger.info('Using vocoder model: %s', self._vocoder_model_dir)
            self.__init_vocoder()

    def __init_vocoder(self) -> None:
        """Vocoderモデルを読み込み、self._vocoder_model, self._vocoder_in_scaler, self._vocoder_config にセットする。"""
        if self._vocoder_model_dir is None:
            msg = 'vocoder_model_dir is not specified.'
            raise ValueError(msg)
        self._vocoder_model, self._vocoder_in_scaler, self._vocoder_config = load_vocoder_model(
            self._vocoder_model_dir
        )

    @property
    def vocoder_model(self) -> torch.nn.Module:
        """ボコーダーモデル"""
        return self._vocoder_model

    @property
    def vocoder_sample_rate(self) -> int:
        """ボコーダーモデルのwav出力サンプリング周波数"""
        return self._vocoder_config.data.sample_rate

    def resamp(self, *, force: bool = False) -> None:
        """
        NeuralNetworkResampを使用してキャッシュファイルを生成する。

        Args:
            force: Trueの場合、キャッシュファイルがあっても生成する。

        """
        Path(self._cache_dir).mkdir(parents=True, exist_ok=True)

        for note in tqdm(
            self.notes, mininterval=0.02, colour='cyan', desc='Resample', unit='note'
        ):
            print('\n---------------')
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

    def append(self) -> None:
        """WorldFeatureWavToolを用いて各ノートのキャッシュWAVまたはWORLD特徴量を連結し、WAV出力する。"""
        if self._force_wav_crossfade is True:
            self.logger.info('WAVでクロスフェードします (force_wav_crossfade=True)')
            super().append()
            return
        # 出力フォルダを作成
        out_dir = Path(self._output_file).parent
        out_dir.mkdir(exist_ok=True, parents=True)
        # wav, npz, whd, dat ファイルがすでに存在する場合は削除
        out_wav_path = Path(self._output_file)
        out_wav_path.unlink(missing_ok=True)
        out_wav_path.with_suffix('.npz').unlink(missing_ok=True)
        out_wav_path.with_suffix('.wav.whd').unlink(missing_ok=True)
        out_wav_path.with_suffix('.wav.dat').unlink(missing_ok=True)

        # 特徴量クロスフェードを実施
        for note in tqdm(
            self.notes, mininterval=0.02, colour='magenta', desc='Append', unit='note'
        ):
            print('\n---------------')
            # direct=True の場合は原音WAVをそのままクロスフェードする
            if note.direct is True:
                stp = note.stp + note.offset
                in_wav_path = note.input_path
            # それ以外の場合はキャッシュWAVまたは特徴量をクロスフェードする
            else:
                stp = note.stp
                in_wav_path = note.cache_path
            self.logger.debug(
                '%s %s %s %s %s',
                out_wav_path,
                in_wav_path,
                note.envelope,
                stp,
                note.output_ms,
            )
            # WorldFeatureWavTool を用いて特徴量を連結
            wavtool = WorldFeatureWavTool(
                output_wav=out_wav_path,
                input_wav=in_wav_path,
                stp=stp,
                length=note.output_ms,
                envelope=[float(item) for item in note.envelope.split(' ')],
                logger=self.logger,
            )
            # WAVと特徴量ファイル出力
            wavtool.append()

        # 必要に応じて vocoder を用いて wav を生成
        if self._use_neural_wavtool is True:
            print('---------------')
            self.logger.info('ニューラルボコーダーでWAVを生成します')
            f0, sp, ap = npzfile_to_world(out_wav_path.with_suffix('.npz'))
            # vocoder で wav を生成
            # WORLD 特徴量を NNSVS 用に変換
            mgc, lf0, vuv, bap = world_to_nnsvs(f0, sp, ap, self.vocoder_sample_rate)
            # モデルに渡す用に特徴量をまとめる
            multistream_features = (mgc, lf0, vuv, bap)
            # print(mgc.shape, lf0.shape, vuv.shape, bap.shape)
            # waveformを生成
            # NOTE: ここのsample_rate って vocoder のサンプルレートで大丈夫？
            waveform = predict_waveform(
                device=get_device(),
                multistream_features=multistream_features,
                vocoder=self._vocoder_model,
                vocoder_config=self._vocoder_config,
                vocoder_in_scaler=self._vocoder_in_scaler,
                sample_rate=self.vocoder_sample_rate,
                frame_period=self._vocoder_frame_period,
                use_world_codec=True,
                feature_type=self._vocoder_feature_type,
                vocoder_type='usfgan',
                vuv_threshold=self._vocoder_vuv_threshold,  # vuv 閾値設定はするけど使われないはず
            )
            
            # Check for non-finite values in the generated waveform and clean them
            if not np.isfinite(waveform).all():
                self.logger.warning(
                    'Generated waveform contains non-finite values (NaN/inf). '
                    'Replacing with zeros to prevent audio errors.'
                )
                waveform = np.where(np.isfinite(waveform), waveform, 0.0)

            # wav ファイルを書き出す
            waveform_to_wavfile(
                waveform,
                out_wav_path,
                in_sample_rate=self.vocoder_sample_rate,
                out_sample_rate=self.vocoder_sample_rate,
                resample_type='soxr_vhq',
            )

    def clean(self) -> None:
        """キャッシュディレクトリと出力ファイルを削除する。"""
        if Path(self._cache_dir).is_dir():
            rmtree(self._cache_dir)
        Path(self._output_file).unlink(missing_ok=True)
        Path(self._output_file).with_suffix('.npz').unlink(missing_ok=True)


# MARK: Main functions
def main_as_resampler() -> None:
    """Resampler (伸縮器) として各ノートの wav 加工を行う。"""
    _main_resampler()


def main_as_wavtool() -> None:
    """WavTool (結合器) として各ノートの wav 結合を行う。"""
    _main_wavtool()


def main_as_integrated_wavtool(path_ust_in: Path | str, path_wav_out: Path | str) -> None:
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
    logger.setLevel('INFO')
    # utaupyでUSTを読み取る
    ust_utaupy = utaupy.ust.load(path_ust_in)
    voice_dir = ust_utaupy.voicedir
    cache_dir = ust_utaupy.setting.get(
        'CacheDir',
        Path(__file__).parent / 'kuresampler.cache',
    )

    # 一時フォルダにustを出力してPyUtauCliで読み直す
    with TemporaryDirectory() as temp_dir:
        # utaupyでプラグインをustファイルとして保存する
        path_temp_ust = Path(temp_dir) / 'temp.ust'
        if isinstance(ust_utaupy, utaupy.utauplugin.UtauPlugin):
            ust_utaupy.as_ust().write(path_temp_ust)
        else:
            ust_utaupy.write(path_temp_ust)
        # PyUtauCliでustを読み込みなおす
        ust = Ust(str(path_temp_ust))
        ust.load()
    # WorldFeatureResampler + WorldFeatureWavTool + NeuralNetworkVocoder でレンダリング
    render = NeuralNetworkRender(
        ust,
        logger=logger,
        voice_dir=str(voice_dir),
        cache_dir=str(cache_dir),
        output_file=str(path_wav_out),
        export_wav=True,
        export_features=False,
        use_neural_resampler=False,
        use_neural_wavtool=False,
        vocoder_model_dir=None,
        force_wav_crossfade=False,
    )
    render.clean()
    render.resamp(force=True)
    print('---------------')
    render.append()
    print('---------------')


if __name__ == '__main__':
    pass

#!/usr/bin/env python3
# Copyright (c) 2025 oatsu
"""UTAU engine for smooth crossfades

# 方針
- UTAU の resampler として、各ノートの WORLD 特徴量を生成する。
- UTAU の wavtool として、各ノートの WORLD 特徴量をクロスフェード結合する。
- 結合した WORLD 特徴量をニューラルボコーダーに入力し、WAV を出力する。

"""

import logging
from argparse import ArgumentParser
from logging import Logger
from pathlib import Path
from shutil import rmtree

import colored_traceback.auto  # noqa: F401
import numpy as np
import PyRwu as pyrwu  # noqa: N813
from nnsvs.gen import predict_waveform
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
from resampler import NeuralNetworkResamp, WorldFeatureResamp
from util import get_device, load_vocoder_model, setup_logger
from wavtool import WorldFeatureWavTool

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

        # 特徴量クロスフェードを実施
        for note in tqdm(self.notes, colour='magenta', desc='Append', unit='note'):
            # direct=True の場合は原音WAVをそのままクロスフェードする
            if note.direct is True:
                stp = note.stp + note.offset
                in_wav_path = note.input_path
            # それ以外の場合はキャッシュWAVまたは特徴量をクロスフェードする
            else:
                stp = note.stp
                in_wav_path = note.cache_path
            self.logger.debug('%s %s %s %s', in_wav_path, note.envelope, stp, note.output_ms)
            # WorldFeatureWavTool を用いて特徴量を連結
            wavtool = WorldFeatureWavTool(
                output_wav=out_wav_path,
                input_wav=in_wav_path,
                stp=stp,
                length=note.output_ms,
                envelope=[float(item) for item in note.envelope.split(' ')],
            )
            # WAVと特徴量ファイル出力
            wavtool.append()

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

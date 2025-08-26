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

from tqdm.auto import tqdm

import colored_traceback.auto  # noqa: F401
import utaupy
from PyUtauCli.projects.Render import Render
from PyUtauCli.projects.Ust import Ust
from os.path import join, dirname

from PyRwu.resamp import Resamp
from tempfile import TemporaryDirectory
import logging
from logging import Logger
import sys
import os


class WorldFeatureResamp(Resamp):
    """WAVファイルの代わりにWORLDの特徴量をファイルに出力するResampler"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def resamp(self) -> tuple:
        """
        WAVファイルの代わりにWORLDの特徴量をファイルに出力する。
        """
        self.parseFlags()
        # print(time.time())
        self.getInputData()
        # print(time.time())
        self.stretch()
        # print(time.time())
        self.pitchShift()
        # print(time.time())
        self.applyPitch()
        # パラメータ確認
        self.logger.debug(f'  input_path  : {self.input_path}')
        self.logger.debug(f'  output_path : {self.output_path}')
        self.logger.debug(f'  framerate   : {self.framerate}')
        self.logger.debug(f'  t.shape     : {self.t.shape}')
        self.logger.debug(f'  f0.shape    : {self.f0.shape}')
        self.logger.debug(f'  sp.shape    : {self.sp.shape}')
        self.logger.debug(f'  ap.shape    : {self.ap.shape}')

        # TODO: WORLD特徴量を npy または npz ファイルに出力する処理を追加する。Path指定から。
        # TODO: 無音だったら無音のWORLD特徴量を出力する？
        return self.f0, self.sp, self.ap


class WorldFeatureRender(Render):
    """
    WAV出力の代わりに WORLD の特徴量ファイルを出力するのに用いる。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
                    logger=self.logger,
                )
                resamp.resamp()
            else:
                self.logger.info(f'Using cache ({note.cache_path})')
        self.logger.debug('------------------------------------------------')


def setup_logger() -> Logger:
    """Loggerを作成する。"""
    # my_package.my_moduleのみに絞ってsys.stderrにlogを出す
    logging.basicConfig(
        stream=sys.stdout,
        format='[%(filename)s] [%(levelname)s] %(message)s',
        level=logging.DEBUG,
    )
    return logging.getLogger(__name__)


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

    for note in ust.notes:
        print(f'[{note.num}] {note.lyric}')
    # PyUtauCli でレンダリング
    render = WorldFeatureRender(
        ust,
        logger=logger,
        voice_dir=str(voice_dir),
        cache_dir=str(cache_dir),
        output_file=str(path_wav_out),
    )
    render.clean()
    render.resamp()
    # render.append()


if __name__ == '__main__':
    render_wav_from_ust(join(dirname(__file__), 'test.ust'), join(dirname(__file__), 'output.wav'))

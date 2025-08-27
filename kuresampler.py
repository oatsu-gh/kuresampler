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
from os.path import dirname, join
from tempfile import TemporaryDirectory

import colored_traceback.auto  # noqa: F401
import pyworld as pw  # pyright: ignore[reportMissingTypeStubs]
import utaupy
from PyRwu import frq_io, settings, wave_io
from PyRwu.resamp import Resamp
from PyUtauCli.projects.Render import Render
from PyUtauCli.projects.Ust import Ust
from tqdm.auto import tqdm


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

    def getInputData_in_nnsvs_format(
        self,
        f0_floor: float = settings.PYWORLD_F0_FLOOR,
        f0_ceil: float = settings.PYWORLD_F0_CEIL,
        frame_period: float = settings.PYWORLD_PERIOD,
        q1: float = settings.PYWORLD_Q1,
        threshold: float = settings.PYWORLD_THRESHOLD,
    ) -> None:
        """
        # NOTE: 親クラスの getInputData() メソッドをもとに、NNSVSでの WORLD 特長量と一致するように改造。
        入力された音声データからworldパラメータを取得し、self._input_data, self._framerate, self._f0, self._sp, self._apを更新します。

        Notes
        -----
        | 音声データの取得方法を変更したい場合、このメソッドをオーバーライドしてください。
        | オーバーライドする際、self._input_dataはこれ以降の処理で使用しないため、更新しなくても問題ありません。

        Parameters
        ----------
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

        Raises
        ------
        FileNotFoundError
            input_pathにファイルがなかったとき
        TypeError
            input_pathで指定したファイルがwavではなかったとき
        """
        # pyworld の npz キャッシュファイルを使用する場合
        if settings.USE_PYWORLD_CACHE:
            self._getInputFromNpz(f0_floor, f0_ceil, frame_period, q1, threshold)
        # pyworld のキャッシュファイルを使用しない場合
        else:
            frq_path: str = os.path.splitext(self._input_path)[0] + '_wav.frq'
            # 原音のWAVファイルを切り出して、データとフレームレートを取得
            self._input_data, self._framerate = wave_io.read(
                self._input_path, self._offset, self._end_ms
            )

            ## f0 ----------------------------------------------------------
            # 周波数表FRQファイルが無い場合は新規作成する
            if not os.path.isfile(frq_path):
                input_data, framerate = wave_io.read(self._input_path, 0, 0)
                frq_io.write(input_data, frq_path, framerate)
            # 周波数表FRQファイルがもとからある場合、もしくは直前に新規作成された場合は読み込む
            if os.path.isfile(frq_path):
                self._f0, self._t = frq_io.read(
                    frq_path, self._offset, self._end_ms, self._framerate, frame_period
                )
            # FRQファイルをうまく作成できていなかった場合、pyworld で直接解析する
            else:
                self._f0, self._t = pw.harvest(  # pyright: ignore[reportAttributeAccessIssue]
                    self._input_data,
                    self._framerate,
                    f0_floor=f0_floor,
                    f0_ceil=f0_ceil,
                    frame_period=frame_period,
                )
                self._f0 = pw.stonemask(  # pyright: ignore[reportAttributeAccessIssue]
                    self._input_data, self._f0, self._t, self._framerate
                )
            ## spectrogram -------------------------------------------------
            self._sp = pw.cheaptrick(  # pyright: ignore[reportAttributeAccessIssue]
                self._input_data,
                self._f0,
                self._t,
                self._framerate,
                q1=q1,
                f0_floor=f0_floor,
            )
            ## aperiodicity -------------------------------------------------
            self._getAp(f0_floor, f0_ceil, frame_period, threshold)
            ##

    # TODO: stretch をオーバーライドして、単パラメータのみ伸縮できるようにする。


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

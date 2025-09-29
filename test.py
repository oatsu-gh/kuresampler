#!/usr/bin/env python3
# Copyright (c) 2025 oatsu
"""
PyWorld で WAV ファイルを読み取って、
PyRwu の形式と NNSVS の形式に変換してみる。

相互変換できるか調査して kuresampler の開発につなげる。
"""

from os import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import colored_traceback.auto  # noqa: F401
import utaupy
from PyUtauCli.projects.Render import Render as PyUtauCliRender
from PyUtauCli.projects.Ust import Ust

from convert import (
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
from kuresampler import NeuralNetworkRender, nnsvs_to_waveform
from util import load_vocoder_model, setup_logger

F0_FLOOR = 150
F0_CEIL = 700
FRAME_PERIOD = 5.0
SAMPLE_RATE = 48000


TEST_WAV_IN = Path('./test/_a_a_n_i_a_u_a_44100.wav')
TEST_WAV_OUT = Path('./test/test_out.wav')
TEST_NPZ_WORLD = Path('./test/test_out_worldfeatures.npz')
TEST_NPZ_NNSVS = Path('./test/test_out_nnsvsfeatures.npz')
TEST_N_ITER = 10
TEST_UST_IN = Path('./test/test.ust')
TEST_VOCODER_MODEL_DIR = Path('./models/usfGAN_EnunuKodoku_0826')


def test_convert(
    path_wav_in: Path = TEST_WAV_IN,
    path_wav_out: Path = TEST_WAV_OUT,
    path_world_npz: Path = TEST_NPZ_WORLD,
    path_nnsvs_npz: Path = TEST_NPZ_NNSVS,
) -> None:
    """全体の処理をする。"""
    print('wavefile_to_waveform ---------------------------------------------------------')
    # read WAV and convert from 44100 -> 48000 Hz
    inprocess_sample_rate = 48000
    waveform, original_sample_rate, _ = wavfile_to_waveform(path_wav_in, inprocess_sample_rate)
    assert original_sample_rate == 44100
    print('waveform.shape:', waveform.shape)
    print()

    print('waveform_to_wavfile ----------------------------------------------------------')
    # write WAV in 44100 Hz
    waveform_to_wavfile(waveform, path_wav_out, inprocess_sample_rate, original_sample_rate)
    print('output wavfile:', path_wav_out.resolve())
    print()

    print('waveform_to_world ------------------------------------------------------------')
    f0, sp, ap = waveform_to_world(waveform, inprocess_sample_rate, frame_period=5)
    print('f0.shape :', f0.shape)
    print('sp.shape :', sp.shape)
    print('ap.shape :', ap.shape)
    print()

    print('world_to_waveform ------------------------------------------------------------')
    waveform = world_to_waveform(f0, sp, ap, inprocess_sample_rate)
    print('waveform.shape:', waveform.shape)
    print()

    print('world_to_nnsvs ---------------------------------------------------------------')
    mgc, lf0, vuv, bap = world_to_nnsvs(f0, sp, ap, inprocess_sample_rate)
    print('mgc.shape :', mgc.shape)
    print('lf0.shape :', lf0.shape)
    print('vuv.shape :', vuv.shape)
    print('bap.shape :', bap.shape)
    print()

    print('nnsvs_to_world ---------------------------------------------------------------')
    f0, sp, ap = nnsvs_to_world(mgc, lf0, vuv, bap, inprocess_sample_rate)
    print('f0.shape :', f0.shape)
    print('sp.shape :', sp.shape)
    print('ap.shape :', ap.shape)
    print()

    print('world_to_npzfile ------------------------------------------------------------')
    world_to_npzfile(f0, sp, ap, path_world_npz)
    print('output npzfile:', path_world_npz.resolve())
    print()

    print('npzfile_to_world ------------------------------------------------------------')
    f0, sp, ap = npzfile_to_world(path_world_npz)
    print('f0.shape :', f0.shape)
    print('sp.shape :', sp.shape)
    print('ap.shape :', ap.shape)
    print()

    print('nnsvs_to_npzfile ------------------------------------------------------------')
    nnsvs_to_npzfile(mgc, lf0, vuv, bap, path_nnsvs_npz)
    print('output npzfile:', path_nnsvs_npz.resolve())
    print()

    print('npzfile_to_nnsvs ------------------------------------------------------------')
    mgc, lf0, vuv, bap = npzfile_to_nnsvs(path_nnsvs_npz)
    print('mgc.shape :', mgc.shape)
    print('lf0.shape :', lf0.shape)
    print('vuv.shape :', vuv.shape)
    print('bap.shape :', bap.shape)
    print()


def test_vocoder_model(
    vocoder_model_dir: Path = TEST_VOCODER_MODEL_DIR,
    path_wav_in: Path = TEST_WAV_IN,
    path_wav_out: Path = TEST_WAV_OUT,
) -> None:
    """Vocoderモデルを読み取れるか、特徴量からのWAV合成ができるかをテストする。"""
    print('test_vocoder_model ---------------------------------------------------------')
    print('Loading vocoder model...')
    vocoder_model, vocoder_in_scaler, vocoder_config = load_vocoder_model(vocoder_model_dir)
    print('type(vocoder_model):', type(vocoder_model))
    print('type(vocoder_in_scaler):', type(vocoder_in_scaler))
    print('type(vocoder_config):', type(vocoder_config))

    # TODO: WORLD 特徴量からvocoder_model を通してWAVを生成するテストをつくる。
    inprocess_sample_rate = 48000
    print('Converting wavfile to nnsvs-world features...')
    waveform_in, _, _ = wavfile_to_waveform(path_wav_in, out_sample_rate=inprocess_sample_rate)
    f0, sp, ap = waveform_to_world(waveform_in, inprocess_sample_rate)
    mgc, lf0, vuv, bap = world_to_nnsvs(f0, sp, ap, inprocess_sample_rate)
    print('Rendering waveform with vocoder model...')
    waveform_out = nnsvs_to_waveform(mgc, lf0, vuv, bap, vocoder_model_dir)
    print('Exporting wavefile...')
    output_sample_rate = 48000
    waveform_to_wavfile(waveform_out, path_wav_out, inprocess_sample_rate, output_sample_rate)
    print('Output wavfile:', path_wav_out.resolve())
    print('Vocoder model test completed!')
    print()


def test_performance(path_wav_in: Path | str = TEST_WAV_IN, n_iter: int = TEST_N_ITER) -> None:
    """実行時間の計測をする。

    ボトルネックになりそうな関数
    - waveform_to_world: resample_type = ['soxr_vhq', 'soxr_hq', 'kaiser_best']
    """
    from time import time  # noqa: PLC0415

    def measure_time(_func, _n_iter, *_args, **_kwargs) -> Any:
        """関数の実行速度を評価する。
        Args:
            func: 評価したい関数
            n_iter: 実行回数
            *args: 関数に渡す位置引数
            **kwargs: 関数に渡すキーワード引数
        """
        print('---------------------------------------------')
        print(f'{_func.__name__} (x{_n_iter})')
        print('args:', _args)
        print('kwargs:', _kwargs)
        t_start = time()
        result = None
        for _ in range(_n_iter):
            result = _func(*_args, **_kwargs)
        t_end = time()
        process_time = round((t_end - t_start) * 1000, 1)
        print('process_time:', process_time, 'ms')
        return result

    original_sample_rate = 44100
    inprocess_sample_rate = 48000
    resample_types = [
        'soxr_vhq',
        'soxr_hq',
        'soxr_mq',
        'soxr_lq',
        'kaiser_best',
        'kaiser_fast',
        'scipy',
    ]
    # WAV読み取り_サンプルレート変換なし
    waveform, _, _ = measure_time(
        wavfile_to_waveform,
        n_iter,
        path_wav_in,
        original_sample_rate,
    )
    # WAV読み取り_サンプルレート変換あり
    for res_type in resample_types:
        waveform, _, _ = measure_time(
            wavfile_to_waveform,
            n_iter,
            path_wav_in,
            inprocess_sample_rate,
            resample_type=res_type,
        )
    # 特徴量抽出
    f0_extractor = ['dio', 'harvest']
    for extractor in f0_extractor:
        f0, sp, ap = measure_time(
            waveform_to_world,
            n_iter,
            waveform=waveform,
            sample_rate=inprocess_sample_rate,
            frame_period=5.0,
            f0_extractor=extractor,
        )


def test_resampler_and_wavtool(
    path_ust_in: Path | str = TEST_UST_IN,
    path_wav_out: Path | str = TEST_WAV_OUT,
    model_dir: Path | str = TEST_VOCODER_MODEL_DIR,
) -> None:
    """NeuralNetworkResampler で UST から WAV を生成するテストを行う。

    Args:
        path_ust_in: UST ファイルのパス
        path_wav_out: 出力する WAV ファイルのパス
        model_dir: ニューラルボコーダーモデルのディレクトリ
    """
    logger = setup_logger()
    logger.setLevel('DEBUG')
    # utaupyでUSTを読み取る
    ust_utaupy = utaupy.ust.load(path_ust_in)
    voice_dir = ust_utaupy.voicedir
    # ust_path = ust.setting.get('Project')
    cache_dir = ust_utaupy.setting.get(
        'CacheDir',
        Path(__file__).parent / 'kuresampler.cache',
    )
    # path_wav_out = ust.setting.get('OutFile', 'output.wav')

    # 一時フォルダにustを出力してPyUtauCliで読み直す
    with TemporaryDirectory() as temp_dir:
        # utaupyでプラグインをustファイルとして保存する
        path_temp_ust = Path(temp_dir) / 'temp.ust'
        if isinstance(ust_utaupy, utaupy.utauplugin.UtauPlugin):
            ust_utaupy.as_ust().write(path_temp_ust)
        else:
            ust_utaupy.write(path_temp_ust)
        # pyutaucliでustを読み込みなおす
        ust = Ust(str(path_temp_ust))
        ust.load()

    # ニューラルボコーダーを使う場合、ResampとWavToolのクラスを差し替える
    print('------------------------------------------------------------')
    print('PyRwu.Resamp (wav) + PyWavTool.WavTool')
    print('------------------------------------------------------------')
    """
    PyUtauCli.projects.Render.Render のテスト
    PyRwu.Resamp + PyWavTool.WavTool
    - Renderの出力: wavのみ
    - WavToolの入力: wavのみ
    """
    render = PyUtauCliRender(
        ust,
        logger=logger,
        voice_dir=str(voice_dir),
        cache_dir=str(cache_dir),
        output_file=str(path_wav_out).replace('.wav', '_pyrwu_wav_pywavtool.wav'),
    )
    render.clean()
    render.resamp(force=True)
    render.append()

    print('------------------------------------------------------------')
    print('NeuralNetworkResamp (wav) + PyWavTool.WavTool (wav crossfade)')
    print('------------------------------------------------------------')
    """
    kersamp.NeuralNetworkRender のテスト
    Case 1: NeuralNetworkResamp + PyWavTool.WavTool
    - WorldFeatureResamp の出力: wav + npz
    - WavToolの入力: wav
    """
    render = NeuralNetworkRender(
        ust,
        logger=logger,
        voice_dir=str(voice_dir),
        cache_dir=str(cache_dir),
        output_file=str(path_wav_out).replace('.wav', '_nnresamp_wav_pywavtool.wav'),
        export_wav=True,
        export_features=False,
        use_neural_resampler=True,
        use_neural_wavtool=False,
        vocoder_model_dir=model_dir,
        force_wav_crossfade=True,  # use PyWavTool.WavTool
    )
    render.clean()
    render.resamp(force=True)
    render.append()

    print('------------------------------------------------------------')
    print('NeuralNetworkResamp (wav) + PyWavTool.WavTool (wav crossfade)')
    print('------------------------------------------------------------')
    """
    kersamp.NeuralNetworkRender のテスト
    Case 1: NeuralNetworkResamp + PyWavTool.WavTool
    - WorldFeatureResamp の出力: wav + npz
    - WavToolの入力: wav
    """
    render = NeuralNetworkRender(
        ust,
        logger=logger,
        voice_dir=str(voice_dir),
        cache_dir=str(cache_dir),
        output_file=str(path_wav_out).replace('.wav', '_nnresamp_wav_pywavtool.wav'),
        export_wav=True,
        export_features=False,
        use_neural_resampler=True,
        use_neural_wavtool=False,
        vocoder_model_dir=model_dir,
        force_wav_crossfade=True,  # use PyWavTool.WavTool
    )
    render.clean()
    render.resamp(force=True)
    render.append()

    print('------------------------------------------------------------')
    print('NeuralNetworkResamp (wav) + NeuralNetworkRender (w/o vocoder-model)')
    print('------------------------------------------------------------')
    render = NeuralNetworkRender(
        ust,
        logger=logger,
        voice_dir=str(voice_dir),
        cache_dir=str(cache_dir),
        output_file=str(path_wav_out).replace('.wav', '_nnresamp_wav_nnwavtool_nomodel.wav'),
        export_wav=True,
        export_features=False,
        use_neural_resampler=True,  # Resampler with neural vocoder
        use_neural_wavtool=False,  # WavTool without neural vocoder
        vocoder_model_dir=model_dir,  # no vocoder model
        force_wav_crossfade=False,  # use NeuralNetworkWavTool
    )
    render.clean()
    render.resamp(force=True)
    render.append()

    print('------------------------------------------------------------')
    print('NeuralNetworkResamp (npz) + WorldFeatureWavTool (w/o vocoder-model)')
    print('------------------------------------------------------------')
    render = NeuralNetworkRender(
        ust,
        logger=logger,
        voice_dir=str(voice_dir),
        cache_dir=str(cache_dir),
        output_file=str(path_wav_out).replace('.wav', '_nnresamp_npz_nnwavtool_nomodel.wav'),
        export_wav=False,
        export_features=True,  # export npz
        use_neural_resampler=False,  # Resampler without neural vocoder
        use_neural_wavtool=False,  # WavTool without neural vocoder
        vocoder_model_dir=None,  # no vocoder model
        force_wav_crossfade=False,  # use NeuralNetworkWavTool
    )
    render.clean()
    render.resamp(force=True)
    render.append()

    # print('------------------------------------------------------------')
    # print('NeuralNetworkResamp (wav) + WorldFeatureWavTool (w/ vocoder-model)')
    # print('------------------------------------------------------------')
    # render = NeuralNetworkRender(
    #     ust,
    #     logger=logger,
    #     voice_dir=str(voice_dir),
    #     cache_dir=str(cache_dir),
    #     output_file=str(path_wav_out).replace('.wav', '_wfresamp_wav_wfwavtool_withVocoder.wav'),
    #     export_wav=True,
    #     export_features=False,
    #     use_neural_resampler=False,
    #     use_neural_wavtool=True,
    #     vocoder_model_dir=model_dir,
    #     force_wav_crossfade=False,
    # )
    # render.clean()
    # render.resamp(force=True)
    # render.append()

    # print('------------------------------------------------------------')
    # print('NeuralNetworkResamp (npz) + WorldFeatureWavTool (w/ vocoder-model)')
    # print('------------------------------------------------------------')
    # render = NeuralNetworkRender(
    #     ust,
    #     logger=logger,
    #     voice_dir=str(voice_dir),
    #     cache_dir=str(cache_dir),
    #     output_file=str(path_wav_out).replace(
    #         '.wav', '_wfresamp_npz_wfwavtool_from_npz_withVocoder.wav'
    #     ),
    #     export_wav=False,
    #     export_features=True,
    #     use_neural_resampler=False,
    #     use_neural_wavtool=True,
    #     vocoder_model_dir=model_dir,
    #     force_wav_crossfade=False,
    # )
    # render.clean()
    # render.resamp(force=True)
    # render.append()


if __name__ == '__main__':
    chdir(Path(__file__).parent)  # カレントディレクトリをこのファイルのある場所に変更する
    # test(_a_a_n_i_a_u_a_44100.wav)
    test_wav = Path('./data/_a_a_n_i_a_u_a_44100.wav')
    if not test_wav.is_file():
        error_msg = f'Test WAV file ({test_wav}) not found.'
        raise FileNotFoundError(error_msg)

    # general function test
    # test_convert(
    #     Path('./../data/_a_a_n_i_a_u_a_44100.wav'),
    #     Path('./../data/test_convert_world_out.wav'),
    #     Path('./../data/test_convert_out_worldfeatures.npz'),
    #     Path('./../data/test_convert_out_nnsvsfeatures.npz'),
    # )

    # function performance test
    # test_performance(
    #     Path('./../data/_a_a_n_i_a_u_a_44100.wav'),
    #     n_iter=10,
    # )

    # test vocoder model
    # test_vocoder_model(
    #     Path('./../models/usfGAN_EnunuKodoku_0826'),
    #     Path('./../data/_a_a_n_i_a_u_a_44100.wav'),
    #     Path('./../test/test_vocoder_out.wav'),
    # )

    # test resampler
    test_resampler_and_wavtool(
        Path('./test/test.ust'),
        Path('./test/test_resampler_out.wav'),
        Path('./models/usfGAN_EnunuKodoku_0826'),
    )

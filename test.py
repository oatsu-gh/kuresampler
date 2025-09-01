#!/usr/bin/env python3
# Copyright (c) 2025 oatsu
"""
PyWorld で WAV ファイルを読み取って、
PyRwu の形式と NNSVS の形式に変換してみる。

相互変換できるか調査して kuresampler の開発につなげる。
"""

from os.path import isfile
from pathlib import Path

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
from kuresampler import (
    load_vocoder_model,
    nnsvs_to_waveform,
)

F0_FLOOR = 150
F0_CEIL = 700
FRAME_PERIOD = 5.0
SAMPLE_RATE = 48000


def test_convert(
    path_wav_in: Path, path_wav_out: Path, path_world_npz: Path, path_nnsvs_npz: Path
) -> None:
    """全体の処理をする。"""
    # wave, spectrogram, mgc, lf0, vuv, bap = read_wav_nnsvs(path_wav)
    # print('read_wav_nnsvs ----------------------------------')
    # print('- wave.shape        :', wave.shape)
    # print('- spectrogram.shape :', spectrogram.shape)
    # print('- mgc.shape         :', mgc.shape)
    # print('- lf0.shape         :', lf0.shape)
    # print('- vuv.shape         :', vuv.shape)
    # print('- bap.shape         :', bap.shape)

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
    f0, sp, ap = waveform_to_world(waveform, inprocess_sample_rate, frame_period=5.0)  # type: ignore
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


def test_vocoder_model(vocoder_model_dir: Path, path_wav_in: Path, path_wav_out: Path) -> None:
    """Vocoderモデルを読み取れるか、特徴量からのWAV合成ができるかをテストする。"""
    print('test_vocoder_model ---------------------------------------------------------')
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
    waveform_out = nnsvs_to_waveform(mgc, lf0, vuv, bap, vocoder_model_dir, inprocess_sample_rate)
    print('Exporting wavefile...')
    output_sample_rate = 48000
    waveform_to_wavfile(waveform_out, path_wav_out, inprocess_sample_rate, output_sample_rate)
    print('Output wavfile:', path_wav_out.resolve())
    print('Vocoder model test completed!')


def test_performance(wav_path: Path | str, n_iter: int) -> None:
    """実行時間の計測をする。

    ボトルネックになりそうな関数
    - waveform_to_world: resample_type = ['soxr_vhq', 'soxr_hq', 'kaiser_best']
    """
    from time import time

    def measure_time(_func, _n_iter, *_args, **_kwargs) -> any:
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
        wav_path,
        original_sample_rate,
    )
    # WAV読み取り_サンプルレート変換あり
    for res_type in resample_types:
        waveform, _, _ = measure_time(
            wavfile_to_waveform,
            n_iter,
            wav_path,
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


if __name__ == '__main__':
    # test(_a_a_n_i_a_u_a_44100.wav)
    if not isfile('./data/_a_a_n_i_a_u_a_44100.wav'):
        raise FileNotFoundError('Test WAV file not found.')
    # general function test
    test_convert(
        Path('./data/_a_a_n_i_a_u_a_44100.wav'),
        Path('./data/test_convert_world_out.wav'),
        Path('./data/test_convert_out_worldfeatures.npz'),
        Path('./data/test_convert_out_nnsvsfeatures.npz'),
    )
    # function performance test
    # test_performance(
    #     Path('./data/_a_a_n_i_a_u_a_44100.wav'),
    #     n_iter=10,
    # )

    # test_vocoder_model
    test_vocoder_model(
        Path('./models/usfGAN_EnunuKodoku_0826'),
        Path('./data/_a_a_n_i_a_u_a_44100.wav'),
        Path('./data/test_vocoder_out.wav'),
    )

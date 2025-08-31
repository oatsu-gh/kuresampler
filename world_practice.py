#!/usr/bin/env python3
# Copyright (c) 2025 oatsu
"""
PyWorld で WAV ファイルを読み取って、
PyRwu の形式と NNSVS の形式に変換してみる。

相互変換できるか調査して kuresampler の開発につなげる。
"""

from pathlib import Path

import librosa
import numpy as np
import pysptk
import pyworld
import soundfile as sf
from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.util import apply_delta_windows
from nnsvs.multistream import get_windows

F0_FLOOR = 150
F0_CEIL = 700
FRAME_PERIOD = 5.0
SAMPLE_RATE = 48000

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


def read_wav_nnsvs(
    wav_path: str | Path,
    f0_extractor: str = 'harvest',
    f0_floor: float = F0_FLOOR,
    f0_ceil: float = F0_CEIL,
    frame_period: float = FRAME_PERIOD,
    mgc_order: int = 59,
    num_windows: int = 3,
    interp_unvoiced_aperiodicity: bool = True,
    sample_rate: int = SAMPLE_RATE,
    d4c_threshold: float = 0.85,
    dynamic_features_flags: None | list[bool] = None,
    use_world_codec: bool = False,
    use_mcep_aperiodicity: bool = False,
    mcep_aperiodicity_order: int = 24,
    res_type: str = 'scipy',
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    WAVファイルを読み取って、NNSVS用のWORLD特徴量を返す。

    output: NNSVS用のWORLD特徴量

    """
    print('------------------------------------------')
    print(f'Reading WAV ({wav_path})')

    x, fs = sf.read(wav_path)
    assert np.max(x) <= 1.0
    assert x.dtype == np.float64

    print(f'- n_samples   : {len(x)}')
    print(f'- original fs : {fs}')
    print(f'- target fs   : {sample_rate}')

    print('------------------------------------------')
    print('Converting WAV to WORLD features with NNSVS format')

    if fs != sample_rate:
        x = librosa.resample(x, orig_sr=fs, target_sr=sample_rate, res_type=res_type)
        fs = sample_rate

    if f0_extractor == 'parselmouth':
        import parselmouth

        assert f0_floor is not None and f0_ceil is not None, 'must be set manually'
        harvest_num_frames = int(int(1000 * len(x) / fs) / frame_period) + 1
        f0 = (
            parselmouth.Sound(x.astype(np.float64), fs)
            .to_pitch_ac(
                time_step=frame_period * 0.001,
                voicing_threshold=0.6,
                very_accurate=False,
                pitch_floor=f0_floor,
                pitch_ceiling=f0_ceil,
            )
            .selected_array['frequency']
        )
        pad = int(np.round((3 / f0_floor) / (frame_period * 0.001)))
        f0 = np.pad(f0, [[0, pad]], mode='constant')
        if len(f0) > harvest_num_frames:
            f0 = f0[:harvest_num_frames]
        elif len(f0) < harvest_num_frames:
            f0 = np.pad(f0, (0, harvest_num_frames - len(f0)), mode='constant')

        timeaxis = np.arange(harvest_num_frames) * frame_period * 0.001
    elif f0_extractor == 'harvest':
        f0, timeaxis = pyworld.harvest(  # type: ignore
            x, fs, frame_period=frame_period, f0_floor=f0_floor, f0_ceil=f0_ceil
        )
    elif f0_extractor == 'dio':
        f0, timeaxis = pyworld.dio(  # type: ignore
            x, fs, frame_period=frame_period, f0_floor=f0_floor, f0_ceil=f0_ceil
        )
        f0 = pyworld.stonemask(x, f0, timeaxis, fs)  # type: ignore
    else:
        raise ValueError(f'unknown f0 extractor: {f0_extractor}')

    # Workaround for https://github.com/r9y9/nnsvs/issues/7
    f0 = np.maximum(f0, 0)

    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)  # pyright: ignore[reportAttributeAccessIssue]
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs, threshold=d4c_threshold)  # pyright: ignore[reportAttributeAccessIssue]

    if np.isnan(aperiodicity).any():
        print(wav_path)
        print(f0_floor, f0_ceil, aperiodicity.shape, fs)
        print(np.isnan(aperiodicity).sum())
        print(aperiodicity)
        raise RuntimeError('Aperiodicity has NaN')

    lf0 = f0[:, np.newaxis].copy()
    nonzero_indices = np.nonzero(lf0)
    lf0[nonzero_indices] = np.log(f0[:, np.newaxis][nonzero_indices])
    if f0_extractor == 'harvest':
        # https://github.com/mmorise/World/issues/35#issuecomment-306521887
        vuv = (aperiodicity[:, 0] < 0.5).astype(np.float32)[:, None]
    else:
        vuv = (lf0 != 0).astype(np.float32)

    # F0 -> continuous F0
    lf0 = interp1d(lf0, kind='slinear')

    if use_world_codec:
        mgc = pyworld.code_spectral_envelope(spectrogram, fs, mgc_order + 1)  # type: ignore
    else:
        mgc = pysptk.sp2mc(spectrogram, order=mgc_order, alpha=pysptk.util.mcepalpha(fs))
    # NOTE: used as the target for post-filters
    spectrogram = np.log(spectrogram)

    # Post-processing for aperiodicy
    # ref: https://github.com/MTG/WGANSing/blob/mtg/vocoder.py
    if interp_unvoiced_aperiodicity:
        is_voiced = (vuv > 0).reshape(-1)
        if not np.any(is_voiced):
            pass  # all unvoiced, do nothing
        else:
            for k in range(aperiodicity.shape[1]):
                aperiodicity[~is_voiced, k] = np.interp(
                    np.where(~is_voiced)[0],
                    np.where(is_voiced)[0],
                    aperiodicity[is_voiced, k],
                )

    if use_mcep_aperiodicity:
        bap = pysptk.sp2mc(
            aperiodicity,
            order=mcep_aperiodicity_order,
            alpha=pysptk.util.mcepalpha(fs),
        )
    else:
        bap = pyworld.code_aperiodicity(aperiodicity, fs)  # type: ignore

    # Compute delta features if necessary
    windows = get_windows(num_windows)
    if dynamic_features_flags is not None:
        if dynamic_features_flags[0] is not None:
            mgc = apply_delta_windows(mgc, windows)

    # Concat features
    features = np.hstack((mgc, lf0, vuv, bap)).astype(np.float32)
    pf_features = np.hstack((spectrogram, lf0, vuv, bap)).astype(np.float32)

    # Align waveform and features
    waveform = x.astype(np.float32)

    assert np.isfinite(features).all()
    assert np.isfinite(waveform).all()
    assert np.isfinite(pf_features).all()

    return waveform, spectrogram, mgc, lf0, vuv, bap


def main() -> None:
    """全体の処理をする。"""
    path_wav = '_ああんいあうあ.wav'

    wave, spectrogram, mgc, lf0, vuv, bap = read_wav_nnsvs(path_wav)
    print('NNSVS reader result ----------------------------------')
    print('- wave.shape        :', wave.shape)
    print('- spectrogram.shape :', spectrogram.shape)
    print('- mgc.shape         :', mgc.shape)
    print('- lf0.shape         :', lf0.shape)
    print('- vuv.shape         :', vuv.shape)
    print('- bap.shape         :', bap.shape)

    print('Converter result -------------------------------------')
    sample_rate = 48000
    waveform = wavfile_to_waveform(path_wav, target_sample_rate=sample_rate)
    print('waveform.shape:', waveform.shape)

    f0, sp, ap = waveform_to_world(waveform, sample_rate, frame_period=5.0)  # type: ignore
    print('- f0.shape :', f0.shape)
    print('- sp.shape :', sp.shape)
    print('- ap.shape :', ap.shape)

    mgc2, lf02, vuv2, bap2 = world_to_nnsvs(f0, sp, ap, sample_rate)
    print('- mgc2.shape       :', mgc2.shape)
    print('- lf02.shape       :', lf02.shape)
    print('- vuv2.shape       :', vuv2.shape)
    print('- bap2.shape       :', bap2.shape)


if __name__ == '__main__':
    main()

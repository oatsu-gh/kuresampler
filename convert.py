#!/usr/bin/env python3
# Copyright (c) 2025 oatsu
"""
音声まわりのファイルや特徴量を相互変換する。

wavfile  (path)         <-> waveform    (np.ndarray)
waveform (np.ndarray)   <-> world       (f0, sp, ap) # NOTE: Use WORLD
world    (f0, sp, ap)   <-> npzfile     (path)
world    (f0, sp, ap)   <-> nnsvs-world (mgc, lf0, vuv, bap)
nnsvs-world (mgc, lf0, vuv, bap) <-> npzfile (path)
nnsvs-world (mgc, lf0, vuv, bap) <-> waveform (np.ndarray) # NOTE: Use WORLD or GAN vocoder
"""

from pathlib import Path

import librosa
import numpy as np
import pyworld
import soundfile as sf

# WAV settings ---------------------
DEFAULT_WAV_DTYPE = np.float64
DEFAULT_RESAMPLE_TYPE = 'soxr_hq'  # [soxr_vhq, soxr_hq, kaiser_best] あたりから選ぶとよい。https://librosa.org/doc/0.11.0/generated/librosa.resample.html#librosa-resample
# WORLD settings -------------------
DEFAULT_FRAME_PERIOD = 5.0  # ms
DEFAULT_F0_FLOOR = 50.0
DEFAULT_F0_CEIL = 2000
DEFAULT_D4C_THRESHOLD = 0.85
DEFAULT_FFT_SIZE = 512
# ----------------------------------


def wavfile_to_waveform(
    wav_path: Path,
    *,
    target_sample_rate: int,
    resample_type: str = DEFAULT_RESAMPLE_TYPE,
    dtype: type = DEFAULT_WAV_DTYPE,
) -> np.ndarray:
    """Convert a WAV file to a waveform (numpy array).

    Args:
        wav_path           (Path)    : The path to the WAV file.
        target_sample_rate (int)     : Sample rate for the output waveform.
        dtype              (np.dtype): dtype for the output waveform.
        resample_type      (str)     : Resampling method. Select from `res_type` options of `librosa.resample`. (kaiser_best / kaiser_fast が良さげ？)

    Returns:
        waveform    (np.ndarray): The waveform as a numpy array.
        sample_rate (int): The sample rate of the audio.
    """
    waveform, original_sample_rate = sf.read(wav_path, dtype=dtype)
    if original_sample_rate != target_sample_rate:
        waveform = librosa.resample(
            waveform,
            orig_sr=original_sample_rate,
            target_sr=target_sample_rate,
            res_type=resample_type,
        )
    return waveform, target_sample_rate


def waveform_to_wavfile(
    waveform: np.ndarray,
    wav_path: Path,
    *,
    original_sample_rate: int,
    target_sample_rate: int,
    resample_type: str = DEFAULT_RESAMPLE_TYPE,
    dtype: type = DEFAULT_WAV_DTYPE,
) -> None:
    """Convert a waveform (numpy array) to a WAV file.

    Args:
        waveform             (np.ndarray): The waveform as a numpy array.
        original_sample_rate (int)       : The original sample rate of the audio.
        target_sample_rate   (int)       : The target sample rate for the output WAV file.
        wav_path             (Path)      : The path to the output WAV file.
    """
    if original_sample_rate != target_sample_rate:
        waveform = librosa.resample(
            waveform,
            orig_sr=original_sample_rate,
            target_sr=target_sample_rate,
            res_type=resample_type,
        )
    sf.write(wav_path, waveform.astype(dtype), target_sample_rate)


def waveform_to_world(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    frame_period: float = DEFAULT_FRAME_PERIOD,
    f0_extractor: str = 'harvest',
    f0_floor: float = DEFAULT_F0_FLOOR,
    f0_ceil: float = DEFAULT_F0_CEIL,
    d4c_threshold: float = DEFAULT_D4C_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a waveform (numpy array) to WORLD features.

    Args:
        waveform     (np.ndarray): The waveform as a numpy array.
        sample_rate  (int)       : The sample rate of the audio.
        frame_period (float)     : The frame period in milliseconds.

    Returns:
        world (tuple): The WORLD features (f0, sp, ap).

    NOTE: CREPE はとても重いらしいので注意。GPUリソースも必要。
    """
    # F0
    if f0_extractor == 'harvest':
        f0, timeaxis = pyworld.harvest(
            waveform, sample_rate, frame_period=frame_period, f0_floor=f0_floor, f0_ceil=f0_ceil
        )
    elif f0_extractor == 'dio':
        f0, timeaxis = pyworld.dio(
            waveform, sample_rate, frame_period=frame_period, f0_floor=f0_floor, f0_ceil=f0_ceil
        )
        f0 = pyworld.stonemask(waveform, f0, timeaxis, sample_rate)
    elif f0_extractor == 'crepe':
        import crepe

        timeaxis, f0, _confidence, _activation = crepe.predict(
            waveform,
            sample_rate,
            model_capacity='full',
            viterbi=False,
            step_size=frame_period,
            verbose=1,
        )
    # f0_extractor の指定が harvest, dio, crepe 以外の場合はエラー
    else:
        error_msg = f'Unknown f0 extractor ({f0_extractor}) is specified. Select from ["harvest", "dio", "crepe"].'
        raise ValueError(error_msg)

    # spectrogram, aperiodicity
    spectrogram = pyworld.cheaptrick(waveform, f0, timeaxis, sample_rate)
    aperiodicity = pyworld.d4c(waveform, f0, timeaxis, sample_rate, threshold=d4c_threshold)

    return f0, spectrogram, aperiodicity


def world_to_waveform(
    f0: np.ndarray,
    spectrogram: np.ndarray,
    aperiodicity: np.ndarray,
    sample_rate: int,
    *,
    frame_period: float = DEFAULT_FRAME_PERIOD,
) -> np.ndarray:
    """Convert WORLD features (f0, spectrogram, aperiodicity) back to a waveform.

    Args:
        f0           (np.ndarray)
        spectrogram  (np.ndarray)
        aperiodicity (np.ndarray)
        sample_rate  (int)

    Returns:
        waveform (np.ndarray): The reconstructed waveform.
    """
    waveform = pyworld.synthesize(f0, spectrogram, aperiodicity, sample_rate, frame_period)

    return waveform


def world_to_npzfile(
    f0: np.ndarray, spectrogram: np.ndarray, aperiodicity: np.ndarray, npz_path: Path
) -> None:
    """Save WORLD features to a NPZ file.

    Args:
        f0           (np.ndarray)
        spectrogram  (np.ndarray)
        aperiodicity (np.ndarray)
        npz_path     (Path): Output NPZ file path.
    """
    # 拡張子をチェック
    assert npz_path.suffix == '.npz', 'Output path must be a .npz file.'
    # 書き出し
    np.savez(npz_path, f0=f0, spectrogram=spectrogram, aperiodicity=aperiodicity)


def npzfile_to_world(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load WORLD features from a NPZ file.

    Args:
        npz_path (Path): Input NPZ file path.

    Returns:
        tuple: The loaded WORLD features (f0, spectrogram, aperiodicity).
    """
    # 拡張子をチェック
    assert npz_path.suffix == '.npz', 'Input path must be a .npz file.'
    # 読み取り
    npz = np.load(npz_path)
    return npz['f0'], npz['spectrogram'], npz['aperiodicity']


def world_to_nnsvs(
    f0: np.ndarray,
    spectrogram: np.ndarray,
    aperiodicity: np.ndarray,
    sample_rate: int,
    *,
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


def nnsvs_to_world(
    mgc: np.ndarray,
    lf0: np.ndarray,
    vuv: np.ndarray,
    bap: np.ndarray,
    sample_rate: int,
    fft_size: int = DEFAULT_FFT_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert NNSVS features to WORLD features.

    Args:
        mgc (np.ndarray): Mel-generalized cepstral coefficients
        lf0 (np.ndarray): Log F0
        vuv (np.ndarray): Voiced / unvoiced flag
        bap (np.ndarray): Band aperiodicity

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: WORLD features (f0, spectrogram, aperiodicity)
    """
    # mgc -> spectrogram
    spectrogram = pyworld.decode_spectral_envelope(mgc, sample_rate, fft_size)
    # lf0 -> f0
    f0 = np.exp(lf0, where=(lf0 > 0))
    # bap -> aperiodicity
    aperiodicity = pyworld.decode_aperiodicity(bap, sample_rate, fft_size)
    return f0, spectrogram, aperiodicity


def nnsvs_to_npzfile(
    mgc: np.ndarray, lf0: np.ndarray, vuv: np.ndarray, bap: np.ndarray, npz_path: Path
) -> None:
    """Save NNSVS features to a NPZ file.

    Args:
        mgc (np.ndarray): Mel-generalized cepstral coefficients
        lf0 (np.ndarray): Log F0
        vuv (np.ndarray): Voiced / unvoiced flag
        bap (np.ndarray): Band aperiodicity
        npz_path (Path): Output NPZ file path.
    """
    # 拡張子をチェック
    assert npz_path.suffix == '.npz', 'Output path must be a .npz file.'
    # 書き出し
    np.savez(npz_path, mgc=mgc, lf0=lf0, vuv=vuv, bap=bap)


def npzfile_to_nnsvs(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load NNSVS features from a NPZ file.

    Args:
        npz_path (Path): Input NPZ file path.

    Returns:
        tuple: The loaded NNSVS features (mgc, lf0, vuv, bap).
    """
    # 拡張子をチェック
    assert npz_path.suffix == '.npz', 'Input path must be a .npz file.'
    # 読み取り
    npz = np.load(npz_path)
    return npz['mgc'], npz['lf0'], npz['vuv'], npz['bap']

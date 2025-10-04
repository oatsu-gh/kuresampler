#!/usr/bin/env python3
# Copyright (c) 2025 oatsu
# NOTE: pyworld が型チェックエラーを出すので無視する
# pyright: reportAttributeAccessIssue=none

"""音声まわりのファイルや特徴量を相互変換する。

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
import torch
from nnsvs.gen import predict_waveform
from nnsvs.util import StandardScaler
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

# WAV settings ---------------------
DEFAULT_WAV_DTYPE: str = 'float64'
DEFAULT_RESAMPLE_TYPE: str = 'soxr_hq'  # [soxr_vhq, soxr_hq, kaiser_best] あたりから選ぶとよい。https://librosa.org/doc/0.11.0/generated/librosa.resample.html#librosa-resample
# WORLD settings -------------------
DEFAULT_FRAME_PERIOD: int = 5  # ms
DEFAULT_F0_FLOOR: float = 50.0
DEFAULT_F0_CEIL: float = 2000.0
DEFAULT_D4C_THRESHOLD: float = 0.50  # default: 0.5 (NNSVS default is 0.5, PyRwu default is 0.85.)
# ----------------------------------


def wavfile_to_waveform(
    wav_path: Path | str,
    out_sample_rate: int | None = None,
    *,
    resample_type: str = DEFAULT_RESAMPLE_TYPE,
    dtype: str = DEFAULT_WAV_DTYPE,
) -> tuple[np.ndarray, int, int]:
    """Convert a WAV file to a waveform (numpy array).

    Args:
        wav_path           (Path)    : Path to the WAV file.
        out_sample_rate    (int)     : Sample rate for the output waveform.
        dtype              (np.dtype): dtype for the output waveform.
        resample_type      (str)     : Resampling method. Select from `res_type` options of `librosa.resample`. (recommended: soxr_vhq, soxr_hq, kaiser_best)

    Returns:
        waveform    (np.ndarray): Waveform as a numpy array.
        in_sample_rate (int): Sample rate of the original audio.
        out_sample_rate   (int): Target sample rate of the returning waveform.

    """
    waveform: np.ndarray
    in_sample_rate: int

    wav_path = Path(wav_path)
    waveform, in_sample_rate = sf.read(wav_path, dtype=dtype)
    # out_sample_rate が None の場合は in_sample_rate と同じにする
    if out_sample_rate is None:
        out_sample_rate = in_sample_rate

    # リサンプリング
    if in_sample_rate != out_sample_rate:
        waveform = librosa.resample(
            waveform,
            orig_sr=in_sample_rate,
            target_sr=out_sample_rate,
            res_type=resample_type,
        )
    return waveform, in_sample_rate, out_sample_rate


def waveform_to_wavfile(
    waveform: np.ndarray,
    wav_path: Path | str,
    in_sample_rate: int,
    out_sample_rate: int,
    *,
    resample_type: str = DEFAULT_RESAMPLE_TYPE,
    dtype: str = DEFAULT_WAV_DTYPE,
) -> None:
    """Convert a waveform (numpy array) to a WAV file.

    Args:
        waveform             (np.ndarray): The waveform as a numpy array.
        wav_path             (Path)      : The path to the output WAV file.
        in_sample_rate       (int)       : The original sample rate of the audio.
        out_sample_rate      (int)       : The target sample rate for the output WAV file.
        resample_type       (str)       : Resampling method. Select from `res_type` options of `librosa.resample`. (recommended: soxr_vhq, soxr_hq, kaiser_best)
        dtype               (np.dtype)  : The dtype for the output WAV file.

    """
    if in_sample_rate == out_sample_rate:
        sf.write(wav_path, waveform.astype(dtype), out_sample_rate)
    else:
        waveform_resampled = librosa.resample(
            waveform,
            orig_sr=in_sample_rate,
            target_sr=out_sample_rate,
            res_type=resample_type,
        )
        sf.write(wav_path, waveform_resampled.astype(dtype), out_sample_rate)


def waveform_to_world(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    frame_period: int = DEFAULT_FRAME_PERIOD,
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
        f0_extractor (str)       : The F0 extraction method. Select from ["harvest", "dio", "crepe"].
        f0_floor     (float)     : The minimum F0 value.
        f0_ceil      (float)     : The maximum F0 value.
        d4c_threshold(float)     : The threshold for the D4C algorithm.

    Returns:
        f0           (np.ndarray): Fundamental frequency.
        spectral_envelope  (np.ndarray): spectral_envelope.
        aperiodicity (np.ndarray): Aperiodicity.

    NOTE: CREPE はとても重いらしいので注意。GPUリソースも必要。
    TODO: harvestのf0推定がとても重い。frq ファイルがあれば読むようにする。なければ logger で警告を出す。独自に krq ファイルを出力する?

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
        msg = 'CREPE f0 extractor is not implemented yet.'
        raise NotImplementedError(msg)
    # f0_extractor の指定が harvest, dio, crepe 以外の場合はエラー
    else:
        error_msg = f'Unknown f0 extractor ({f0_extractor}) is specified. Select from ["harvest", "dio", "crepe"].'
        raise ValueError(error_msg)

    # spectral_envelope, aperiodicity
    spectral_envelope = pyworld.cheaptrick(waveform, f0, timeaxis, sample_rate)
    aperiodicity = pyworld.d4c(waveform, f0, timeaxis, sample_rate, threshold=d4c_threshold)

    return f0, spectral_envelope, aperiodicity


def world_to_waveform(
    f0: np.ndarray,
    spectral_envelope: np.ndarray,
    aperiodicity: np.ndarray,
    sample_rate: int,
    *,
    frame_period: float = DEFAULT_FRAME_PERIOD,
) -> np.ndarray:
    """Convert WORLD features (f0, spectral_envelope, aperiodicity) back to a waveform.

    Args:
        f0                (np.ndarray): F0 [Hz]
        spectral_envelope (np.ndarray): Spectral Envelope
        aperiodicity      (np.ndarray): Aperiodicity
        frame_period      (float)     : Frame period [ms]
        sample_rate       (int)       : Sample rate [Hz]

    Returns:
        waveform (np.ndarray): The reconstructed waveform.

    """
    # 特徴量の nan と inf を除去
    f0 = np.nan_to_num(f0, nan=0)
    spectral_envelope = np.nan_to_num(spectral_envelope, nan=0, posinf=1, neginf=0)
    aperiodicity = np.nan_to_num(aperiodicity, nan=1)
    # 特徴量の範囲を制限
    f0 = np.clip(f0, 0, None)
    spectral_envelope = np.clip(spectral_envelope, np.finfo(spectral_envelope.dtype).tiny, 1)
    aperiodicity = np.clip(aperiodicity, np.finfo(aperiodicity.dtype).tiny, 1)
    # waveform を合成
    waveform = pyworld.synthesize(f0, spectral_envelope, aperiodicity, sample_rate, frame_period)
    return waveform


def world_to_npzfile(
    f0: np.ndarray,
    spectral_envelope: np.ndarray,
    aperiodicity: np.ndarray,
    npz_path: Path | str,
    *,
    compress: bool = False,
) -> None:
    """Save WORLD features to a NPZ file.

    Args:
        f0 (np.ndarray)                : F0 [Hz]
        spectral_envelope (np.ndarray) : Spectral envelope
        aperiodicity (np.ndarray)      : Aperiodicity
        npz_path (Path)                : Output NPZ file path.
        compress (bool)                : Whether to use compression when saving the NPZ file.

    """
    npz_path = Path(npz_path)
    # 拡張子をチェック
    if npz_path.suffix != '.npz':
        msg = 'Output path must be .npz file.'
        raise ValueError(msg)
    # 書き出し
    if not compress:
        np.savez(
            npz_path,
            f0=f0,
            spectral_envelope=spectral_envelope,
            aperiodicity=aperiodicity,
        )
    else:
        np.savez_compressed(
            npz_path,
            f0=f0,
            spectral_envelope=spectral_envelope,
            aperiodicity=aperiodicity,
        )


def npzfile_to_world(npz_path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load WORLD features from a NPZ file.

    Args:
        npz_path (Path): Input NPZ file path.

    Returns:
        tuple: The loaded WORLD features (f0, spectral_envelope, aperiodicity).

    """
    npz_path = Path(npz_path)
    # 拡張子をチェック
    if npz_path.suffix != '.npz':
        msg = 'Input path must be .npz file.'
        raise ValueError(msg)
    # 読み取り
    npz = np.load(npz_path)
    return npz['f0'], npz['spectral_envelope'], npz['aperiodicity']


def world_to_nnsvs(
    f0: np.ndarray,
    sp: np.ndarray,
    ap: np.ndarray,
    sample_rate: int,
    *,
    number_of_mgc_dimensions: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """World 特徴量を nnsvs 対応の形式に変換する。

    Args:
        f0 (np.ndarray)               : F0 [Hz]
        sp (np.ndarray)               : Spectral envelope
        ap (np.ndarray)               : Aperiodicity
        sample_rate (int)             : Output sample rate of the audio [Hz]
        number_of_mgc_dimensions (int): Number of mel-generalized cepstral coefficients

    Returns:
        mgc (np.ndarray): mel-generalized cepstral coefficients
        lf0 (np.ndarray): log F0
        vuv (np.ndarray): voiced / unvoiced flag
        bap (np.ndarray): band aperiodicity

    """
    # 特徴量の nan と inf を除去
    f0 = np.nan_to_num(f0, nan=0)
    sp = np.nan_to_num(sp, nan=0, posinf=1, neginf=0)
    ap = np.nan_to_num(ap, nan=1)
    # 特徴量の範囲を制限
    f0 = np.clip(f0, 0, None)
    sp = np.clip(sp, np.finfo(sp.dtype).tiny, 1)
    ap = np.clip(ap, np.finfo(ap.dtype).tiny, 1)

    # sp -> mgc, f0
    mgc = pyworld.code_spectral_envelope(sp, sample_rate, number_of_mgc_dimensions)
    # f0 -> lf0
    lf0 = np.zeros_like(f0)
    lf0[np.nonzero(f0)] = np.log(f0[np.nonzero(f0)])
    # vuv を計算
    vuv = (f0 > 0).astype(np.float32)
    # aperiodicity -> bap
    bap = pyworld.code_aperiodicity(ap, sample_rate)
    # nnsvs 向けの world 特徴量を返す
    return mgc, lf0.reshape(-1, 1), vuv.reshape(-1, 1), bap


def nnsvs_to_world(
    mgc: np.ndarray,
    lf0: np.ndarray,
    vuv: np.ndarray,  # noqa: ARG001
    bap: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert NNSVS features to WORLD features.

    Args:
        mgc (np.ndarray) : Mel-generalized cepstral coefficients
        lf0 (np.ndarray) : Log F0
        vuv (np.ndarray) : Voiced / unvoiced flag
        bap (np.ndarray) : Band aperiodicity
        sample_rate (int): Original sample rate of the audio, before feature extraction.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: WORLD features (f0, spectral_envelope, aperiodicity)

    """
    # Automatically determine fft_size from sample_rate
    fft_size = pyworld.get_cheaptrick_fft_size(sample_rate)
    # mgc -> spectral_envelope
    spectral_envelope = pyworld.decode_spectral_envelope(mgc, sample_rate, fft_size)
    # lf0 -> f0
    f0 = np.exp(lf0, where=(lf0 > 0))  # NOTE: lf0 のみで計算しているがvuvを使うこともできる。
    # bap -> aperiodicity
    aperiodicity = pyworld.decode_aperiodicity(bap, sample_rate, fft_size)
    return f0, spectral_envelope, aperiodicity


def nnsvs_to_npzfile(
    mgc: np.ndarray, lf0: np.ndarray, vuv: np.ndarray, bap: np.ndarray, npz_path: Path | str
) -> None:
    """Save NNSVS features to a NPZ file.

    Args:
        mgc (np.ndarray): Mel-generalized cepstral coefficients
        lf0 (np.ndarray): Log F0
        vuv (np.ndarray): Voiced / unvoiced flag
        bap (np.ndarray): Band aperiodicity
        npz_path (Path): Output NPZ file path.

    """
    npz_path = Path(npz_path)
    # 拡張子をチェック
    if npz_path.suffix != '.npz':
        msg = 'Output path must be .npz file.'
        raise ValueError(msg)
    # 書き出し
    np.savez(npz_path, mgc=mgc, lf0=lf0, vuv=vuv, bap=bap)


def npzfile_to_nnsvs(
    npz_path: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load NNSVS features from a NPZ file.

    Args:
        npz_path (Path): Input NPZ file path.

    Returns:
        tuple: The loaded NNSVS features (mgc, lf0, vuv, bap).

    """
    npz_path = Path(npz_path)
    # 拡張子をチェック
    if npz_path.suffix != '.npz':
        msg = 'Input path must be .npz file.'
        raise ValueError(msg)
    # 読み取り
    npz = np.load(npz_path)
    return npz['mgc'], npz['lf0'], npz['vuv'], npz['bap']

def world_to_nnsvs_to_waveform(
    device: torch.device,
    f0: np.ndarray,
    sp: np.ndarray,
    ap: np.ndarray,
    *,
    input_sample_rate: int,
    output_sample_rate: int,
    vocoder_model: torch.nn.Module,
    vocoder_config: DictConfig | ListConfig,
    vocoder_in_scaler: StandardScaler,
    frame_period: int = 5,
    use_world_codec: bool = True,
    feature_type: str = 'world',
    vocoder_type: str = 'usfgan',
    vuv_threshold: float = 0.5,
    mgc_dimensions: int = 60,
    resample_type: str = 'soxr_vhq',
):
    """通常のWORLD特徴量 (f0, sp, ap) から NNSVS特徴量 (mgc, lf0, vuv, bap) を経由して waveform に変換する。

    Args:
        device (torch.device)          : The device to run the model on.
        f0 (np.ndarray)                : F0 [Hz]
        sp (np.ndarray)                : Spectral envelope
        ap (np.ndarray)                : Aperiodicity
        input_sample_rate (int)        : Original sample rate of the audio, before feature extraction.
        output_sample_rate (int)       : Target sample rate for the output waveform.
        vocoder_model (torch.nn.Module): The vocoder model.
        vocoder_config (DictConfig)    : The configuration for the vocoder model.
        vocoder_in_scaler (StandardScaler): The input scaler for the vocoder model.
        frame_period (int)             : Frame period [ms]
        use_world_codec (bool)         : Whether to use WORLD codec for waveform generation.
        feature_type (str)             : Feature type for the vocoder. Select from ["world", "mel"].
        vocoder_type (str)             : Type of the vocoder. Select from ["world", "pwgan", "usfgan"].
        vuv_threshold (float)          : Threshold for voiced/unvoiced decision.
        mgc_dimensions (int)           : Number of mel-generalized cepstral coefficients.
        resample_type (str)            : Resampling method. Select from `res_type` options of `librosa.resample`. (recommended: soxr_vhq, soxr_hq, kaiser_best)

    Returns:
        waveform (np.ndarray): The generated waveform.

    """
    # vocoder のサンプリング周波数を取得
    vocoder_sample_rate = vocoder_config.data.sample_rate

    # WORLD -> NNSVS 変換
    mgc, lf0, vuv, bap = world_to_nnsvs(
        f0,
        sp,
        ap,
        input_sample_rate,  # 原音WAVの特徴量抽出に使ったサンプリング周波数を渡す
        number_of_mgc_dimensions=mgc_dimensions,
    )
    multistream_features = (mgc, lf0, vuv, bap)

    # NNSVS -> waveform 変換
    waveform = predict_waveform(
        device,
        multistream_features,
        vocoder=vocoder_model,
        vocoder_config=vocoder_config,
        vocoder_in_scaler=vocoder_in_scaler,
        sample_rate=vocoder_sample_rate,
        frame_period=frame_period,
        use_world_codec=use_world_codec,
        feature_type=feature_type,
        vocoder_type=vocoder_type,
        vuv_threshold=vuv_threshold,
    )  # この時点のサンプリング周波数は vocoder_sample_rate

    # リサンプリング
    if vocoder_sample_rate != output_sample_rate:
        waveform = librosa.resample(
            waveform,
            orig_sr=vocoder_sample_rate,
            target_sr=output_sample_rate,
            res_type=resample_type,
        )  # ここで output_sample_rate のサンプリング周波数に変換される

    # waveform を返す (サンプリング周波数は output_sample_rate)
    return waveform

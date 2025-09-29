# Copyright (c) 2025 oatsu
"""
Utility functions for kuresampler.
"""

import logging
from pathlib import Path
from warnings import warn

import colored_traceback.auto  # noqa: F401
import numpy as np
import torch
from colorlog import ColoredFormatter
from nnsvs.usfgan import USFGANWrapper
from nnsvs.util import StandardScaler
from nnsvs.util import load_vocoder as nnsvs_load_vocoder
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


def get_device() -> torch.device:
    """PyTorch デバイスを取得する。"""
    device = torch.accelerator.current_accelerator(check_available=True) or torch.device('cpu')
    return device


def setup_logger(level=logging.INFO) -> logging.Logger:
    """Loggerを作成する。"""
    formatter = ColoredFormatter(
        '[%(filename)s][%(log_color)s%(levelname)s%(reset)s] %(message)s',
        log_colors={
            'DEBUG': 'green',
            'INFO': 'blue',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    _logger = logging.getLogger(__name__)
    _logger.setLevel(level)
    _logger.addHandler(handler)
    return _logger


def easy_interpolate(y: list[float] | np.ndarray) -> float:
    """スパイクノイズ除去のため、線形補間またはキュービック補間で x=0 の値を求める。
    y の長さが 2 のときは x1, x2 = (-1, +1) として x=0 の値を求める。
    y の長さが 4 のときは x1, x2, x3, x4 = (-2, -1, +1, +2) として x=0 の値を求める。

    Args:
        y1 (float): 1つ目のデータ
        y2 (float): 2つ目のデータ
        y3 (float): 3つ目のデータ
        y4 (float): 4つ目のデータ

    Returns:
        float: x=0 のときの y の値
    """
    # 補間に使う重みを決定する。y の長さが2の時は線形補間、y の長さが4の時はキュービック補間。
    weights = [1, 1] if len(y) == 2 else [-1, 4, 4, -1] if len(y) == 4 else None
    if weights is None:
        msg = 'y must be a list of length 2 or 4.'
        raise ValueError(msg)
    return sum([a * b for a, b in zip(y, weights, strict=True)]) / sum(weights)


def denoise_spike(f0: np.ndarray, iqr_multiplier: float = 1.5) -> np.ndarray:
    """1次元配列の f0 のスパイクノイズを除去する。

    Args:
        f0 (np.ndarray): スパイクノイズを除去する対象の f0 配列

    Returns:
        np.ndarray: スパイクノイズを除去した f0 配列
    """
    f0_clean = f0.copy()
    window = 5
    half_window = window // 2

    for i in range(half_window, len(f0_clean) - half_window):
        window_data = f0_clean[i - half_window : i + half_window + 1]
        q1 = np.quantile(window_data, 0.25)
        a3 = np.quantile(window_data, 0.75)
        iqr = max(a3 - q1, 0.01)  # IQRが小さすぎると誤検知するので下限を設定
        # 外れ値の閾値を計算
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = a3 + iqr_multiplier * iqr
        # スパイクノイズを除去
        if f0_clean[i] < lower_bound or f0_clean[i] > upper_bound:
            warn(f'Spike noise detected at index {i}: {f0_clean[i]}', stacklevel=2)
            y = [
                f0_clean[i - 2],
                f0_clean[i - 1],
                f0_clean[i + 1],
                f0_clean[i + 2],
            ]
            f0_clean[i] = easy_interpolate(y)
    return f0_clean


def overlap_world_feature(
    feature_a: np.ndarray,
    feature_b: np.ndarray,
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
    result = crossfade_world_feature(
        feature_a,
        feature_b,
        overlap_samples,
        shape=None,
    )
    return result


def crossfade_world_feature(
    feature_a: np.ndarray,
    feature_b: np.ndarray,
    overlap_samples: int,
    shape: str | None = 'linear',
    *,
    calc_in_log: bool = False,
) -> np.ndarray:
    """WORLD特徴量をクロスフェードさせる。f0でつかう想定。

    Args:
        features_a (np.ndarray): The first set of WORLD features.
        features_b (np.ndarray): The second set of WORLD features.
        fade_samples (int): The number of samples to fade.

    Returns:
        np.ndarray: The crossfaded WORLD features.
    """
    # オーバーラップ区間が0の場合は単純結合して返す
    if overlap_samples == 0:
        return np.concatenate([feature_a, feature_b], axis=0)
    # オーバーラップ区間が長すぎる場合はエラーを返す
    if overlap_samples > min(feature_a.shape[0], feature_b.shape[0]):
        msg = f'Invalid overlap_samples: {overlap_samples}. Overlap must be less than the length of both existing feature ({feature_a.shape[0]}) and new feature ({feature_b.shape[0]}).'
        raise ValueError(msg)

    ## クロスフェード部分の計算
    if calc_in_log:
        # f0を対数に変換してからクロスフェード
        feature_a = np.log(np.clip(feature_a, a_min=1, a_max=None))
        feature_b = np.log(np.clip(feature_b, a_min=1, a_max=None))

    # クロスフェードなしで合算 (エンベロープ反映後の sp, ap で使う想定)
    if shape is None:
        fade_out = np.ones((overlap_samples, 1))  # 不要だが後続バグ予防のためダミー
        fade_in = np.ones((overlap_samples, 1))  # 不要だが後続バグ予防のためダミー
        overlap_area = feature_a[-overlap_samples:] + feature_b[:overlap_samples]
    # 線形クロスフェード (f0で使う想定)
    elif shape == 'linear':
        fade_out = np.linspace(1, 0, overlap_samples)[:, np.newaxis]
        fade_in = np.linspace(0, 1, overlap_samples)[:, np.newaxis]
        overlap_area = (
            feature_a[-overlap_samples:] * fade_out + feature_b[:overlap_samples] * fade_in
        )
    # cosineクロスフェード
    elif shape in ('cosine', 'cos'):
        t = np.linspace(0, np.pi / 2, overlap_samples)[:, np.newaxis]
        fade_out = np.cos(t)
        fade_in = np.sin(t)
        overlap_area = (
            feature_a[-overlap_samples:] * fade_out + feature_b[:overlap_samples] * fade_in
        )
    else:
        msg = f'Invalid shape: {shape}. Choose from None, "linear", "cosine", or "cos".'
        raise ValueError(msg)

    # 前・クロスフェード部分・後ろを結合して返す
    result = np.concatenate(
        [feature_a[:-overlap_samples], overlap_area, feature_b[overlap_samples:]]
    )
    # 対数から元に戻す
    if calc_in_log:
        result = np.exp(result)
        result = np.where(result <= 1, 0, result)
    return result


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

    vocoder_model, vocoder_in_scaler, vocoder_config = nnsvs_load_vocoder(
        vocoder_model_path,
        device,
        acoustic_config,
    )
    return vocoder_model, vocoder_in_scaler, vocoder_config

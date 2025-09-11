# Copyright (c) 2025 oatsu
"""
Utility functions for audio processing
"""

import logging
import sys
from pathlib import Path

import colored_traceback.auto  # noqa: F401
import numpy as np
import torch
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


def setup_logger() -> logging.Logger:
    """Loggerを作成する。"""
    # my_package.my_moduleのみに絞ってsys.stderrにlogを出す
    logging.basicConfig(
        stream=sys.stdout,
        format='[%(filename)s][%(levelname)s] %(message)s',
        level=logging.INFO,
    )
    return logging.getLogger(__name__)


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
    print('overlap_area.shape:', overlap_area.shape)

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

# Copyright (c) 2025 oatsu
"""Utility functions for kuresampler.

Functions:
    get_device: PyTorch デバイスを取得する。
    setup_logger: Loggerを作成する。
    easy_interpolate: スパイクノイズ除去のため、線形補間またはキュービック補間で x=0 の値を求める。
    denoise_spike: 1次元配列の f0 のスパイクノイズを除去する。
    overlap_f0: f0をオーバーラップさせる。オーバーラップ区間に0Hzが含まれる場合は0Hzではない方の値を使用する。
    overlap_sp: WORLD特徴量の sp (Spectral envelope) をオーバーラップさせる。
    fill_nan_pair: 2つの配列のうち、片方がNaNで片方が数値のとき、NaNを数値で埋める。
    _crossfade_world_feature: WORLD特徴量をクロスフェードさせる。f0でつかう想定。
    load_vocoder_model: NNSVSのボコーダモデルを読み込む。

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
        '[%(filename)s:%(lineno)d][%(log_color)s%(levelname)s%(reset)s] %(message)s',
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
        y (list[float] | np.ndarray): 補間に使う y の値。長さは 2 または 4。

    Returns:
        float: x=0 のときの y の値

    """
    # 補間に使う重みを決定する。y の長さが2の時は線形補間、y の長さが4の時はキュービック補間。
    weights = [1, 1] if len(y) == 2 else [-1, 4, 4, -1] if len(y) == 4 else None
    if weights is None:
        msg = 'y must be a list of length 2 or 4.'
        raise ValueError(msg)
    return sum([a * b for a, b in zip(y, weights, strict=True)]) / sum(weights)


def denoise_spike(
    f0: np.ndarray, iqr_multiplier: float = 1.5, *, logger: None | logging.Logger
) -> np.ndarray:
    """1次元配列の f0 のスパイクノイズを除去する。

    Args:
        f0 (np.ndarray)               : スパイクノイズを除去する対象の f0 配列
        iqr_multiplier (float)        : IQR (Interquartile Range) の何倍を外れ値とみなすか
        logger (logging.Logger | None): Logger

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
            if logger is not None:
                logger.warning(f'Spike noise detected at index {i}: {f0_clean[i]}')
            else:
                warn(f'Spike noise detected at index {i}: {f0_clean[i]}', stacklevel=2)
            y = [
                f0_clean[i - 2],
                f0_clean[i - 1],
                f0_clean[i + 1],
                f0_clean[i + 2],
            ]
            f0_clean[i] = easy_interpolate(y)
    return f0_clean


def fill_nan_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2つの配列のうち、片方がNaNで片方が数値のとき、NaNを数値で埋める。

    Args:
        a (np.ndarray): The first array.
        b (np.ndarray): The second array.

    Returns:
        tuple[np.ndarray, np.ndarray]: The filled arrays.

    Examples:
        >>> fill_nan_pair(
        ...     np.array([1.0, np.nan, 3.0, np.nan]),
        ...     np.array([np.nan, 2.0, 5.0, np.nan])
        ... )
        (array([ 1.,  2.,  3., nan]), array([ 1.,  2.,  5., nan]))

    """
    if a.shape != b.shape:
        msg = f'Shape mismatch: a.shape={a.shape}, b.shape={b.shape}'
        raise ValueError(msg)
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    a[nan_a & ~nan_b] = b[nan_a & ~nan_b]
    b[nan_b & ~nan_a] = a[nan_b & ~nan_a]
    return a, b


def _crossfade_world_feature(
    feature_a: np.ndarray,
    feature_b: np.ndarray,
    n_overlap: int,
    crossfade_shape: str | None,
) -> np.ndarray:
    """WORLD特徴量をクロスフェードさせる。f0でつかう想定。

    Args:
        feature_a (np.ndarray): The first set of WORLD features.
        feature_b (np.ndarray): The second set of WORLD features.
        n_overlap (int): The number of samples to fade.
        crossfade_shape (str | None): The shape of the crossfade. Choose from None, 'linear', 'cosine', or 'cos'.

    Returns:
        np.ndarray: The crossfaded WORLD features.

    Examples:
        1. オーバーラップ区間が0の場合は単純結合して返す
        >>> _crossfade_world_feature(
        ...     np.array([[-1, -2], [-1, -2], [-1, -2], [-1, -2], [-1, -2], [-1, -2], [-1, -2]]),
        ...     np.array([[+1, +2], [+1, +2], [+1, +2], [+1, +2], [+1, +2], [+1, +2]]),
        ...     n_overlap=5,
        ...     crossfade_shape=None
        ... )
        array([[-1., -2.],
               [-1., -2.],
               [ 0.,  0.],
               [ 0.,  0.],
               [ 0.,  0.],
               [ 0.,  0.],
               [ 0.,  0.],
               [ 1.,  2.]])

        2. 線形クロスフェード
        >>> _crossfade_world_feature(
        ...     np.array([[-1, -2], [-1, -2], [-1, -2], [-1, -2], [-1, -2], [-1, -2], [-1, -2]]),
        ...     np.array([[+1, +2], [+1, +2], [+1, +2], [+1, +2], [+1, +2], [+1, +2]]),
        ...     n_overlap=5,
        ...     crossfade_shape='linear'
        ... )
        array([[-1. , -2. ],
               [-1. , -2. ],
               [-1. , -2. ],
               [-0.5, -1. ],
               [ 0. ,  0. ],
               [ 0.5,  1. ],
               [ 1. ,  2. ],
               [ 1. ,  2. ]])

        3. cosineクロスフェード
        # TODO: 後日テスト

        4. オーバーラップ区間が長すぎる場合はエラーを返す
        >>> _crossfade_world_feature(
        ...     np.array([[-1, -2], [-1, -2], [-1, -2], [-1, -2], [-1, -2], [-1, -2], [-1, -2]]),
        ...     np.array([[+1, +2], [+1, +2], [+1, +2], [+1, +2], [+1, +2], [+1, +2]]),
        ...     n_overlap=10,
        ...     crossfade_shape='linear'
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Invalid n_overlap: 10. Overlap must be shorter than both of feature_a (7) and feature_b (6).

    """
    # オーバーラップ区間が0の場合は単純結合して返す
    if n_overlap == 0:
        return np.concatenate([feature_a, feature_b], axis=0)
    # オーバーラップ区間が長すぎる場合はエラーを返す
    if n_overlap > min(feature_a.shape[0], feature_b.shape[0]):
        msg = (
            f'Invalid n_overlap: {n_overlap}. '
            f'Overlap must be shorter than both of feature_a ({feature_a.shape[0]}) and feature_b ({feature_b.shape[0]}).'
        )
        raise ValueError(msg)

    # オーバーラップ区間の NaN を補完
    overlap_a, overlap_b = fill_nan_pair(feature_a[-n_overlap:], feature_b[:n_overlap])
    # 単純オーバーラップ (エンベロープ反映後の sp で使用)
    if crossfade_shape is None:
        fade_out = np.ones((n_overlap, 1))
        fade_in = np.ones((n_overlap, 1))
    # 線形クロスフェード (f0, sp で使用)
    elif crossfade_shape == 'linear':
        fade_out = np.linspace(1, 0, n_overlap)[:, np.newaxis]
        fade_in = np.linspace(0, 1, n_overlap)[:, np.newaxis]
    # cosineクロスフェード (f0, sp で使用)
    elif crossfade_shape in ('cosine', 'cos'):
        t = np.linspace(0, np.pi / 2, n_overlap)[:, np.newaxis]
        fade_out = np.cos(t)
        fade_in = np.sin(t)
    else:
        msg = f'Invalid shape: {crossfade_shape}. Choose from None, "linear", "cosine", or "cos".'
        raise ValueError(msg)
    # クロスフェード区間計算
    overlap_area = overlap_a * fade_out + overlap_b * fade_in
    # 前・クロスフェード部分・後ろを結合して返す
    result = np.concatenate([feature_a[:-n_overlap], overlap_area, feature_b[n_overlap:]], axis=0)
    return result


def overlap_f0(
    f0_a: np.ndarray,
    f0_b: np.ndarray,
    n_overlap: int,
    crossfade_shape: str = 'linear',
) -> np.ndarray:
    """f0をオーバーラップさせる。オーバーラップ区間に0Hzが含まれる場合は0Hzではない方の値を使用する。

    Args:
        f0_a (np.ndarray): The first set of f0.
        f0_b (np.ndarray): The second set of f0.
        n_overlap (int): The number of samples to fade.
        crossfade_shape (str): The shape of the crossfade. Choose from 'linear', 'cosine', or 'cos'.

    Returns:
        np.ndarray: The crossfaded f0.

    """
    # TODO: オーバーラップ領域以外も対数変換をしていて無駄なので、オーバーラップ領域だけ変換できるようにする。
    # reshape
    f0_a = f0_a.reshape(-1, 1)
    f0_b = f0_b.reshape(-1, 1)
    # f0 = 0 を nan に置換
    f0_a = np.where(f0_a == 0, np.nan, f0_a)
    f0_b = np.where(f0_b == 0, np.nan, f0_b)
    # 対数変換
    log_f0_a = np.log(f0_a)  # nan 補完してあるのでもとの f0 に 0 は含まれない
    log_f0_b = np.log(f0_b)  # nan 補完してあるのでもとの f0 に 0 は含まれない
    # クロスフェード
    result = _crossfade_world_feature(
        log_f0_a,
        log_f0_b,
        n_overlap,
        crossfade_shape=crossfade_shape,
    )
    # 指数変換して元に戻す
    result = np.exp(result)
    # nan を 0 に置換して元に戻す
    result = np.where(np.isnan(result), 0, result)
    # reshape を戻してから返す
    return result.reshape(-1)


def overlap_sp(
    sp_a: np.ndarray,
    sp_b: np.ndarray,
    n_overlap: int,
    crossfade_shape: None | str = None,
) -> np.ndarray:
    """WORLD特徴量の sp (Spectral envelope) をオーバーラップさせる。あらかじめ音量エンベロープが反映されていることを想定。

    音量エンベロープによって線形ではなく2乗でパラメータフェードイン/フェードアウトされており、
    そのままオーバーラップすると音量が下がってしまうため、
    平方根スケールでオーバーラップしてから2乗に戻す必要がある。
    オーバーラップ領域を平方根スケールに変換 → 合算 → 2乗 → 前後と結合

    Args:
        sp_a (np.ndarray): The first set of WORLD features.
        sp_b (np.ndarray): The second set of WORLD features.
        n_overlap (int): The number of samples to fade.
        crossfade_shape (str): The shape of the crossfade. Choose from None, 'linear', 'cosine', or 'cos'.

    Returns:
        np.ndarray: The crossfaded WORLD features.

    """
    # TODO: オーバーラップ領域以外も対数変換をしていて無駄なので、オーバーラップ領域だけ変換できるようにする。
    sp_a = np.sqrt(sp_a)
    sp_b = np.sqrt(sp_b)
    result = _crossfade_world_feature(
        sp_a,
        sp_b,
        n_overlap,
        crossfade_shape=crossfade_shape,
    )
    result = np.square(result)
    return result


def overlap_ap(
    ap_a: np.ndarray,
    ap_b: np.ndarray,
    n_overlap: int,
    crossfade_shape: None | str = 'linear',
) -> np.ndarray:
    """WORLD特徴量の ap (Aperiodicity) をオーバーラップさせる。クロスフェードを行う。

    Args:
        ap_a (np.ndarray): The first set of aperiodicity features.
        ap_b (np.ndarray): The second set of aperiodicity features.
        n_overlap (int): The number of samples to fade.
        crossfade_shape (str): The shape of the crossfade. Choose from 'linear', 'cosine', or 'cos'.

    Returns:
        np.ndarray: The crossfaded aperiodicity features.

    """
    result = _crossfade_world_feature(ap_a, ap_b, n_overlap, crossfade_shape=crossfade_shape)
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

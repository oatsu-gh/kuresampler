#!/usr/bin/env python3
# Copyright (c) 2025 oatsu
"""
kuresampler の配布準備を行う。
"""

import shutil
import subprocess
from pathlib import Path

import colored_traceback.auto  # noqa: F401
from send2trash import send2trash
from tqdm import tqdm

RELEASE_ROOT = Path('_release')
SOURCE_DIR = Path(__file__).parent

PYTHON_DIR = SOURCE_DIR / 'python-3.12.10-embed-amd64'

REQUIRED_DIRS = [
    'resampler',
    'models',
]

EXE_FILES = [
    'kuresampler_K.exe',
    'kuresampler_R.exe',
    'kuresampler_fast_K.exe',
    'kuresampler_fast_R.exe',
]

BAT_FILES = [
    'kuresampler_K_child.bat',
    'kuresampler_R_child.bat',
    'reinstall_torch.bat',
]

CS_FILES = [
    'kuresampler.cs',
]

PY_FILES = [
    'resampler.py',
    'convert.py',
    'util.py',
]

OTHRER_FILES = [
    'LICENSE',
    'README.md',
    'requirements.txt',
]


def install_torch_cpu():
    """
    CPU版のpytorchをインストールする。
    """
    python_exe = str(PYTHON_DIR / 'python.exe')
    uninstall_command = [python_exe, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio', '--no-input']  # fmt: skip
    subprocess.run(uninstall_command, check=True, shell=True)  # noqa: S603
    install_command = [python_exe, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--no-warn-script-location']  # fmt: skip
    subprocess.run(install_command, check=True, shell=True)  # noqa: S603


def remove_cache_files(path_dir: Path):
    """
    キャッシュファイルを削除する。
    """
    # キャッシュフォルダを再帰的に検索
    # __pycache__フォルダを再帰的に検索
    dirs_to_remove = list(path_dir.rglob('__pycache__'))
    dirs_to_remove = [
        path for path in dirs_to_remove if path.is_dir() and path.name == '__pycache__'
    ]
    # キャッシュフォルダを削除
    for cache_dir in tqdm(dirs_to_remove):
        shutil.rmtree(cache_dir)


def prepare_release(version: str):
    """配布用フォルダを作成し、zip圧縮する。"""
    # 配布フォルダを作成する
    RELEASE_ROOT.mkdir(exist_ok=True)
    release_dir = RELEASE_ROOT / f'kuresampler-{version}'
    if release_dir.exists():
        shutil.rmtree(release_dir)
        print(f'Removed old release dir: {release_dir}')
    release_dir.mkdir()
    print(f'Created release dir: {release_dir}')
    print('──────────────────────────────────────────────')

    # 必要なフォルダをコピーする
    for dname in REQUIRED_DIRS:
        print(f'Copying dir: {dname}')
        shutil.copytree(SOURCE_DIR / dname, release_dir / dname)
    print('──────────────────────────────────────────────')

    # 必要なファイルをコピーする
    required_files = EXE_FILES + BAT_FILES + CS_FILES + PY_FILES + OTHRER_FILES
    for fname in required_files:
        print(f'Copying file: {fname}')
        shutil.copy2(SOURCE_DIR / fname, release_dir / fname)
    print('──────────────────────────────────────────────')

    # CPU版のpytorchをインストールする
    print('Installing CPU version of pytorch...')
    install_torch_cpu()
    print('──────────────────────────────────────────────')

    # Pythonフォルダ内の __pycache__ を削除する
    print('Removing __pycache__ folders...')
    remove_cache_files(PYTHON_DIR)
    print('──────────────────────────────────────────────')
    # Pythonフォルダをコピーする
    print(f'Copying Python dir: {PYTHON_DIR}')
    shutil.copytree(PYTHON_DIR, release_dir / PYTHON_DIR.name)
    print('──────────────────────────────────────────────')
    # Pythonフォルダ内の __pycache__ を再度削除する
    print('Removing __pycache__ folders...')
    remove_cache_files(release_dir / PYTHON_DIR.name)
    print('──────────────────────────────────────────────')

    # zip圧縮する
    zip_path = Path(str(release_dir) + '.zip')
    if zip_path.exists():
        send2trash(str(zip_path))
        print(f'Removed old zip: {zip_path}')

    print(f'Creating zip: {zip_path}')
    shutil.make_archive(
        str(release_dir),
        format='zip',
        root_dir=RELEASE_ROOT,
        base_dir=release_dir.name,
    )
    print(f'Created zip: {zip_path}')
    print('──────────────────────────────────────────────')

    zip_size = zip_path.stat().st_size
    print(f'Zip size: {zip_size / 1024**2:.2f} MB')
    print('Done!')


if __name__ == '__main__':
    version = input('kuresamplerのバージョンを入力してください。\n>>> ').lstrip('v')
    assert '.' in version
    prepare_release(version)
    input('Press Enter to exit...\n')

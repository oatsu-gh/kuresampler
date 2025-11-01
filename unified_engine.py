#!/usr/bin/env python3
# Copyright (c) 2025 oatsu
# ruff: noqa: T201, T203 (print, pprint)
r"""UTAU の temp.bat と helper.bat を読み取って、wavtool と resampler をまとめて実行する。

wavtool として実行されることを想定している。
resampler として実行されると初回か否かの判定ができないため。


temp_helper.bat の内容
--------------------
@if exist %temp% goto A
@if exist "%cachedir%\%9_*.wav" del "%cachedir%\%9_*.wav"
@"%resamp%" %1 %temp% %2 %vel% %flag% %5 %6 %7 %8 %params%
:A
@"%tool%" "%output%" %temp% %stp% %3 %env%
---------------------
"""
import argparse
import functools
import logging
import os
import shlex
import sys
from copy import copy
from logging import INFO, Logger
from pathlib import Path
from pprint import pprint
from warnings import warn

import colored_traceback.auto  # noqa: F401
import librosa
import numpy as np
import torch
from nnsvs.util import StandardScaler
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from tqdm import tqdm
from tqdm.contrib import tzip
from tqdm.contrib.concurrent import thread_map
from tqdm.contrib.logging import logging_redirect_tqdm

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))  # for local import
from convert import waveform_to_wavfile
from resampler import NeuralNetworkResamp
from util import fade_waveform, load_vocoder_model, overlap_waveform, setup_logger, str2float
from wavtool import NeuralNetworkWavTool

TEMP_BAT = Path('temp.bat')
TEMP_WAV = Path('temp.wav')
TEMP_NPZ = Path('temp.npz')
DEFAULT_ENCODING = 'cp932'
DEFAULT_DTYPE = 'float64'
MAX_WORKERS = (os.cpu_count() or 1) * 2

def i_am_the_first(wav_path: Path) -> bool:
    """自分が1番目の処理なのかそれ以外なのかを判定する。

    判定方法: temp.wav があるかどうかで判定
    """
    # すでに temp.wav が存在する場合は、2番目以降の処理と判定して False を返す。
    if wav_path.exists():
        return False
    # temp.wav が存在しない場合は1番目の処理と判断して、
    # True を返すとともに、ダミーwavを生成する。
    wav_path.touch()
    return True


def emulate_helper_bat(
    helper_path: Path,
    args: list,
    *,
    encoding: str = DEFAULT_ENCODING,
    temp: str,
    cachedir: str,
    resamp: str,
    vel: str,
    flag: str,
    params: str,
    output: str,
    stp: str,
    env: str,
    tool: str,
    **_unused_kwargs,
):
    r"""temp_helper.bat の内容をエミュレートする。

    もとの temp_helper.bat の内容:
        ----------------------
        @if exist %temp% goto A
        @if exist "%cachedir%\%9_*.wav" del "%cachedir%\%9_*.wav"
        @"%resamp%" %1 %temp% %2 %vel% %flag% %5 %6 %7 %8 %params%
        :A
        @"%tool%" "%output%" %temp% %stp% %3 %env%
        ---------------------

    処理内容:
    - %temp% (キャッシュwav) が存在する場合は、resamplerの処理をスキップする。
    - %cachedir%\%9_*.wav (キャッシュwav) が存在する場合は削除する。
    - resamplerを実行し、%temp% にwavを生成する。
    """
    resamp_command = ''
    tool_command = ''
    expected_helper_lines = [
        r'@if exist %temp% goto A',
        r'@if exist "%cachedir%\%9_*.wav" del "%cachedir%\%9_*.wav"',
        r'@"%resamp%" %1 %temp% %2 %vel% %flag% %5 %6 %7 %8 %params%',
        r':A',
        r'@"%tool%" "%output%" %temp% %stp% %3 %env%',
    ]
    # helper.bat の内容を読み取って、内容が想定通りか比較する。
    if helper_path.exists():
        with helper_path.open(encoding=encoding) as f:
            actual_helper_lines = [line.strip() for line in f.read().splitlines() if line.strip()]
        # helper.bat の内容が期待されるものと異なる場合はエラーを出す
        if actual_helper_lines != expected_helper_lines:
            msg = (
                f'temp_helper.bat の内容が想定と一致しません。'
                'UTAUのバージョンが異なる可能性があります。\n'
                f'  Expected:\n{expected_helper_lines}\n'
                f'  Actual:\n{actual_helper_lines}'
            )
            warn(msg, stacklevel=2)

    # temp.wav が存在しない場合のみ、
    # キャッシュwavを削除してresampler用のコマンドを生成する。
    if not Path(temp).exists():
        # resamplerコマンドを生成
        resamp_command = f'"{resamp}" {args[0]} {temp} {args[1]} {vel} {flag} {args[4]} {args[5]} {args[6]} {args[7]} {params}'  # noqa: E501
        # %cachedir%\%9_*.wav (キャッシュwav) を削除
        cachedir_path = Path(cachedir)
        if cachedir_path.exists():
            for wav_file in cachedir_path.glob(f'{args[8]}_*.wav'):
                wav_file.unlink(missing_ok=True)

    # wavtoolコマンドを生成
    tool_command = f'"{tool}" "{output}" {temp} {stp} {args[2]} {env}'

    # コマンド文字列を返す
    return resamp_command, tool_command


def clear_helper_bat(helper_path: Path):
    """helper.bat の内容を空文字列にする"""
    with helper_path.open('w', encoding=DEFAULT_ENCODING) as f:
        f.write(f'@REM helper.bat was cleared by {Path(__file__).name}\n')


def parse_temp_bat(
    temp_bat_path: Path = TEMP_BAT,
    encoding: str = DEFAULT_ENCODING,
):
    """temp.bat を読み取って、その内容と同等の処理をする"""
    # temp.bat を読み取る
    if not temp_bat_path.exists():
        msg = f'{temp_bat_path} is not found.'
        raise FileNotFoundError(msg)
    with temp_bat_path.open(encoding=encoding) as f:
        lines = f.read().splitlines()

    # 前後の空白文字と"@"を削除
    lines = [line.strip().lstrip('@') for line in lines]
    # 空の行を削除
    lines = [line for line in lines if line]
    # コメント行を削除
    lines = [line for line in lines if not line.strip().startswith('::')]

    variables = {}
    resamp_commands = []  # resampler実行コマンドのリスト
    tool_commands = []  # wavtool実行コマンドのリスト

    # helper.bat のパス (バッチファイルに記載があればそちらで後ほど上書きされる)
    helper_path: Path = Path('helper.bat')

    for line in lines:
        # 変数定義
        if line.startswith('set '):
            key, value = line[4:].split('=', 1)
            variables[key.strip()] = value.strip()
            helper_path = Path(variables.get('helper', 'helper.bat'))
            continue
        # wavtool を実行する行の場合
        if line.startswith(r'"%tool%"'):
            tool_commands.append(line)
            continue
        # helper.bat を実行する行の場合
        if line.startswith(r'call %helper%'):
            if not Path(helper_path).exists():
                msg = f'Helper file not found: {helper_path}'
                raise FileNotFoundError(msg)
            # コマンドライン引数を抽出
            args = line.split()[2:]  # 'call %helper%' の後ろの部分
            # helper.bat をエミュレートして resampler 用と wavfile 用のコマンドを取得
            resamp_cmd, tool_cmd = emulate_helper_bat(
                helper_path,
                args=args,
                **variables,
            )
            if resamp_cmd:
                resamp_commands.append(resamp_cmd)
            if tool_cmd:
                tool_commands.append(tool_cmd)
            continue

    # resamp_commands と tool_commands の中の特定の文字列を置換する
    for i in range(len(resamp_commands)):
        for key, value in variables.items():
            resamp_commands[i] = resamp_commands[i].replace(f'%{key}%', value)
    for i in range(len(tool_commands)):
        for key, value in variables.items():
            tool_commands[i] = tool_commands[i].replace(f'%{key}%', value)
    # 各コマンドをリストに分割する
    resamp_commands = [shlex.split(cmd) for cmd in resamp_commands if cmd.strip()]
    tool_commands = [shlex.split(cmd) for cmd in tool_commands if cmd.strip()]
    return variables, resamp_commands, tool_commands


def _process_resampler_command(cmd_and_logger: tuple[list[str], Logger]) -> None:
    """単一の resampler コマンドを処理する（マルチプロセス用ワーカー関数）。

    Args:
        cmd_and_logger: (resampler_command, logger) のタプル

    """
    cmd, logger = cmd_and_logger

    logger.debug(cmd)
    len_cmd = len(cmd)
    # 引数の数をチェック
    if len_cmd < 5 or len_cmd > 14:
        logger.error(f'Number of arguments must be 5 to 14 ({len_cmd}): {cmd}')
        return

    # 引数を14個に揃える
    cmd_14 = cmd + [None] * (14 - len_cmd)
    (
        input_path,
        output_path,
        target_tone,
        velocity,
        flag_value,
        offset,
        target_ms,
        fixed_ms,
        end_ms,
        volume,
        modulation,
        tempo,
        pitchbend,
    ) = cmd_14[1:]  # cmd[0] は resampler の実行ファイルパス

    logger.debug(f'  input_path  : {Path(input_path).name}')  # pyright: ignore[reportArgumentType]
    logger.debug(f'  output_path : {Path(output_path).name}')  # pyright: ignore[reportArgumentType]
    logger.debug(f'  target_tone : {target_tone}')
    logger.debug(f'  velocity    : {velocity}')
    logger.debug(f'  flag_value  : {flag_value}')
    logger.debug(f'  offset      : {offset}')
    logger.debug(f'  target_ms   : {target_ms}')
    logger.debug(f'  fixed_ms    : {fixed_ms}')
    logger.debug(f'  end_ms      : {end_ms}')
    logger.debug(f'  volume      : {volume}')
    logger.debug(f'  modulation  : {modulation}')
    logger.debug(f'  tempo       : {tempo}')
    logger.debug(f'  pitchbend   : {pitchbend}')

    try:
        resampler = NeuralNetworkResamp(
            input_path=input_path,  # pyright: ignore[reportArgumentType]
            output_path=output_path,  # pyright: ignore[reportArgumentType]
            target_tone=target_tone,  # pyright: ignore[reportArgumentType]
            velocity=velocity,  # pyright: ignore[reportArgumentType]
            flag_value=flag_value,
            offset=offset,
            target_ms=target_ms,
            fixed_ms=fixed_ms,
            end_ms=end_ms,
            volume=volume,
            modulation=modulation,
            tempo=tempo,
            pitchbend=pitchbend,
            use_vocoder_model=False,
            logger=logger,
            export_features=True,
        )

        resampler.resamp()
    # 例外を握り潰す
    except Exception as e:
        logger.error(f'Resampler error: {e}')


def batch_resampler(
    logger: Logger, resampler_commands: list[list[str]], *, max_workers: int = MAX_WORKERS
):
    """resampler_commands に基づいて resampler をマルチプロセスで実行する。

    コマンドの例
    -----------------------
    @"%resamp%" %1 %temp% %2 %vel% %flag% %5 %6 %7 %8 %params%
    -----------------------

    """
    # 各コマンドとロガーをタプルにして準備
    commands_with_logger = [(cmd, logger) for cmd in resampler_commands]

    # マルチスレッドでのログ出力抑制
    original_log_level = logger.level
    logger.setLevel(logging.WARNING)

    # マルチプロセスで resampler を実行
    with logging_redirect_tqdm([logger]):
        thread_map(
            _process_resampler_command,
            commands_with_logger,
            desc='Resampler',
            unit='note',
            colour='green',
            chunksize=1,
            mininterval=0,
            max_workers=max_workers,
        )
    # logger レベルを元に戻す
    logger.setLevel(original_log_level)


def generate_silent_waveform(duration_ms: float, sample_rate: int, dtype='float64') -> np.ndarray:
    """指定された長さの無音波形を生成する。

    Args:
        duration_ms: 無音の長さ[ms]
        sample_rate: サンプリングレート[Hz]
        dtype: waveform のデータ型

    Returns:
        無音波形 (np.ndarray)

    """
    num_samples = int(duration_ms * sample_rate / 1000)
    return np.zeros(num_samples, dtype=dtype)


def batch_wavtool(
    wavtool_commands: list[list[str]],
    *,
    logger: Logger,
    use_vocoder_model: bool,
    frame_period: int = 5,
    vocoder_model: torch.nn.Module | None = None,
    vocoder_in_scaler: StandardScaler | None = None,
    vocoder_config: DictConfig | ListConfig | None = None,
    vocoder_type: str = 'usfgan',
    vocoder_feature_type: str = 'world',
    vocoder_vuv_threshold: float = 0.5,
    vocoder_frame_period: int = 5,
    internal_sample_rate: int = 48000,
    target_sample_rate: int = 44100,
    resample_type: str = 'soxr_vhq',
    dtype: str = DEFAULT_DTYPE,
) -> tuple[np.ndarray, float, float]:
    """wavetool_commands に基づいて wavtool を順次実行する。

    コマンドの例
    -----------------------
    <tool> <output> <temp> <stp> <length> <envelope>
    -----------------------
        tool   : wavtool.exe など実行ファイルのパス
        output : 出力wavファイルパス
        temp   : 入力wavファイルパス
        stp    : 入力wav使用開始位置[ms]
        length : ノート長さ[ms]
        env    : 音量エンベロープのパラメータ (複数可)

    """
    # ノート数
    n_notes = len(wavtool_commands)
    # wavtool インスタンス保持用の変数
    wavtool: NeuralNetworkWavTool
    # ノート時刻の丸め誤差
    residual_error: float = 0.0
    # メモリ上で累積特徴量を保持 (ファイルI/Oを減らすため)
    accumulated_features: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # ノート数が 0 の時は何もしないで空の値を返す
    if n_notes == 0:
        msg = 'wavtool に渡されるノート数が 0 です。'
        raise ValueError(msg)
    # 各ノートのパラメータを格納するリスト
    overlap_list: list[float] = []
    residual_error_list: list[float] = []

    # 各ノートのwav加工を行う
    with logging_redirect_tqdm([logger]):
        for cmd in tqdm(
            wavtool_commands,
            mininterval=0,
            desc='WavTool',
            unit='note',
            colour='blue',
        ):
            try:
                logger.debug(cmd)
                if len(cmd) < 6:
                    logger.error(
                        f'Number of wavtool arguments must be 6 or larger ({len(cmd)}): {cmd}'
                    )
                    continue
                (
                    output_path,
                    input_path,
                    stp,
                    length,
                    *envelope,
                ) = cmd[1:]  # cmd[0] は wavtool の実行ファイルパス

                length_ms = str2float(length)

                # 各パラメータをログ出力
                logger.debug(f'  output_path  : {Path(output_path).name}')
                logger.debug(f'  input_path   : {Path(input_path).name}')
                logger.debug(f'  stp          : {stp}')
                logger.debug(f'  length (str) : {length}')
                logger.debug(f'  length (ms)  : {length_ms} [ms]')
                logger.debug(f'  envelope     : {envelope}')

                logger.debug('residual_error (before wavtool) : %.3f [ms]', residual_error)
                # wavtool インスタンスを生成
                wavtool = NeuralNetworkWavTool(
                    output_wav=output_path,
                    input_wav=input_path,
                    stp=float(stp),
                    length=length_ms,
                    envelope=list(map(float, envelope)),
                    logger=logger,
                    use_vocoder_model=use_vocoder_model,
                    frame_period=frame_period,
                    residual_error=residual_error,
                    vocoder_model=vocoder_model,
                    vocoder_in_scaler=vocoder_in_scaler,
                    vocoder_config=vocoder_config,
                    vocoder_type=vocoder_type,
                    vocoder_feature_type=vocoder_feature_type,
                    vocoder_vuv_threshold=vocoder_vuv_threshold,
                    vocoder_frame_period=vocoder_frame_period,
                    internal_sample_rate=internal_sample_rate,
                    target_sample_rate=target_sample_rate,
                    resample_type=resample_type,
                    accumulated_features=accumulated_features,
                )
                residual_error = wavtool.residual_error
                logger.debug('residual_error (after wavtool)  : %.3f [ms]', residual_error)

                # wavtool を実行
                wavtool.append()
                # メモリ上で累積特徴量を更新
                accumulated_features = (
                    wavtool.f0_appended,
                    wavtool.sp_appended,
                    wavtool.ap_appended,
                )
                # 各ノートのオーバーラップ長[ms]を保存
                overlap_list.append(wavtool.overlap)
                # 各ノートの residual_error[ms] を保存
                residual_error_list.append(residual_error)
            except Exception as e:
                logger.critical(f'Exception occurred for the note: {cmd}')
                raise e

    # 最終ノートの wavtool が持っている特徴量と wav を保存
    if accumulated_features is None:
        msg = 'accumulated_features is None after wavtool processing.'
        raise RuntimeError(msg)
    logger.debug('Synthesizing waveform...')
    logger.debug(f'Accumulated features shape: {[feat.shape for feat in accumulated_features]}')

    # 特徴量の要素数が0の場合は waveform で空のndarrayを返す
    if accumulated_features[0].shape[0] == 0:
        logger.debug('No features to synthesize. Returning empty waveform.')
        return np.array([], dtype=DEFAULT_DTYPE), overlap_list[0], residual_error_list[-1]

    # 無音かどうかの判定
    sp_appended = accumulated_features[1]
    seg_is_silence = sp_appended.sum() == 0.0
    # 特徴量の要素数が0ではないが無音の場合は、ボコーダーを通さずに完全な無音のwaveformを生成する
    if seg_is_silence:
        total_duration_ms = sp_appended.shape[0] * frame_period
        logger.debug(
            'This segment is a silence. Generating silent waveform (%.3f ms).',
            total_duration_ms,
        )
        silent_waveform = generate_silent_waveform(
            duration_ms=total_duration_ms,
            sample_rate=target_sample_rate,
            dtype=DEFAULT_DTYPE,
        )
        resampled_waveform = silent_waveform
        first_overlap_ms = overlap_list[0]
        last_residual_error_ms = residual_error_list[-1]

    # 特徴量の要素数が0ではないかつ無音ではない場合は、ボコーダーでwaveformを合成する
    else:
        wavtool.synthesize()  # pyright: ignore[reportPossiblyUnboundVariable]
        logger.debug('Synthesis complete.')

        # 結合された特徴量をもとに生成した waveform
        waveform = copy(wavtool.waveform)  # pyright: ignore[reportPossiblyUnboundVariable]
        resampled_waveform = librosa.resample(
            waveform,
            orig_sr=internal_sample_rate,
            target_sr=target_sample_rate,
            res_type=resample_type,
        )
        first_overlap_ms = overlap_list[0]
        last_residual_error_ms = residual_error_list[-1]

    # 生成した waveform と 音声長の誤差[ms] を返す
    return resampled_waveform.astype(dtype), first_overlap_ms, last_residual_error_ms


def fix_waveform_length(
    waveform: np.ndarray, residual_error_ms: float, sample_rate: int
) -> np.ndarray:
    """Waveform の長さを residual_error_ms に基づいて修正する。

    Args:
        waveform: 入力波形
        residual_error_ms: 波形長の誤差[ms]
        sample_rate: サンプリングレート[Hz]

    Returns:
        修正後の波形

    """
    num_error_samples = round(residual_error_ms * sample_rate / 1000)
    # wav が目標よりも短い場合はゼロパディングする。
    if num_error_samples > 0:
        waveform = np.pad(waveform, (0, num_error_samples))
    # wav が目標よりも長い場合は切り詰める。
    elif num_error_samples < 0:
        waveform = waveform[:num_error_samples]
    else:
        pass  # 誤差なし
    return waveform


def segmented_wavtool(
    logger: Logger,
    wavtool_commands: list[list[str]],
    *,
    use_vocoder_model: bool,
    frame_period: int = 5,
    vocoder_model: torch.nn.Module | None = None,
    vocoder_in_scaler: StandardScaler | None = None,
    vocoder_config: DictConfig | ListConfig | None = None,
    vocoder_type: str = 'usfgan',
    vocoder_feature_type: str = 'world',
    vocoder_vuv_threshold: float = 0.5,
    vocoder_frame_period: int = 5,
    internal_sample_rate: int = 48000,
    target_sample_rate: int = 44100,
    resample_type: str = 'soxr_vhq',
):
    """休符ごとにwav生成して最後につなげる。

    wavtool コマンドの例
    -----------------------
    <tool> <output> <temp> <stp> <length> <envelope>
    -----------------------

    """
    # 各ノートの入力ファイルの有無
    is_silence_note_list: list[bool] = [(not Path(cmd[2]).exists()) for cmd in wavtool_commands]

    # 休符で区切ってセグメントを作成
    segment_list: list[list[list[str]]] = []
    is_silent_segment_list: list[bool] = []
    current_segment: list[list[str]] = []

    # ノートを休符区切りでグループ化してセグメントにする。
    # 休符が出現したり、休符の後に音符が出現したらセグメントを区切る
    with logging_redirect_tqdm([logger]):
        for cmd, current_note_is_silence in tzip(
            wavtool_commands,
            is_silence_note_list,
            mininterval=0,
            desc='Segmenting',
            unit='note',
            colour='cyan',
        ):
            # 休符や無音の場合は、セグメントを切り替える
            if current_note_is_silence:
                logger.debug('********* 休符です!!!! ***********')
                # 既存セグメントが空でない場合
                if len(current_segment) > 0:
                    is_silent_segment_list.append(False)
                    segment_list.append(current_segment)
                # 休符自身単体をセグメントのリストに追加し、次ノートのために新規セグメントを開始
                is_silent_segment_list.append(True)
                segment_list.append([cmd])
                current_segment = []
            # 音符の場合はセグメントに追加
            else:
                current_segment.append(cmd)

    # ループ後に残ったセグメントをリストに追加
    if current_segment:
        is_silent_segment_list.append(False)
        segment_list.append(current_segment)

    # セグメントのリストの要素数をチェック
    if len(segment_list) != len(is_silent_segment_list):
        msg = (
            f'lengths of segment_list ({len(segment_list)}) '
            f'and is_silent_segment_list ({len(is_silent_segment_list)}) do not match.'
        )
        raise ValueError(msg)

    # セグメント状態をログ出力
    logger.info(f'Total segments: {len(segment_list)}')
    logger.debug(f'Note count in each segment        : {[len(seg) for seg in segment_list]}')
    logger.debug(f'Silence or non-silence of segment : {is_silent_segment_list}')

    # 部分関数を作成
    partial_batch_wavtool = functools.partial(
        batch_wavtool,
        logger=logger,
        use_vocoder_model=use_vocoder_model,
        frame_period=frame_period,
        vocoder_model=vocoder_model,
        vocoder_in_scaler=vocoder_in_scaler,
        vocoder_config=vocoder_config,
        vocoder_type=vocoder_type,
        vocoder_feature_type=vocoder_feature_type,
        vocoder_vuv_threshold=vocoder_vuv_threshold,
        vocoder_frame_period=vocoder_frame_period,
        internal_sample_rate=internal_sample_rate,
        target_sample_rate=target_sample_rate,
        resample_type=resample_type,
    )

    # セグメントごとに waveform を生成してリストに格納
    # TODO: 休符セグメントは無音波形を生成する。overlap と residual_error に注意。
    waveform_list: list[np.ndarray] = []
    residual_error_list: list[float] = []
    overlap_list: list[float] = []
    with logging_redirect_tqdm([logger]):
        for seg_commands in tqdm(
            segment_list, desc='Synthesizing wav segments', unit='seg', colour='magenta'
        ):
            wav, first_overlap, last_residual_error = partial_batch_wavtool(seg_commands)
            waveform_list.append(wav)
            overlap_list.append(first_overlap)
            residual_error_list.append(last_residual_error)

    # zip ループを回す前に要素数チェック
    if not (len(waveform_list) == len(overlap_list) == len(residual_error_list)):
        msg = (
            f'Length of waveform_list ({len(waveform_list)}) and '
            f'adjusted_overlap_list ({len(overlap_list)}) and '
            f'residual_error_list ({len(residual_error_list)}) do not match.'
        )
        raise ValueError(msg)
    # 各セグメントの waveform をつなげる
    long_waveform: np.ndarray = waveform_list[0]
    adjusted_overlap: float = overlap_list[0]
    residual_error: float = residual_error_list[0]
    with logging_redirect_tqdm([logger]):
        for seg_wav, seg_overlap, seg_res_err in tzip(
            waveform_list[1:],
            overlap_list[1:],
            residual_error_list[1:],
            mininterval=0,
            colour='red',
            desc='Concatenating waveform segments',
            unit='seg',
        ):
            # オーバーラップ値を補正する
            ## residual_error が正の時は wav が短めなので、次の休符とのoverlapを短くする。
            ## residual_error が負の時は wav が長めなので、次の休符とのoverlapを長くする。
            adjusted_overlap = seg_overlap - residual_error
            residual_error = seg_res_err
            logger.debug('Overlapping waveforms with overlap: %.3f [ms]', adjusted_overlap)
            logger.debug('Waveform shapes: %s, %s', long_waveform.shape, seg_wav.shape)
            # waveform の長さが0の場合はスキップする
            if len(seg_wav) == 0:
                logger.debug('Skipping empty segment waveform.')
                # オーバーラップが行われないことでずれるので次のノートの引き継がせる
                # 次のノートのオーバーラップを長くしないといけない -> res_error を小さくする
                residual_error = seg_res_err - adjusted_overlap
                continue
            # 微小なフェードインとフェードアウトを行う
            faded_seg_wav = fade_waveform(
                seg_wav,
                fade_in_ms=frame_period,
                fade_out_ms=frame_period,
                sample_rate=target_sample_rate,
            )
            # waveform をオーバーラップさせて結合する
            long_waveform = overlap_waveform(
                long_waveform,
                faded_seg_wav,
                overlap_ms=adjusted_overlap,
                sample_rate=target_sample_rate,
            )
    # 最終的な waveform を返す
    return long_waveform, adjusted_overlap, residual_error

# noqa: T201
def main():
    """全体の処理を行う。"""
    pprint(sys.argv)

    parser = argparse.ArgumentParser(description='UTAU wavtool crossfading WORLD features')
    # 中間のコマンドライン引数は無視
    parser.add_argument(
        '_unused_args',
        nargs='*',
        help='Unused wavtool arguments',
    )
    # ボコーダーモデルを使用するか否か
    parser.add_argument(
        '--use_vocoder_model',
        help='Whether to use vocoder model for waveform synthesis. If False, use WORLD.',
        action='store_true',
        default=False,
    )

    # モデルを指定
    parser.add_argument(
        '--model_dir',
        help='Vocoder model directory (optional; required for neural network vocoder)',
        type=str,
        default=None,
    )
    # デバッグモード
    parser.add_argument(
        '--debug',
        help='Enable debug logging',
        action='store_true',
        default=False,
    )
    args = parser.parse_args()
    logger = setup_logger(INFO)

    # デバッグモード
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # temp.bat を解析して変数とコマンドを取得
    bat_variables, resamp_commands, wavtool_commands = parse_temp_bat()
    # バッチファイル内の変数を取得
    sample_rate = int(bat_variables.get('samples', 44100))
    wav_out_path = Path(bat_variables.get('output', TEMP_WAV))
    npz_out_path = wav_out_path.with_suffix('.npz')
    # 2番目以降の処理の場合
    if not i_am_the_first(wav_out_path):
        print("I'm not the first process. Just passing through.")
        return

    # ボコーダーモデルを使用するか否か
    use_vocoder_model = args.use_vocoder_model

    # モデルロードを試みる
    if use_vocoder_model:
        if args.model_dir is None:
            msg = 'When --use_vocoder_model is specified, --model_dir must be provided.'
            logger.error(msg)
            raise ValueError(msg)
        vocoder_model, vocoder_in_scaler, vocoder_config = load_vocoder_model(args.model_dir)
    else:
        vocoder_model = None
        vocoder_in_scaler = None
        vocoder_config = None

    # helper.bat の内容を空にする
    helper_path = Path(bat_variables.get('helper', 'helper.bat'))
    clear_helper_bat(helper_path)

    if args.debug:
        print('\nVariables:')
        pprint(bat_variables)
        print(f'\nResampler commands ({len(resamp_commands)}):')
        pprint(resamp_commands, compact=True)
        print(f'\nWavTool commands ({len(wavtool_commands)}):')
        pprint(wavtool_commands, compact=True)

    print('------------------------------------')
    # resampler を順次実行する
    batch_resampler(logger, resamp_commands)
    print('------------------------------------')

    # 一時ファイルを削除
    wav_out_path.unlink(missing_ok=True)
    npz_out_path.unlink(missing_ok=True)

    # wavtool を順次実行する(一括版)
    # waveform, _first_overlap, last_residual_error = batch_wavtool(
    #     wavtool_commands,
    #     logger=logger,
    #     use_vocoder_model=use_vocoder_model,
    #     vocoder_model=vocoder_model,
    #     vocoder_in_scaler=vocoder_in_scaler,
    #     vocoder_config=vocoder_config,
    #     target_sample_rate=sample_rate,
    # )
    # # 最終的な長さずれ分のサンプル数を補正する
    # waveform = fix_waveform_length(
    #     waveform,
    #     last_residual_error,
    #     sample_rate=sample_rate,
    # )
    # # WAV ファイル出力
    # waveform_to_wavfile(
    #     waveform,
    #     wav_out_path,
    #     original_sample_rate=sample_rate,
    #     target_sample_rate=sample_rate,
    # )

    # segmented wavtool を実行する (セグメント版)
    waveform, _first_overlap, last_residual_error = segmented_wavtool(
        logger,
        wavtool_commands,
        use_vocoder_model=use_vocoder_model,
        vocoder_model=vocoder_model,
        vocoder_in_scaler=vocoder_in_scaler,
        vocoder_config=vocoder_config,
        target_sample_rate=sample_rate,
    )
    # 最終セグメントの長さずれの分だけサンプル数を補正する
    waveform = fix_waveform_length(
        waveform,
        last_residual_error,
        sample_rate=sample_rate,
    )
    # WAV ファイル出力
    waveform_to_wavfile(
        waveform,
        wav_out_path,
        original_sample_rate=sample_rate,
        target_sample_rate=sample_rate,
    )
    print('----------------- END -------------------')


if __name__ == '__main__':
    main()

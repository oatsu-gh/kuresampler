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
import shlex
import sys
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING, Logger  # noqa: F401
from pathlib import Path
from pprint import pprint
from warnings import warn

import colored_traceback.auto  # noqa: F401
import torch
from nnsvs.util import StandardScaler
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from tqdm import tqdm
from tqdm.contrib import tenumerate

from convert import world_to_npzfile

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent))  # for local import
from resampler import NeuralNetworkResamp
from util import load_vocoder_model, setup_logger, str2float
from wavtool import NeuralNetworkWavTool

TEMP_BAT = Path('temp.bat')
TEMP_WAV = Path('temp.wav')
TEMP_NPZ = Path('temp.npz')
DEFAULT_ENCODING = 'cp932'


def i_am_the_first(temp_wav_path: Path = TEMP_WAV) -> bool:
    """自分が1番目の処理なのかそれ以外なのかを判定する。

    判定方法: temp.wav があるかどうかで判定
    """
    # すでに temp.wav が存在する場合は、2番目以降の処理と判定して False を返す。
    if temp_wav_path.exists():
        return False
    # temp.wav が存在しない場合は1番目の処理と判断して、
    # True を返すとともに、ダミーwavを生成する。
    temp_wav_path.touch()
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


def batch_resampler(logger: Logger, resampler_commands: list[list[str]]):
    """resampler_commands に基づいて resampler を順次実行する。

    コマンドの例
    -----------------------
    @"%resamp%" %1 %temp% %2 %vel% %flag% %5 %6 %7 %8 %params%
    -----------------------

    """
    # 各ノートのピッチシフトや伸縮を行う
    for cmd in tqdm(resampler_commands, desc='Resampler', unit='note', colour='green'):
        print()
        logger.info(cmd)
        if len(cmd) != 14:
            logger.error(f'Unexpected number of arguments ({len(cmd)}): {cmd}')
            continue
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
        ) = cmd[1:]  # cmd[0] は resampler の実行ファイルパス

        logger.debug(f'  input_path  : {Path(input_path).name}')
        logger.debug(f'  output_path : {Path(output_path).name}')
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

        resampler = NeuralNetworkResamp(
            input_path=input_path,
            output_path=output_path,
            target_tone=target_tone,
            velocity=int(velocity),
            flag_value=flag_value,
            offset=float(offset),
            target_ms=float(target_ms),
            fixed_ms=float(fixed_ms),
            end_ms=float(end_ms),
            volume=int(volume),
            modulation=int(modulation),
            tempo=str(tempo),
            pitchbend=pitchbend,
            use_vocoder_model=False,
            logger=logger,
            export_features=True,
        )

        resampler.resamp()


def batch_wavetool(
    logger: Logger,
    wavetool_commands: list[list[str]],
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
    """wavetool_commands に基づいて wavtool を順次実行する。

    コマンドの例
    -----------------------
    @"%tool%" "%output%" %temp% %stp% %3 %env%
    -----------------------

    """
    # 一時ファイルを削除
    TEMP_NPZ.unlink(missing_ok=True)
    n_notes = len(wavetool_commands)
    # ノート時刻の丸め誤差
    residual_error: float = 0.0
    # メモリ上で累積特徴量を保持 (ファイルI/Oを減らすため)
    accumulated_features: tuple | None = None

    # 各ノートのwav加工を行う
    for i, cmd in tenumerate(
        wavetool_commands,
        mininterval=0,
        desc='WavTool',
        unit='note',
        colour='blue',
    ):
        print()
        logger.info(cmd)
        if len(cmd) < 6:
            logger.error(f'Number of arguments must be 6 or larger ({len(cmd)}): {cmd}')
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
        # 最終ノートの時のみ npz と wav を出力
        if i == n_notes - 1:
            # npz 出力
            world_to_npzfile(
                wavtool.f0_appended,
                wavtool.sp_appended,
                wavtool.ap_appended,
                wavtool.internal_sample_rate,
                wavtool.output_npz,
                compress=False,
            )
            logger.info('Rendering WAV...')
            wavtool.synthesize()
            logger.info('Render complete.')


# noqa: T201
def main():
    """全体の処理を行う。"""
    print(sys.argv)
    # 2番目以降の処理の場合
    if not i_am_the_first():
        print("I'm not the first process. Just passing through.")
        return

    # 1番目の処理の場合
    print('\nI am the first process!')
    logger = setup_logger(INFO)

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

    # デバッグモード
    if args.debug:
        logger.setLevel(DEBUG)

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

    variables, resamp_commands, wavtool_commands = parse_temp_bat()
    # helper.bat の内容を空にする
    helper_path = Path(variables.get('helper', 'helper.bat'))
    clear_helper_bat(helper_path)
    print('\nVariables:')
    pprint(variables)
    print(f'\nResampler commands ({len(resamp_commands)}):')
    pprint(resamp_commands, compact=True)
    print(f'\nWavTool commands ({len(wavtool_commands)}):')
    pprint(wavtool_commands, compact=True)

    print('------------------------------------')
    # resampler を順次実行する
    batch_resampler(logger, resamp_commands)
    print('------------------------------------')
    # wavtool を順次実行する
    batch_wavetool(
        logger,
        wavtool_commands,
        use_vocoder_model=use_vocoder_model,
        vocoder_model=vocoder_model,
        vocoder_in_scaler=vocoder_in_scaler,
        vocoder_config=vocoder_config,
    )
    print('----------------- END -------------------')


if __name__ == '__main__':
    main()

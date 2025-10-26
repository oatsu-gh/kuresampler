#!/usr/bin/env python3
# Copyright (c) 2025 oatsu
r"""
UTAU の temp.bat と helper.bat を読み取って蓄積し、エンジンを一括実行できるようにする。

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

import shlex
from pathlib import Path
from pprint import pprint
from warnings import warn

TEMP_BAT = Path("temp.bat")
TEMP_WAV = Path("temp.wav")
DEFAULT_ENCODING = "cp932"


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
    resamp_command = ""
    tool_command = ""
    expected_helper_lines = [
        r"@if exist %temp% goto A",
        r'@if exist "%cachedir%\%9_*.wav" del "%cachedir%\%9_*.wav"',
        r'@"%resamp%" %1 %temp% %2 %vel% %flag% %5 %6 %7 %8 %params%',
        r":A",
        r'@"%tool%" "%output%" %temp% %stp% %3 %env%',
    ]
    # helper.bat の内容を読み取って、内容が想定通りか比較する。
    if helper_path.exists():
        with helper_path.open(encoding=encoding) as f:
            actual_helper_lines = [
                line.strip() for line in f.read().splitlines() if line.strip()
            ]
        # helper.bat の内容が期待されるものと異なる場合はエラーを出す
        if actual_helper_lines != expected_helper_lines:
            msg = (
                f"temp_helper.bat の内容が想定と一致しません。"
                "UTAUのバージョンが異なる可能性があります。\n"
                f"  Expected:\n{expected_helper_lines}\n"
                f"  Actual:\n{actual_helper_lines}"
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
            for wav_file in cachedir_path.glob(f"{args[8]}_*.wav"):
                wav_file.unlink(missing_ok=True)

    # wavtoolコマンドを生成
    tool_command = f'"{tool}" "{output}" {temp} {stp} {args[2]} {env}'

    # コマンド文字列を返す
    return resamp_command, tool_command


def clear_helper_bat(helper_path: Path):
    """helper.bat の内容を空文字列にする"""
    with helper_path.open("w", encoding=DEFAULT_ENCODING) as f:
        f.write(f"@REM helper.bat was cleared by {Path(__file__).name}\n")


def parse_temp_bat(
    temp_bat_path: Path = TEMP_BAT,
    encoding: str = DEFAULT_ENCODING,
):
    """temp.bat を読み取って、その内容と同等の処理をする"""

    # temp.bat を読み取る
    if not temp_bat_path.exists():
        msg = f"{temp_bat_path} is not found."
        raise FileNotFoundError(msg)
    with temp_bat_path.open(encoding=encoding) as f:
        lines = f.read().splitlines()

    # 前後の空白文字と"@"を削除
    lines = [line.strip().lstrip("@") for line in lines]
    # 空の行を削除
    lines = [line for line in lines if line]
    # コメント行を削除
    lines = [line for line in lines if not line.strip().startswith("::")]

    variables = {}
    resamp_commands = []  # resampler実行コマンドのリスト
    tool_commands = []  # wavtool実行コマンドのリスト

    # helper.bat のパス (バッチファイルに記載があればそちらで後ほど上書きされる)
    helper_path: Path = Path("helper.bat")

    for line in lines:
        # 変数定義
        if line.startswith("set "):
            key, value = line[4:].split("=", 1)
            variables[key.strip()] = value.strip()
            helper_path = Path(variables.get("helper", "helper.bat"))
            continue
        # wavtool を実行する行の場合
        if line.startswith(r'"%tool%"'):
            tool_commands.append(line)
            continue
        # helper.bat を実行する行の場合
        if line.startswith(r"call %helper%"):
            if not Path(helper_path).exists():
                msg = f"Helper file not found: {helper_path}"
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
            resamp_commands[i] = resamp_commands[i].replace(f"%{key}%", value)
    for i in range(len(tool_commands)):
        for key, value in variables.items():
            tool_commands[i] = tool_commands[i].replace(f"%{key}%", value)
    # 各コマンドをリストに分割する
    resamp_commands = [shlex.split(cmd) for cmd in resamp_commands if cmd.strip()]
    tool_commands = [shlex.split(cmd) for cmd in tool_commands if cmd.strip()]
    return variables, resamp_commands, tool_commands


def main():
    import sys

    print(sys.argv)
    if i_am_the_first():
        print("\nI am the first process!")
        variables, resamp_commands, tool_commands = parse_temp_bat()
        # helper.bat の内容を空にする
        helper_path = Path(variables.get("helper", "helper.bat"))
        clear_helper_bat(helper_path)
        print("\nVariables:")
        pprint(variables)
        print(f"\nResampler commands ({len(resamp_commands)}):")
        pprint(resamp_commands, compact=True)
        print(f"\nWavTool commands ({len(tool_commands)}):")
        pprint(tool_commands, compact=True)
    else:
        print("I'm not the first process. Just passing through.")


if __name__ == "__main__":
    main()

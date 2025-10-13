# ruff: noqa: T201
"""kuresampler のうち resampler 部分を FastAPI サーバー化したもの。

LEIRH (https://github.com/bullets1234) さんが作ってくれました。

"""

import logging
import sys
import urllib.parse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool

# スクリプトが存在するディレクトリを sys.path に追加
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))


from resampler import main_resampler  # noqa: E402
from util import get_device, load_vocoder_model  # noqa: E402

app = FastAPI()
vocoder_model = None  # グローバル変数
vocoder_config = None  # グローバル変数
vocoder_in_scaler = None  # グローバル変数
vocoder_model_dir = None  # グローバル変数


@app.get('/health')
async def health_check():
    """Server の起動状態をチェックする。"""
    return {'message': 'Health Check OK - Server is running'}


@app.post('/load_models')
async def api_load_models(request: Request):
    """モデルを読みこむ。"""
    # do_parallel(ust_path, wavout_path)
    # global current_task_load_models

    global vocoder_model
    global vocoder_config
    global vocoder_in_scaler

    # print(await request.body())
    body = await request.body()
    args = str(urllib.parse.unquote(body))

    vocoder_model, vocoder_in_scaler, vocoder_config = await run_in_threadpool(
        load_vocoder_model, args, get_device()
    )

    return {'message': 'load_models done'}


@app.post('/resampler')
async def api_resampler(request: Request):
    r"""Resampler を実行する。

    resampler引数の例:
    <input_file> <output_file> <tone> <velocity> <flags> <offset> <length> <consonant> <cutoff> <volume> <modulation> <tempo> <pitchbends...>

    args = [
        'C:\\Users\\XXXX\\あ.wav',  # input_path
        './aaaa.wav',               # output_path
        'A4',                       # target_tone
        '107',                      # velocity (整数値)
        '',                         # flags
        '20',                       # offset (from '20@168+217.318')
        '217.318',                  # target_ms (length)
        '224.7583',                 # fixed_ms (consonant)
        '3197.114',                 # end_ms (cutoff)
        '300',                      # volume
        '0',                        # modulation (整数値)
        '!120',                     # tempo (デフォルト)
        ''                          # pitchbend (ピッチベンド値)
    ]

    """
    print(await request.body())
    body = await request.body()
    split_argument = split_arguments(str(urllib.parse.unquote(body)))

    _ = await run_in_threadpool(
        main_resampler,
        arg_list=split_argument,
        vocoder_model=vocoder_model,
        vocoder_config=vocoder_config,
        vocoder_in_scaler=vocoder_in_scaler,
    )

    return {'message': 'resampler done'}

# from hifisampler GitHub:
def split_arguments(input_string: str):
    """文字列扱いのコマンドライン引数を分割する"""
    return input_string.split(',')


if __name__ == '__main__':
    # サーバー起動
    logging.debug('sys.argv: %s', sys.argv)
    uvicorn.run(app, host='127.0.0.1', port=55903, log_level='debug')

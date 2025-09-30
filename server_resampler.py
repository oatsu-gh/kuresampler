import sys
from pathlib import Path

# スクリプトが存在するディレクトリを sys.path に追加
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import logging
import sys

from fastapi import FastAPI ,Request
from fastapi.concurrency import run_in_threadpool
import uvicorn
import urllib.parse
import asyncio

from resampler import main_resampler
from util import load_vocoder_model , get_device


import nnsvs  # pylint: disable=import-error
from nnsvs.svs import SPSVS

semaphore = asyncio.Semaphore(5)

app = FastAPI()
vocoder_model = None  # グローバル変数
vocoder_config = None  # グローバル変数
vocoder_in_scaler = None  # グローバル変数
vocoder_model_dir = None # グローバル変数
labels = None # グローバル変数

@app.get("/health")
async def health_check():
    return {"message": "Health Check OK - Server is running"}


@app.post("/load_models")
async def api_load_models(request:Request):
    # do_parallel(ust_path, wavout_path)
    # global current_task_load_models
    # await has_current_task(current_task_load_models)
    
    global vocoder_model
    global vocoder_config
    global vocoder_in_scaler

    print(await request.body())
    body = await request.body()
    args = str(urllib.parse.unquote(body))

    vocoder_model,vocoder_in_scaler,vocoder_config = await run_in_threadpool(load_vocoder_model,args,get_device())

    return {"message": "load_models done"}

@app.post("/resampler")
async def api_resampler(request:Request):
    global current_task_create_labels
    global vocoder_model
    global vocoder_config
    global vocoder_in_scaler
    print(await request.body())
    body = await request.body()
    # args = str(urllib.parse.unquote(body)).split('')
    split_argument = split_arguments(str(urllib.parse.unquote(body)))
    # UTAU resampler引数の正しい順序:
    # <input_file> <output_file> <tone> <velocity> <flags> <offset> <length> <consonant> <cutoff> <volume> <modulation> <tempo> <pitchbends...>
    # args = [
    #     "C:\\Users\\XXXX\\あ.wav",  # input_path
    #     "./aaaa.wav",               # output_path
    #     "A4",                       # target_tone
    #     "107",                      # velocity (整数値)
    #     "",                         # flags
    #     "20",                       # offset (from "20@168+217.318")
    #     "217.318",                  # target_ms (length)
    #     "224.7583",                 # fixed_ms (consonant)
    #     "3197.114",                 # end_ms (cutoff)
    #     "300",                      # volume
    #     "0",                        # modulation (整数値)
    #     "!120",                     # tempo (デフォルト)
    #     ""                          # pitchbend
    # ]

    result = await run_in_threadpool(main_resampler,arg_list=split_argument,vocoder_model=vocoder_model,vocoder_config=vocoder_config,vocoder_in_scaler=vocoder_in_scaler)


# from hifisampler github:
def split_arguments(input_string: str):
    otherargs = input_string.split(' ')[-12:]
    file_path_strings = ' '.join(input_string.split(' ')[:-12])
    first_file, second_file = file_path_strings.split('.wav ')
    return [first_file + ".wav", second_file] + otherargs


async def has_current_task(current_task):
    if current_task is not None and not current_task.done():
        print("Cancelled previous task.")
        current_task.cancel()
        try:
            await current_task
        except asyncio.CancelledError:
            pass

if __name__ == '__main__':
    # サーバー起動
    uvicorn.run(app,host="0.0.0.0", port=55903, log_level="debug")
    logging.debug('sys.argv: %s', sys.argv)

@echo off
echo ============================== kuresampler ==================================
@REM echo %0
echo %*
set ROOT=%~dp0..\
set python_exe=python
@REM set python_exe=%ROOT%python-3.12.10-embed-amd64\python.exe
set python_script=%ROOT%resampler.py
set model_dir=%ROOT%models\usfGAN_EnunuKodoku_0826\
%python_exe% %python_script% %* --model_dir %model_dir% --use_vocoder_model --debug
echo =============================================================================

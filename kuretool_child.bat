@echo off
echo =============================== kuretool ====================================
@REM echo %0
echo %*
set selfdir=%~dp0
set python_exe=python
@REM set python_exe=%selfdir%python-3.12.10-embed-amd64\python.exe
set python_script=%selfdir%wavtool.py
set model_dir=%selfdir%models\usfGAN_Namineritsu_4130\
%python_exe% %python_script% %* --model_dir %model_dir% --use_vocoder_model --debug
echo =============================================================================

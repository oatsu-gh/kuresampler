@echo off
echo ============================== kuresampler ==================================
@REM echo %0
echo %*
set selfdir=%~dp0
set python_exe=python
set python_script=%selfdir%..\src\resampler.py
@REM set model_dir="%selfdir%..\\models\\usfGAN_EnunuKodoku_0826\\"
set model_dir="%selfdir%..\\models\\usfGAN_Namineritsu_4130\\"
%python_exe% "%python_script%" %* --model_dir %model_dir%
echo =============================================================================

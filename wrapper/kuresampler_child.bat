@echo off
echo ============================== kuresampler ==================================
echo %0
echo %*
set selfdir=%~dp0
set python_exe=python
set python_script=%selfdir%..\src\resampler.py
set model_dir="%selfdir%..\\models\\usfGAN_EnunuKodoku_0826\\"
%python_exe% "%python_script%" "%*" --model_dir %model_dir%
echo =============================================================================

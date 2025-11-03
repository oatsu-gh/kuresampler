@echo off

@REM If output file already exists, skip processing
@if exist %output% goto :eof

@echo =============================================================================
@echo                          Unified Kuresampler
@echo =============================================================================
@REM Batch filename
@REM @echo %0

@REM Arguments
@REM @echo %*

@REM Current directory
@set ROOT=%~dp0..\

@REM @echo %selfdir%
@set python_exe=python
@set python_script=%ROOT%unified_engine.py
@set model_dir=%ROOT%models\usfGAN_EnunuKodoku_0826\
@REM @set model_dir=%ROOT%models\usfGAN_NamineRitsu_4130\

@REM Run python script if output file does not exist
@REM @%python_exe% %python_script% %* --model_dir %model_dir% --use_vocoder_model
@%python_exe% %python_script% %* --model_dir %model_dir% --use_vocoder_model --debug
@REM @echo =============================================================================
PAUSE
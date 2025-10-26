@echo off

@REM If output file already exists, skip processing
@if exist %output% goto :eof

@echo =============================================================================
@echo                          Utau Engine Wrapper
@echo =============================================================================
@REM Batch filename
@REM @echo %0

@REM Arguments
@REM @echo %*

@REM Current directory
@set selfdir=%~dp0

@REM @echo %selfdir%
@set python_exe=python
@set python_script=%selfdir%uhacker.py

@REM Run python script if output file does not exist
@%python_exe% %python_script% %*
@REM @echo =============================================================================
PAUSE
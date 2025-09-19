@echo off
setlocal

echo Cleaning Python cache files and directories...
echo.

REM Delete all *.pyc files recursively
echo Deleting *.pyc files...
del /s /q "*.pyc" 2>nul && echo *.pyc files deleted successfully || echo No *.pyc files found

echo.
echo Deleting __pycache__ directories...

REM Delete all __pycache__ directories recursively
for /f "tokens=*" %%d in ('dir /s /b /ad "__pycache__" 2^>nul') do (
    rmdir /s /q "%%d" 2>nul
)
echo __pycache__ directories deleted successfully

echo.
echo Cleanup completed!
pause

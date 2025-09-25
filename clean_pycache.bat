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
echo Searching for __pycache__ directories...
for /f "tokens=*" %%d in ('dir /s /b /ad "__pycache__" 2^>nul') do (
    echo Deleting: %%d
    rmdir /s /q "%%d" 2>nul
    if exist "%%d" (
        echo Warning: Failed to delete %%d
    ) else (
        echo Successfully deleted: %%d
    )
)

REM Additional cleanup for any remaining __pycache__ directories
if exist "__pycache__" rmdir /s /q "__pycache__" 2>nul
for /d /r . %%d in (*__pycache__*) do if exist "%%d" rmdir /s /q "%%d" 2>nul

echo __pycache__ directories cleanup completed

echo.
echo Cleanup completed!
pause

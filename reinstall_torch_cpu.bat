@REM --------------------------------
@REM Install PyTorch CPU version
@REM --------------------------------
set python_dir=%~dp0python-3.12.10-embed-amd64\

@REM Uninstall existing torch packages
%python_dir%python.exe -m pip uninstall torch torchaudio torchvision -y
@REM Remove torch directories
rmdir /s /q %python_dir%Lib\site-packages\torch 2>NUL
rmdir /s /q %python_dir%Lib\site-packages\torchaudio 2>NUL
rmdir /s /q %python_dir%Lib\site-packages\torchvision 2>NUL
rmdir /s /q %python_dir%Lib\site-packages\~orch 2>NUL
rmdir /s /q %python_dir%Lib\site-packages\~orchaudio 2>NUL
rmdir /s /q %python_dir%Lib\site-packages\~orchvision 2>NUL
rmdir /s /q %python_dir%Lib\site-packages\~~orch 2>NUL
rmdir /s /q %python_dir%Lib\site-packages\~~orchaudio 2>NUL
rmdir /s /q %python_dir%Lib\site-packages\~~orchvision 2>NUL

@REM Install CPU versions of torch packages
%python_dir%python.exe -m pip install torch torchaudio torchvision --no-warn-script-location

PAUSE

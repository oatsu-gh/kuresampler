@REM Install PyTorch CPU version
python-3.12.10-embed-amd64\python.exe -m pip uninstall torch torchaudio torchvision -y
python-3.12.10-embed-amd64\python.exe -m pip install torch torchaudio torchvision
PAUSE

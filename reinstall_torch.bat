python-3.12.10-embed-amd64\python.exe -m pip uninstall torch torchaudio torchvision -y
python-3.12.10-embed-amd64\python.exe -m pip install --upgrade light-the-torch --no-warn-script-location
python-3.12.10-embed-amd64\python.exe -m light_the_torch install torch torchaudio torchvision --no-warn-script-location
PAUSE

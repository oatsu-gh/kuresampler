python-3.12.10-embed-amd64\python.exe -m pip uninstall fastapi uvicorn -y
python-3.12.10-embed-amd64\python.exe -m pip install --upgrade fastapi uvicorn --no-warn-script-location
@REM python-3.12.10-embed-amd64\python.exe -m light_the_torch install torch torchaudio torchvision --no-warn-script-location
PAUSE
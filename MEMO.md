## 開発メモ

#### nnsvs.gen.predict_waveform

各ボコーダで WAV 生成するのに必要な特徴量

| vocoder_type | feature_type |  mgc  |  lf0  |  vuv  |  bap  |  mel  |  f0   | spectrogram | aperiodicity | 備考                                   |
| :----------: | :----------: | :---: | :---: | :---: | :---: | :---: | :---: | :---------: | :----------: | :------------------------------------- |
|  **world**   |  **world**   |   ○   |   ○   |   ○   |   ○   |   ×   |   ×   |      ×      |      ×       | 4 stream                               |
|    world     |  world_org   |   ×   |   ×   |   ×   |   ×   |   ×   |   ○   |      ○      |      ○       | 3 stream                               |
|    world     |   neutrino   |   ○   |   ×   |   ×   |   ○   |   ×   |   ○   |      ×      |      ×       | f0→lf0, vuvを内部で生成                |
|    world     |    melf0     |   ×   |   ○   |   ○   |   ×   |   ○   |   ×   |      ×      |      ×       | 3 stream                               |
|   **pwg**    |  **world**   |   ○   |   ○   |   ○   |   ○   |   ×   |   ×   |      ×      |      ×       | 4 stream                               |
|     pwg      |    melf0     |   ×   |   ○   |   ○   |   ×   |   ○   |   ×   |      ×      |      ×       | 3 stream                               |
|    usfgan    |    world     |   ○   |   ○   |   ○   |   ○   |   ×   |   ×   |      ×      |      ×       | bapの次元数でmcep/aperiodicity分岐あり |
|    usfgan    |   neutrino   |   ○   |   ×   |   ×   |   ○   |   ×   |   ○   |      ×      |      ×       | bapの次元数でmcep/aperiodicity分岐あり |
|    usfgan    |    melf0     |   ×   |   ○   |   ○   |   ×   |   ○   |   ×   |      ×      |      ×       | 3つのstream                            |
|    usfgan    |  world_org   |   ×   |   ×   |   ×   |   ×   |   ×   |   ×   |      ×      |      ×       | 未実装（NotImplementedError）          |


#### 特徴量フォーマットの相互変換

```txt
aperiodicity -----------------------------------> bap 
               pyworld.code_aperiodicity
               pyworld.decode_aperiodicity
aperiodicity <----------------------------------- bap 

spectrogram  -----------------------------------> mgc
               pyworld.code_spectral_envelope
               pyworld.decode_spectral_envelope
spectrogram  <----------------------------------- mgc

# lf0 はモデルに渡す前に前処理必要
f0           -----------------------------------> lf0
                lf0 = np.zeros_like(f0)
                nonzero_indices = np.nonzero(f0)
                lf0[nonzero_indices] = np.log(f0[nonzero_indices])

                                          exp(f0)
f0           <----------------------------------- lf0

f0           -----------------------------------> vuv
                vuv = (f0 > 0).astype(np.float32)
```

## 環境構築メモ

### python-3.12.10-embed-amd64 での環境構築手順

- pysptk をインストールするときに dll とか h ファイルとかが必要なので、インストール版の Python からコピーする。(2025/04/09)
  - python/include/  → python-embeddable/include/
  - python/libs/  → python-embeddable/libs/
- pysptk をインストールするときと uSFGAN を使うときに tcl/tk が必要なので、インストール版の Python から下記内容でコピーして対処。(2025/04/09)
  - python/tcl/  → python-embeddable/tcl/
  - python/Lib/tkinter/ → python/tkinter/
  - python/DLLs/\_tkinter.pyd → python-embeddable/\_tkinter.pyd
  - python/DLLs/tcl86t.dll → python-embeddable/tcl86t.dll
  - python/DLLs/tk86t.dll→ python-embeddable/tk86t.dll
  - python/DLLs/zlib1.dll → python-embeddable/zlib1.dll

- ```python
  python-3.12.10-embed-amd64\get-pip.py --no-warn-script-location
  python-3.12.10-embed-amd64\python.exe -m pip install -r requirements.txt --no-warn-script-location
  python-3.12.10-embed-amd64\python.exe -m pip install nnsvs
  python-3.12.10-embed-amd64\python.exe -m pip uninstall nnsvs torch torchaudio torchvision -y
  ```

- nnsvs をダウンロード (https://github.com/nnsvs/nnsvs)
- uSFGAN は HN-UnifiedSourceFilterGAN-nnsvs を DL
- ParallelWaveGAN は ParallelWaveGAN-nnsvs をDL (https://github.com/nnsvs/ParallelWaveGAN)
- SiFiGAN は SiFiGAN-nnsvs を DL (https://github.com/nnsvs/SiFiGAN)
  - pip で python embeddable に SiFiGAN をインストールする場合は docopt が無いとエラーが出るので、インストール版の Python から docopt をコピーして対処。(2024/05/19)

# kuresampler
WORLD と ニューラルボコーダーを利用した UTAU エンジンです。自然なクロスフェードと高品質な出音の両立を目指します。


## 仕様調査

### nnsvs.gen.predict_waveform
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


### 特徴量フォーマットの相互変換

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

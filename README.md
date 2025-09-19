# kuresampler
WORLD と ニューラルボコーダーを利用した UTAU エンジンです。自然なクロスフェードと高品質な出音の両立を目指します。

## 特徴

- resampler のみで使用する場合
  - 長所①：パラメトリック歌声合成のわりに自然な声が得られる。
  - 短所①：一般的なUTAUエンジンに比べてレンダリングが遅い。
  - 短所②：CUDA対応グラフィックボードまたは強力なCPUが必要。
- resampler と wavtool 両方で使用する場合
  - 長所①：パラメトリック音声合成のわりに自然な声が得られる。
  - 長所②：原音の音階が異なるクロスフェードでも、比較的自然な合成ができる。
  - 短所①：一般的なUTAUエンジンに比べてレンダリングが遅い。
  - 短所②：CUDA対応グラフィックボードまたは強力なCPUが必要。

## How to use

UTAU エンジンの resampler の代わりに **kuresampler.exe** を指定してください。OpenUtau ではなく通常の UTAU を使う場合は **kuresampler_fast.exe** のほうが高速なのでおすすめです。

### Flags

デフォルトの伸縮方法はストレッチ式 ( `e` ) です。ループ式に変更したい場合はループフラグ ( `l` ) を使用してください。

| flag     | range         | default        | description                                                                                                                                                                           |
| -------- | ------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| B        | 0 ～ 100      | default:50     | 息成分の強さ（ブレシネス）。大きいほど息っぽい。0～49では B0 の時非周期性指標が全て0になるように乗算。51～100では B100 の時、1000Hz～5000Hz 帯の非周期性指標が全て1になるように加算。 |
| eb       | 0 ～ 100      | default:0      | 語尾の息成分の強さ。大きいほど息っぽい。                                                                                                                                              |
| ebs      | -1000 ～ 1000 | default:0      | ノート前半部分の語尾息がかからない時間を5ms単位で指定。負の数を指定するとノート末尾からの時間。                                                                                       |
| eba      | 0 ～ 1000     | default:0      | ebフラグのアタックタイムを5ms単位で指定。                                                                                                                                             |
| g        | -100 ～ 100   | default:0      | 疑似ジェンダー値。負で女声化・若年化、正で男声化・大人化。                                                                                                                            |
| t        | -100 ～ 100   | default:0      | 音程の補正。1cent単位。                                                                                                                                                               |
| P        | 0 ～ 100      | default:86     | ピークコンプレッサー。P100 の時 volume 適用前の音量最大値が -6dB になるよう正規化。P0 の時は無効。                                                                                    |
| e        | -             | default: True  | wav の伸縮方法。通常はループ方式で、このフラグを設定するとストレッチ式になる。                                                                                                        |
| l (エル) | -             | default: False | wav の伸縮方法をループ式にする。                                                                                                                                                      |
| A        | -100 ～ 100   | default:0      | ピッチ変動にあわせて音量が変化。1～100では基準より高いとき音量が小さく、-1～-100では基準より低いとき音量が小さくなる。                                                                |
| gw       | 0 ～ 500      | default:0      | うなり声（グロウル）。                                                                                                                                                                |
| gws      | -1000 ～ 1000 | default:0      | ノート前半部分のグロウルがかからない時間を5ms単位で指定。負の数を指定するとノート末尾からの時間。                                                                                     |
| gwa      | 0 ～ 1000     | default:0      | gwフラグのアタックタイムを5ms単位で指定。                                                                                                                                             |
| vf       | -500 ～ 500   | default:0      | 疑似エッジ。エッジがかかる長さを5ms単位で指定。                                                                                                                                       |
| vfw      | 0 ～ 300      | default:100    | 疑似エッジの1回あたりの長さ（%指定）。                                                                                                                                                |
| vfp      | 0 ～ 100      | default:20     | 疑似エッジの1回あたりの無音の長さ（%指定）。                                                                                                                                          |

## How to make your vocoder model

wavファイルだけ用意してnnsvs


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

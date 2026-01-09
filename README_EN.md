# kuresampler
WORLD is a UTAU engine utilizing a neural vocoder. It aims to achieve both natural crossfades and high-quality sound output.

[JA (Original) README](https://github.com/oatsu-gh/kuresampler/blob/main/README.md)

## Usage
### Server/Client Mode (Recommended)
- Added in kuresampler v0.1.0. Higher-speed server-client architecture
- **Compatible with UTAU & OpenUtau**
- Used as a resampler with standard wavtool.
- Specify one of the following as the resampler:
  - `kuresampler_K_Client.exe`
  - `kuresampler_R_Client.exe`
- On CUDA-enabled systems, running `reinstall_torch.bat` enables GPU-accelerated rendering.

### Standalone Mode (Deprecated)
- Operates identically to kuresampler v0.0.1 and earlier.
- Not compatible with OpenUtau.
- Used as a resampler with standard wavtool.
- Specify one of the following as the resampler:
  - `kuresampler_K.exe`
  - `kuresampler_R.exe`
  - `kuresampler_fast_K.exe`
  - `kuresampler_fast_R.exe`
- Engines with "fast" are faster but more resource-intensive.
- On CUDA-enabled systems, running `reinstall_torch.bat` enables GPU-accelerated rendering.

## Engine Comparison

| Engine              | Vocoder Model           | Description                                                                                                   | Info                                                                                                                  | Training Dataset                                  | License/Terms                                                          | Requires specification? |
| ------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------- |
| `kuresampler_K.exe` | usfGAN_EnunuKodoku_0826 | Trained on the "ENUNU蠱毒企画 歌唱データベース" ("ENUNU Gu-Doku Project Vocal Database").                          | Suitable for a wide range of voice qualities, including both male and female voices across low to high registers.     | https://github.com/oatsu-gh/enunu_kodoku_singing  |https://github.com/oatsu-gh/enunu_kodoku_singing/blob/main/README.md    |NO                       |
| `kuresampler_R.exe` | usfGAN_NamineRitsu#4310 | Used in "NNSVS/ENUNU 波音リツ #4139 CRISSCROSS 5スタイル" ("NNSVS/ENUNU Namine Ritsu #4139 CRISSCROSS 5 Styles"). | Suitable for female voices in the low-mid to high range. Provides consistent vocal quality regardless of the pitch range. | private                                           | https://www.canon-voice.com/terms/                                     | NO                      |


```
 　　　＿＿＿
 　 .//⌒＿＿＿ ＼　　+　　　　。　　　　　+　　　　。　　　　　＊　 　　　。
 　//_／　　 ＼＼　＼ 　　　　+　　　　。　　　　　+　　　　。　　　　　＊　 　　　。
　　　　　　 　 ＼＼　＼
　　＊　　　 +　 ((　　|　　　 イヤッッホォォォオオォオウ！
　　　　　　 　　 |　　∩
　　　+　　　。　 | 　| |　＊　 　　　+　　　　。　　　　　+　　　。　+
　　　　　　　 　 | 　| |
　　　　　　　　  | 　/ |  ._　 +　　　　。　　　　　+　　　　+　　　　　＊
　　　　　＼￣￣ ~/　 　/~￣.＼
　 　　　 ||＼ 　~^~^~^~　 　　＼　　　　+　　　　。　　　　　+　　　　+　　　　　＊
　 　　 　||＼||￣￣￣￣￣￣￣||￣　　　　　　　　　　　　　　それがＶＩＰクオリティ
　 　　 　||　||￣￣￣￣￣￣￣||　　　　　　　　　　　　　https://hebi.5ch.net/news4vip/

```


## Features

### Pros & Cons

- When used as resampler
  - Pro: Produces natural-sounding voice for parametric vocal synthesis.
  - Con: Slow rendering compared to typical UTAU engines.
- When used as resampler & wavtool (not yet implemented)
  - Pro: Produces natural-sounding voice for parametric vocal synthesis.
  - Pro: Relatively natural synthesis even with crossfades where the original pitch is different.
  - Con: Slow rendering compared to typical UTAU engines.
  
### How to make a vocoder model

- You can create your own uSFGAN vocoder model using nnsvs if you prepare wav files.

### Flags

- Flags other than `B`, `g`, `e`, & `l` have not yet been tested
- The default stretching method is stretch mode (`e`). To change to loop mode, use the loop flag (`l`).

| Flag     | Value Range  | Default        | Description                                                                                                                                                                           |
| -------- | ------------ | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `B`      | 0 - 100      | Default: 50    | Breathiness: Higher values produce a breathier sound. From 0 to 49: multiplies values so that all non-periodicity indices become 0 at B0. From 51 to 100: adds values so that all non-periodicity indices in the 1000Hz to 5000Hz band become 1 at B100 |
| `eb`     | 0 - 100      | Default: 0     | Ending Breathiness: Strength of the breath component at the ending. Higher values produce a more breathy sound.                                                                                                                                              |
| `ebs`    | -1000 - 1000 | Default: 0     | Specifies the start time in 5ms increments relative to the note's initial attack phase. Negative values indicate time measured from the note's end                                                     |
| `eba`    | 0 to 1000    | Default: 0     | Specifies the attack time for the eb flag in 5ms increments                                                                                                                                             |
| `g`      | -100 - 100   | Default: 0     | Gender Factor: Negative values make the voice more "feminine", positive values make the voice more "masculine"                                                                                                                            |
| `t`      | -100 to 100  | Default: 0     | Pitch correction: 1-cent increments                                                                                                                                                               |
| `P`      | 0 - 100      | Default: 86    | Normalize (Peak compressor): At P100, normalizes the maximum volume before volume application to -6dB (Disabled at P0)                                                                                   |
| `e`      | boolean      | Default: True  | Force Stretch: Wav stretching method. Normally uses looping mode; setting this flag switches to stretching mode                                                                                                        |
| `l` (L)  | Boolean      | Default: False | Force Loop: Sets the wav scaling method to loop                                                                                                                                                      |
| `A`      | -100 - 100   | Default: 0     | Pitch-following volume adjustment: From 1 to 100, volume decreases when pitch is higher than the reference pitch. From -1 to -100, volume decreases when pitch is lower than the reference pitch.                                                                |
| `gw`     | 0 - 500      | Default: 0     | Growl                                                                                                                                                                |
| `gws`    | -1000 - 1000 | Default: 0     | Growl Delay: Specifies the start time in milliseconds for when the growl effect begins. Negative values indicate time measured from the end of the note.                                                                                     |
| `gwa`    | 0 - 1000     | Default: 0     | Growl Attack Time: Specifies the attack time for the `gw` flag in 5ms increments.                                                                                                                                             |
| `vf`     | -500 - 500   | Default: 0     | Pseudo-vocal-fry: Specifies the vocal fry duration in 5ms increments.                                                                                                                                       |
| `vfw`    | 0 - 300      | Default: 100   | Pseudo-Vocal-Fry Length: Length of pseudo-vocal-fry (percentage).                                                                                                                                                |
| `vfp`    | 0 - 100      | Default: 20    | Pseudo-Vocal-Fry Silence Length: Silent length per pseudo-edge (percentage).                                                                                                                                          |


## Special Thanks

LEIRH (https://x.com/LEIRHds)

## Update History

### 0.0.1-alpha

- Initial release

### 0.0.1

- Fixed an issue where voices sounded too young
- Fixed an issue where WAV synthesis failed when applying the `g` flag or `B` flag (modified the included PyRwu)
- Changed the formant shift scale for the `g` flag from log to mel (modified included PyRwu)
- Added `reinstall_torch.bat`

### 0.1.0

- Added server & client mode using FastAPI. This speeds up rendering.
  - Specify `kuresampler_K_Client.exe` / `kuresampler_R_Client.exe` as the resampler.
- Added OpenUtau support

Translated with [DeepL](https://www.deepl.com/en/translator)

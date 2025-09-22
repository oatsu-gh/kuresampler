# フォルマントシフト修正について

## 問題の概要

kuresamplerで生成された音声が女声寄りに聞こえる（フォルマントシフトしている）問題を修正しました。

## 修正内容

### 修正されたファイル
- `convert.py`: `nnsvs_to_world()`関数を修正

### 修正の詳細

**修正前:**
```python
def nnsvs_to_world(mgc, lf0, vuv, bap, sample_rate, fft_size=DEFAULT_FFT_SIZE):
    # 固定のFFTサイズ（512）を使用してスペクトラムエンベロープを復元
    spectrogram = pyworld.decode_spectral_envelope(mgc, sample_rate, fft_size)
```

**修正後:**
```python
def nnsvs_to_world(mgc, lf0, vuv, bap, sample_rate, fft_size=None):
    # BAPの次元数から適切なFFTサイズを自動計算
    if fft_size is None:
        n_bands = bap.shape[-1]
        fft_size = (n_bands - 1) * 2
    
    spectrogram = pyworld.decode_spectral_envelope(mgc, sample_rate, fft_size)
```

## 修正の効果

1. **フォルマント周波数の保持**: エンコード時とデコード時で一貫したスペクトラムエンベロープサイズを使用することで、フォルマント周波数の歪みを防止

2. **音声の性別感の維持**: 適切なフォルマント構造により、元の音声の性別特性が保たれる

3. **下位互換性**: 既存のコードは修正なしで動作し、必要に応じて明示的なFFTサイズ指定も可能

## テスト方法

修正の効果を確認するには:

1. **主観評価**: 同じ音声素材で修正前後のkuresamplerを比較
2. **客観評価**: フォルマント周波数の測定・比較
3. **自動テスト**: `test_formant_fix.py`で修正ロジックの動作確認

## 技術的詳細

### 原因
- MGCエンコード時のスペクトラムエンベロープサイズとデコード時のFFTサイズの不一致
- 固定FFTサイズ（512）による周波数分解能の劣化

### 解決方法
- BAPの次元数（`n_bands`）からFFTサイズを逆算: `fft_size = (n_bands - 1) * 2`
- これにより元のスペクトラムエンベロープサイズと一致

### 影響範囲
- `nnsvs_to_world()`関数の呼び出し元すべて
- 既存のAPIは変更なしで動作（下位互換性確保）
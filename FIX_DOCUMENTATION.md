# NeuralNetworkResamp.ap Zero Issue - Fix Documentation

## 問題概要

B フラグや g フラグを変更したときに、`resampler._synthesize_with_vocoder_model` の前の時点で `self.ap` がすべて 0 になってしまう問題。

## 根本原因

`NeuralNetworkResamp.synthesize()` メソッドで PyRwu の effects が適用される際、AP_EFFECTS や WORLD_EFFECTS の処理によって `self._ap` が意図せず全て 0 になってしまうケースがあった。

## 実装した修正

### 1. デバッグとモニタリングの追加

**`synthesize()` メソッド:**
- effects 適用前後での ap 値の監視
- AP_EFFECTS と WORLD_EFFECTS で ap が全て 0 になった場合の検出
- 詳細なログ出力

**`resamp()` メソッド:**
- 各処理ステップでの ap 状態監視
- parseFlags, getInputData, stretch, pitchShift, applyPitch の各段階でのチェック

### 2. B0 フラグの正確な検出

B フラグが 0 の場合は意図的に ap=0 にする仕様のため、以下の正規表現を使用：

```python
b_zero_pattern = r'(?:^|[^0-9])B0(?:[^0-9]|$)'
```

これにより以下を正確に区別：
- `B0` → B0として検出 ✓
- `B0,g50` → B0として検出 ✓  
- `B01` → B0として検出しない ✓
- `B10` → B0として検出しない ✓

### 3. ap 値の復元メカニズム

```python
if self._ap is not None and np.all(self._ap == 0):
    # B0 フラグの場合は意図的な動作なので復元しない
    if not is_b_zero_intended and ap_before_this_effect is not None:
        self.logger.warning('Restoring previous ap values (not B0 case)')
        self._ap = ap_before_this_effect
```

## テスト結果

### B フラグ検出テスト
すべてのテストケースがパス：
- 空文字列、B0、B0,g50、g50,B0、B0g50
- B10、B50、B1、B01（これらは B0 として検出されない）
- eB0、B0e（これらは B0 として検出される）

### ap 復元条件テスト
すべてのシナリオで正しい動作を確認：
- 通常ケース：ap が予期せず 0 になった場合は復元
- B0 ケース：意図的な ap=0 は復元しない
- 既に 0 だった場合：復元しない

## ログ出力例

```
DEBUG: Before effects - ap shape: (100, 1025), nonzero count: 102500
WARNING: AP effect BreakinessEffect set all ap values to zero! Effect: <BreakinessEffect B50>
WARNING: Restoring previous ap values (not B0 case)
DEBUG: After effects - ap shape: (100, 1025), nonzero count: 102500
```

## 影響範囲

- `NeuralNetworkResamp` クラスのみ
- 既存の正常動作（B0 フラグでの意図的 ap=0）は維持
- 予期しない ap=0 の場合のみ復元処理が動作
- パフォーマンスへの影響は最小限（デバッグログのみ）

## 今後の課題

1. PyRwu の effects で ap が 0 になる具体的な条件の特定
2. 根本的な修正（effects 側での修正）
3. より詳細なテストケースの追加

この修正により、B フラグや g フラグ使用時の ap=0 問題が解決され、音質の劣化を防ぐことができます。
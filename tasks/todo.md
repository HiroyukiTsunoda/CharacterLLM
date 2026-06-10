# フルリファクタリング＋モデル更新 (2026-06-11)

計画: `C:\Users\HT\.claude\plans\delightful-doodling-mochi.md`

## タスク

- [x] 1. InferenceProvider Protocol 導入（src/core/provider_protocol.py 新規）
- [x] 2. OpenAI GPT-5.x 移行（openai_provider.py / main.py / config.json）
- [x] 3a. _filter_think_stream のキャラクタライゼーションテスト作成（tests/test_think_filter.py、分割前に通す）
- [x] 3b. chat_engine.py:414 バグ修正（model_loader → active_provider）
- [x] 3c. _filter_think_stream をフェーズ別メソッドに分割（テストで等価性確認）
- [x] 4. モデルカタログJSON外部化＋Qwen3.6追加・旧モデル削除（HF APIでファイル名検証済み）
- [x] 5. UI層修正（config保存統一 / torch同期のコア移動 / Whisper large-v3-turbo追加）
- [x] 6. コア小修正（import re / redirect_stdout / load_config副作用 / 履歴トリム / voice_inputロック）
- [x] 7. 死コード削除（src/data/）＋README・docs更新
- [x] 検証: インポートチェック・unittest・アプリ起動スモークテスト

## レビュー

### 実施内容
- **OpenAI移行（壊れていた機能の修復）**: gpt-4o系（全廃止済み）→ GPT-5.4-mini/5.4/5.5/5.4-nano。
  推論モデル対応で `max_completion_tokens`（+1024バッファ）/`reasoning_effort`（config可変、デフォルトlow）に移行。
- **バグ修正**: OpenAI使用時にローカルモデルの `template_inserts_think` を参照していた問題
  （chat_engine.py の `_filter_think_stream`）。
- **構造改善**: InferenceProvider Protocol 導入、270行ステートマシンを5つのフェーズメソッドに分割、
  モデルカタログを `resources/recommended_models.json` に外部化（Qwen3.6世代を追加、2024年世代を削除）、
  config保存4重複を `_save_config()` に統一、GPU同期をUI層からコア層へ移動。
- **死コード削除**: src/data/（SQLite、未使用）と chat_engine の関連メソッドを削除、README/docs の記載を実態に修正。

### 検証結果
- ユニットテスト 10/10 合格（分割前にベースライン確立→分割後も同一テスト合格＝挙動等価）
- 全モジュールのインポート・コンパイルチェック OK
- アプリ起動スモークテスト OK（12秒稼働、ログエラーなし）
- 未実施: OpenAI実APIへの接続テスト（課金が発生するためUIの「テスト」ボタンでの確認を推奨）

### 残課題（今回のスコープ外）
- Qwen3TTSEngine のモデルキャッシュ二重管理（_model_cache / _char_model_map）の整理
- チャット履歴の永続化が必要になった場合は新規設計（旧database.pyは削除済み）

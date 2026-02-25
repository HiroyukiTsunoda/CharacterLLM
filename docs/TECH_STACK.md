# CharacterLLM — 技術スタック詳細

## 目次

1. [アーキテクチャ概要](#アーキテクチャ概要)
2. [コアランタイム](#コアランタイム)
3. [LLM推論エンジン](#llm推論エンジン)
4. [音声処理](#音声処理)
5. [UIフレームワーク](#uiフレームワーク)
6. [データ永続化](#データ永続化)
7. [依存パッケージ一覧](#依存パッケージ一覧)
8. [設計思想と内部構造](#設計思想と内部構造)

---

## アーキテクチャ概要

```
┌──────────────────────────────────────────────────┐
│                   PySide6 UI                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ Character│ │  Model   │ │   Chat Widget    │  │
│  │  Panel   │ │  Panel   │ │  (Streaming)     │  │
│  └────┬─────┘ └────┬─────┘ └────────┬─────────┘  │
│       │             │                │            │
├───────┴─────────────┴────────────────┴────────────┤
│                   Core Layer                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │Character │ │  Model   │ │   Chat Engine    │  │
│  │ Manager  │ │  Loader  │ │ (Think Filter)   │  │
│  └────┬─────┘ └────┬─────┘ └────────┬─────────┘  │
│       │             │                │            │
│  ┌────┴─────┐ ┌─────┴──────┐ ┌──────┴─────────┐  │
│  │Voice I/O │ │ llama-cpp  │ │    SQLite DB   │  │
│  │Whisper+  │ │  -python   │ │  (chat_history) │  │
│  │SBV2 TTS  │ │  (CUDA)    │ │                │  │
│  └──────────┘ └────────────┘ └────────────────┘  │
└──────────────────────────────────────────────────┘
         ↕              ↕              ↕
   [ マイク/スピーカー ] [ NVIDIA GPU ]  [ ディスク ]
```

アプリケーションは3層構造で構成されています。

| レイヤー | 責務 | 主なモジュール |
|---------|------|---------------|
| **UI層** | ユーザーインタラクション、表示 | `src/ui/` |
| **Core層** | ビジネスロジック、推論、音声処理 | `src/core/` |
| **Data層** | 永続化、ファイルI/O | `src/data/` |

---

## コアランタイム

### Python 3.12

| 項目 | 詳細 |
|------|------|
| バージョン | 3.11+（3.12 推奨） |
| 仮想環境 | `venv` 標準モジュール |
| 型ヒント | `from __future__ import annotations` による遅延評価を全面採用 |

Python 3.12 を推奨する理由:
- llama-cpp-python の CUDA ビルド済みホイールが cp312 向けに提供されている
- パフォーマンス改善（CPython 3.12 のインタプリタ高速化）

### PyTorch (CUDA)

| 項目 | 詳細 |
|------|------|
| 用途 | TTS (Style-Bert-VITS2) 推論、BERTモデル推論 |
| CUDA版 | cu124 (CUDA 12.4) |
| インストール | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |

LLM推論は llama-cpp-python (ggml) が担当するため、PyTorch は TTS 関連でのみ使用されます。同一GPU上で llama-cpp-python と PyTorch が cuBLAS ハンドルを共有することによる競合を、推論前の `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` で回避しています。

---

## LLM推論エンジン

### llama-cpp-python

| 項目 | 詳細 |
|------|------|
| バージョン | 0.3.4 |
| バックエンド | ggml (C/C++) + CUDA カーネル |
| モデル形式 | GGUF (量子化モデル) |
| GPU対応 | CUDA 12.4、全レイヤーGPUオフロード |

#### 主な機能

- **モデル読み込み**: GGUFファイルからの直接読み込み、GPU層数指定
- **ストリーミング生成**: `create_chat_completion(stream=True)` によるトークン単位の逐次出力
- **KVキャッシュ管理**: 生成間のステート自動リセット（Mamba2ハイブリッドモデル対応）
- **VRAM推定**: ファイルサイズから必要VRAM量を概算

#### GPU設定

```python
Llama(
    model_path=path,
    n_gpu_layers=-1,     # 全レイヤーをGPUへ
    main_gpu=0,          # プライマリGPU
    n_ctx=4096,          # コンテキスト長
    n_batch=512,         # バッチサイズ
    verbose=False,
)
```

#### HuggingFace連携

`huggingface-hub` を使用して、おすすめモデルの一覧表示とワンクリックダウンロードを実装しています。ダウンロードしたファイルは `models/` ディレクトリに保存されます。

### 思考モデル対応

複数の「思考（Thinking）」フォーマットに対応しています。

| モデル系列 | フォーマット | 処理方法 |
|-----------|------------|---------|
| Qwen3 | `<think>...</think>` | タグ検出・除去 |
| Qwen3 (テンプレート自動挿入) | 暗黙的思考 → `</think>` | 先頭テキストを思考として抽出 |
| Nemotron | テンプレート `<think>` + 暗黙テキスト | `</think>` マーカーで分離 |
| GPT-OSS | `<\|channel\|>analysis` / `final` | チャネルタグで分離 |
| 汎用 | `思考内容` / `応答内容` セクション | キーワードベース検出 |

ストリーミング中は `THINK_STREAM_START` / `THINK_STREAM_END` マーカーでUI側に思考セクションを通知し、リアルタイムで思考と応答を分離表示します。

### 動的トークン管理

ユーザー入力の内容と長さを分析し、`max_tokens` を動的に決定します。

```
詳細要求（「教えて」「詳しく」等） → 768トークン
質問・相談                         → 512トークン
通常会話                           → 256トークン
短い挨拶                           → 128トークン
```

思考モデルの場合、上記に加えて2048トークンの思考予算が加算されます。さらに、応答長に応じたガイダンスをシステムプロンプトに付加し、モデルが適切な応答長を生成するよう誘導します。

---

## 音声処理

### 音声入力: faster-whisper

| 項目 | 詳細 |
|------|------|
| ライブラリ | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| バックエンド | CTranslate2 (Whisper の高速推論) |
| モデルサイズ | small / medium / large |
| GPU対応 | CUDA (float16) |

#### 処理フロー

```
マイク → sounddevice → numpy配列 → faster-whisper → テキスト
       (16kHz, mono, float32)      (VADフィルタ付き)
```

- **サンプリングレート**: 16,000 Hz
- **チャンネル数**: 1 (モノラル)
- **VADフィルタ**: 500ms 以上の無音区間で自動分割
- **ビームサーチ**: beam_size=5

### 音声出力: Style-Bert-VITS2

| 項目 | 詳細 |
|------|------|
| ライブラリ | [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) |
| BERTモデル | ku-nlp/deberta-v2-large-japanese-char-wwm |
| 出力形式 | int16 PCM (44.1kHz) |
| GPU対応 | CUDA |

#### 処理フロー

```
テキスト → クリーニング → ト書き解析 → 文分割 → BERT埋め込み → TTS推論 → sounddevice
         (Markdown除去等)  (感情抽出)  (100文字単位)             (GPU)     (再生)
```

#### テキストクリーニング

TTS に渡す前に以下の処理を行います。

1. Markdownコードブロック・インラインコードの除去
2. Markdown書式（太字、見出し、リンク等）の除去
3. URL の除去
4. 日本語文字・基本ASCIIのみ保持
5. 連続空白の正規化

#### ト書き（演出指示）システム

括弧内のキーワードを正規表現で検出し、TTS パラメータを動的にオーバーライドします。

```python
_STAGE_DIR_RE = re.compile(r'[（(]([^）)]+)[）)]')
```

プリセット例:
- **小声系**: noise=0.3, noise_w=0.4, length=1.2（静かでゆっくり）
- **叫び系**: style=Angry, noise=0.9, length=0.85（力強く速い）
- **感情系**: style=Happy/Sad/Angry 等、style_weight=7.0〜8.0

#### モデルキャッシュ

キャラクターIDをキーに TTSModel インスタンスをメモリにキャッシュし、キャラクター切替時の再読み込みを回避します。GPU設定変更時はキャッシュを自動クリアします。

#### dtype問題の対策

Style-Bert-VITS2 の safetensors モデルが float16 で保存されている場合、float32 のバイアスとの dtype 不一致で CUDA assert が発生します。対策として:

- BERTモデルを `float()` で明示的に float32 に変換
- TTS ネットワーク (`net_g`) も float32 に変換

### オーディオデバイス管理

`sounddevice` ライブラリ（PortAudio ラッパー）を使用して、入力/出力デバイスの列挙・選択を行います。

---

## UIフレームワーク

### PySide6

| 項目 | 詳細 |
|------|------|
| バージョン | 6.8.3 |
| バインディング | Qt 6 公式 Python バインディング |
| ライセンス | LGPL |

#### UIコンポーネント

| モジュール | 責務 |
|-----------|------|
| `main_window.py` | メインウィンドウレイアウト（サイドバー + チャットエリアのスプリッター） |
| `chat_widget.py` | チャットメッセージ表示、ストリーミング応答のリアルタイム更新 |
| `character_panel.py` | キャラクター一覧表示・選択 |
| `model_panel.py` | ローカルモデル一覧、HuggingFaceダウンロード、モデル読み込み |
| `voice_panel.py` | 音声入力/出力設定（デバイス選択、モデル選択） |
| `voice_button.py` | 音声入力ボタン（録音開始/停止のトグル） |
| `settings_dialog.py` | 設定ダイアログ |
| `styles.py` | QSS (Qt Style Sheets) によるテーマ定義 |

#### ダークテーマ

QSS (Qt Style Sheets) でカスタムダークテーマを定義しています。

**カラーパレット**:

| 用途 | カラーコード | 説明 |
|------|------------|------|
| 背景（暗） | `#1a1b2e` | メインウィンドウ背景 |
| 背景（中） | `#232540` | パネル背景 |
| 背景（明） | `#2d2f52` | カード・バブル背景 |
| アクセント | `#7c6ff7` | ボタン、選択、ユーザーバブル |
| アクセント（ホバー） | `#9a8fff` | インタラクション時 |
| テキスト（主） | `#e8e8f0` | 本文テキスト |
| テキスト（副） | `#a0a0b8` | 補助テキスト |
| ボーダー | `#3a3c60` | 区切り線 |
| 成功 | `#4caf50` | 成功状態 |
| 警告 | `#ff9800` | 警告状態 |
| エラー | `#f44336` | エラー状態 |

**チャットバブル**:
- ユーザーメッセージ: アクセントカラー (`#7c6ff7`) 背景
- アシスタントメッセージ: 中間背景 (`#2d2f52`)

---

## データ永続化

### SQLite

| 項目 | 詳細 |
|------|------|
| ライブラリ | Python 標準 `sqlite3` |
| データベースパス | `data/chat_history.db` |

#### 管理対象

- **セッション**: 会話セッション（作成日時、キャラクターID）
- **メッセージ**: 各メッセージ（ロール、本文、キャラクターID、思考内容）

#### 履歴管理

- メモリ上の履歴は最大50メッセージ（system メッセージ除く）
- 古いメッセージは自動的に先頭から削除（FIFO）
- セッション切替時にDBへ永続化

### キャラクター定義

JSON ファイルベース（`characters/` ディレクトリ）。

### アプリ設定

`config.json`（アプリルート直下）。起動時に読み込み、存在しない場合はデフォルト値で自動生成。

### ログ

| 項目 | 詳細 |
|------|------|
| 出力先 | `logs/` ディレクトリ |
| ローテーション | 24時間で自動ローテーション |
| レベル | コンソール: INFO、ファイル: DEBUG |

2種類のログを出力:
- **アプリケーションログ**: 通常の動作ログ
- **LLM Raw Output**: モデルの生出力テキスト（思考ブロック含む、デバッグ用）

---

## 依存パッケージ一覧

### requirements.txt 管理

| パッケージ | バージョン制約 | 用途 |
|-----------|--------------|------|
| PySide6 | `==6.8.3` | GUIフレームワーク |
| numpy | `<2` | 数値演算・音声データ操作 |
| huggingface-hub | `>=0.25.0` | HuggingFace モデルダウンロード |
| setuptools | `<75` | ビルドツール |
| faster-whisper | (最新) | 音声認識 (STT) |
| sounddevice | (最新) | オーディオ入出力 |
| style-bert-vits2 | (最新) | 音声合成 (TTS) |

### 別途インストール

| パッケージ | インストール方法 | 用途 |
|-----------|----------------|------|
| llama-cpp-python | CUDA版ホイールを直接インストール | LLM推論 |
| PyTorch (CUDA) | `--index-url` 指定でインストール | TTS推論バックエンド |

### 主要な間接依存

| パッケージ | 経由 | 用途 |
|-----------|------|------|
| CTranslate2 | faster-whisper | Whisper 高速推論 |
| PortAudio | sounddevice | クロスプラットフォームオーディオ |
| ggml | llama-cpp-python | テンソル演算 (C/C++) |

---

## 設計思想と内部構造

### モジュール分割

**Core層** (`src/core/`) はUIに依存しない純粋なビジネスロジック層です。

| モジュール | 責務 | 主要クラス |
|-----------|------|-----------|
| `character.py` | キャラクター定義の読み書き | `Character`, `CharacterManager` |
| `chat_engine.py` | チャット履歴管理、推論実行、思考フィルタ | `ChatEngine`, `ChatMessage` |
| `model_loader.py` | GGUFモデル読み込み・アンロード・生成 | `ModelLoader`, `ModelInfo`, `HFModelEntry` |
| `gpu_utils.py` | NVML経由のGPU VRAM情報取得 | ユーティリティ関数 |
| `voice_input.py` | マイク録音・Whisper音声認識 | `VoiceEngine`, `AudioDevice` |
| `voice_output.py` | Style-Bert-VITS2 音声合成・再生 | `TTSEngine`, `TTSStyleOverride` |
| `tts_model_manager.py` | TTSモデルの検出・管理 | `TTSModelManager` |

### スレッディングモデル

- **UIスレッド**: PySide6 イベントループ（メインスレッド）
- **推論スレッド**: LLM ストリーミング生成はバックグラウンドスレッドで実行
- **録音スレッド**: sounddevice のコールバック方式（別スレッドで音声データを収集）
- **TTS再生スレッド**: `sounddevice.play()` はブロッキング呼び出しのため必要に応じて分離

各コンポーネントは `threading.Lock` で排他制御を行います。

### エラーハンドリング

- CUDA 状態のリセット（`torch.cuda.synchronize()` + `empty_cache()`）
- cuDNN の無効化（TTSモデルロード時、dtype不一致防止）
- KVキャッシュのリセット（Mamba2ハイブリッドモデルのデコードエラー防止）
- speaker_id のクランプ（モデルの話者数を超えた場合は0にフォールバック）

### ビルド・デプロイ

| ファイル | 用途 |
|---------|------|
| `Start.cmd` | ワンクリック起動（venv作成 → 依存インストール → アプリ起動） |
| `build_llama_cuda.cmd` | llama-cpp-python ソースビルド（VS2022 + CUDA 12.8） |
| `requirements.txt` | pip 依存パッケージ |
| `.gitignore` | `venv/`, `models/*.gguf`, `tts_models/`, `logs/` を除外 |

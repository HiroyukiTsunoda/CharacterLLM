# CharacterLLM

<div align="center">

**ローカルLLM & OpenAI API で動くキャラクターAIチャット — Windows デスクトップアプリ（サンドボックス）**

GGUF モデルの GPU 推論と OpenAI API (ChatGPT) の両方に対応。
個性豊かなキャラクターと音声付きで会話できます。

</div>

---

## 特徴

| 機能 | 概要 |
|------|------|
| **ローカルLLM推論** | llama-cpp-python + CUDA でGGUFモデルをGPUフル活用 |
| **OpenAI API 対応** | GPT-4o / GPT-4o mini / o3-mini 等をAPI経由で利用可能。ワンクリックでローカル⇔クラウドを切替 |
| **キャラクター切替** | JSON定義のキャラクターを自由に追加・即座に切替 |
| **性格カスタマイズ** | システムプロンプト・口調・背景設定をキャラクターごとに編集 |
| **モデル管理** | ローカルGGUFファイルの切替 + HuggingFaceからワンクリックダウンロード |
| **思考モデル対応** | Qwen3 / GPT-OSS / Nemotron / o3-mini 等の thinking モデルに対応（思考過程の表示・折りたたみ） |
| **音声入力 (STT)** | faster-whisper による日本語音声認識 |
| **音声出力 (TTS)** | Qwen3-TTS によるキャラクター別音声合成（CustomVoice / VoiceDesign / 音声クローン対応） |
| **チャット履歴** | SQLiteによるセッション管理・履歴保存・復元 |
| **ストリーミング生成** | トークン単位のリアルタイム出力（ローカル・OpenAI 両対応） |
| **動的トークン調整** | ユーザー入力の内容・長さに応じてmax_tokensを自動最適化 |
| **ダークテーマUI** | PySide6ベースのモダンなデスクトップUI |

## 動作要件

- **OS**: Windows 10 / 11
- **Python**: 3.11+ (3.12 推奨)
- **GPU**: NVIDIA GPU (CUDA対応、VRAM 8GB以上推奨) ※ローカル推論を使う場合
- **CUDA Toolkit**: 12.4+ ※ローカル推論を使う場合
- **OpenAI API Key**: OpenAI API を使う場合のみ必要

## クイックスタート

### ワンクリック起動

```
Start.cmd をダブルクリック
```

仮想環境の作成 → 依存パッケージのインストール → アプリ起動をすべて自動で行います。

### 手動セットアップ

```bash
# 仮想環境を作成
python -m venv venv
venv\Scripts\activate

# llama-cpp-python (CUDA 12.4版) をインストール
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl

# PyTorch (CUDA版) をインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# その他の依存パッケージをインストール
pip install -r requirements.txt

# models/ ディレクトリに .gguf ファイルを配置

# アプリを起動
python main.py
```

### OpenAI API を使う場合

GPUがなくても OpenAI API を使ってチャットできます。

1. `.env.example` を `.env` にコピーし、APIキーを設定:

```
OPENAI_API_KEY=sk-proj-your-api-key-here
```

2. アプリを起動し、サイドバーの「モデル」タブで推論エンジンを **OpenAI API (ChatGPT)** に切り替え
3. モデルを選択して「接続」をクリック

## 推論エンジン

アプリ内のサイドバーからローカル推論と OpenAI API をワンクリックで切り替えられます。

| エンジン | 特徴 | 要件 |
|----------|------|------|
| **ローカル (GGUF)** | プライバシー重視・無料・オフライン可 | NVIDIA GPU + CUDA |
| **OpenAI API** | 高品質・GPU不要・セットアップ簡単 | APIキー + インターネット接続 |

### OpenAI 対応モデル

| モデル | 特徴 |
|--------|------|
| GPT-4o | 最新・高性能 |
| GPT-4o mini | 高速・低コスト |
| GPT-4 Turbo | 高性能 |
| GPT-3.5 Turbo | 最安 |
| o3-mini | 推論（思考）モデル |

## おすすめローカルモデル

アプリ内の「モデルダウンロード」からHuggingFaceより直接ダウンロードできます。

### 日本語特化モデル

| モデル | 量子化 | VRAM目安 | 思考 | 特徴 |
|--------|--------|----------|------|------|
| GPT-OSS-Swallow-20B-RL | Q4_K_M | ~16.5GB | ✅ | 日本語強化GPT-OSS・MoE |
| Nemotron-Nano-9B-v2-JP | Q4_K_M | ~7GB | ✅ | NVIDIA Mamba2ハイブリッド |
| Qwen3-Swallow-8B-SFT | Q4_K_M | ~5.5GB | ✅ | 日本語強化Qwen3 |

### 汎用モデル

| モデル | 量子化 | VRAM目安 | 思考 | 特徴 |
|--------|--------|----------|------|------|
| Qwen3-8B | Q4_K_M | ~5GB | ✅ | 軽量・高品質 |
| Qwen3-14B | Q4_K_M | ~9GB | ✅ | バランス型 |
| Gemma3-12B-IT | Q4_K_M | ~7.5GB | ✅ | 感情表現が豊か |
| GPT-OSS-20B | Q4_K_M | ~12GB | ✅ | MoE軽量高性能 |
| Llama3.1-8B-Inst | Q4_K_M | ~5GB | — | 汎用・RP派生豊富 |
| Mistral-Nemo-12B | Q4_K_M | ~7.5GB | — | RP特化向き |

## 同梱キャラクター

| ID | 名前 | 概要 |
|----|------|------|
| `default_assistant` | アシスタント | 丁寧語の汎用AIアシスタント |
| `tsundere_maid` | ツンデレメイド アリス | ツンデレ口調のメイド |
| `wise_sage` | 賢者マーリン | 古風な口調の賢者 |

キャラクターは `characters/` に JSON ファイルを追加することで自由に増やせます。

## 設定

### アプリ設定 (`config.json`)

| セクション | キー | 説明 | デフォルト |
|------------|------|------|-----------|
| `gpu` | `n_gpu_layers` | GPUオフロード層数 (`-1` = 全層) | `-1` |
| `gpu` | `main_gpu` | 使用するGPUのインデックス | `0` |
| `inference` | `context_length` | コンテキスト長 | `4096` |
| `inference` | `temperature` | 温度パラメータ | `0.7` |
| `inference` | `top_p` | Top-p サンプリング | `0.9` |
| `inference` | `max_tokens` | 最大生成トークン数 | `512` |
| `inference` | `repeat_penalty` | 繰り返しペナルティ | `1.1` |
| `ui` | `theme` | UIテーマ | `dark` |
| `ui` | `font_size` | フォントサイズ | `14` |
| `voice` | `whisper_model` | Whisperモデルサイズ | `medium` |
| `voice` | `device` | 推論デバイス | `cuda` |
| `voice` | `language` | 認識言語 | `ja` |
| `tts` | `enabled` | TTS有効化 | `false` |
| `tts` | `use_gpu` | TTS GPU使用 | `true` |
| `tts` | `auto_play` | 自動再生 | `true` |
| `openai` | `model` | OpenAI モデル名 | `gpt-4o` |
| `openai` | `enabled` | 起動時に OpenAI モードを有効化 | `false` |

### 環境変数 (`.env`)

| 変数名 | 説明 |
|--------|------|
| `OPENAI_API_KEY` | OpenAI API キー（OpenAI モードを使う場合に必須） |

`.env.example` をコピーして `.env` を作成してください。

## ディレクトリ構成

```
CharacterLLM/
├── main.py                  # エントリポイント
├── config.json              # アプリ設定
├── .env.example             # 環境変数テンプレート (APIキー等)
├── Start.cmd                # ワンクリック起動スクリプト
├── build_llama_cuda.cmd     # llama-cpp-python CUDAビルド用
├── requirements.txt         # Python依存パッケージ
├── characters/              # キャラクター定義JSON
│   ├── default_assistant.json
│   ├── tsundere_maid.json
│   └── wise_sage.json
├── models/                  # GGUFモデル格納先 (.gitignore対象)
├── tts_models/              # TTSモデル格納先 (.gitignore対象) ※Qwen3-TTSはHuggingFaceキャッシュを使用
├── src/
│   ├── core/
│   │   ├── character.py     # キャラクター読込・管理
│   │   ├── chat_engine.py   # チャットエンジン・推論実行
│   │   ├── model_loader.py  # GGUFモデル管理・HFダウンロード
│   │   ├── openai_provider.py # OpenAI API プロバイダー
│   │   ├── gpu_utils.py     # GPU情報取得 (NVML)
│   │   ├── voice_input.py   # 音声入力 (Whisper STT)
│   │   ├── qwen3_tts_engine.py     # 音声出力 (Qwen3-TTS)
│   │   ├── qwen3_tts_model_manager.py # Qwen3-TTS モデル管理
│   │   ├── tts_base.py             # TTS エンジン抽象基底
│   │   └── tts_router.py           # TTS ルーター
│   ├── ui/
│   │   ├── main_window.py   # メインウィンドウ
│   │   ├── chat_widget.py   # チャット表示・入力
│   │   ├── character_panel.py # キャラクター選択パネル
│   │   ├── model_panel.py   # モデル選択・エンジン切替パネル
│   │   ├── voice_panel.py   # 音声設定パネル
│   │   ├── voice_button.py  # 音声入力ボタン
│   │   ├── settings_dialog.py # 設定ダイアログ
│   │   └── styles.py        # QSSスタイル定義
│   └── data/
│       └── database.py      # SQLiteチャット履歴
├── logs/                    # ログ出力先 (.gitignore対象)
├── resources/
│   └── icons/               # アイコン素材
└── docs/
    ├── USAGE.md             # 使い方マニュアル
    └── TECH_STACK.md        # 技術スタック詳細
```

## ドキュメント

- **[使い方マニュアル](docs/USAGE.md)** — セットアップから各機能の操作方法まで
- **[技術スタック詳細](docs/TECH_STACK.md)** — 使用技術・アーキテクチャ・設計思想

## ライセンス

MIT License

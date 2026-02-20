# CharacterLLM

キャラクターAIとチャットできるWindowsデスクトップアプリケーション。  
ローカルLLM (GGUF) をGPU推論し、定義したキャラクターの口調・性格で会話できます。

## 特徴

- **ローカルLLM推論** — llama-cpp-python + CUDA でGGUFモデルをGPU推論
- **キャラクター切替** — JSON定義のキャラクターを自由に追加・切替
- **性格カスタマイズ** — システムプロンプト・口調・背景設定をキャラクターごとに編集
- **モデル管理** — ローカルGGUFファイルの切替 + HuggingFaceからのダウンロード
- **チャット履歴** — SQLiteによるセッション管理・履歴保存
- **ストリーミング生成** — トークン単位のリアルタイム出力
- **ダークテーマUI** — PySide6ベースのモダンなデスクトップUI

## 動作要件

- Windows 10/11
- Python 3.11+ (3.12推奨)
- NVIDIA GPU (CUDA対応、VRAM 8GB以上推奨)
- CUDA Toolkit 12.4

## クイックスタート

`Start.cmd` をダブルクリックすると、仮想環境の作成・依存パッケージのインストール・アプリ起動をすべて自動で行います。

### 手動セットアップ

```bash
# 仮想環境を作成
python -m venv venv
venv\Scripts\activate

# llama-cpp-python (CUDA 12.4版) をインストール
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl

# その他の依存パッケージをインストール
pip install -r requirements.txt

# models/ ディレクトリに .gguf ファイルを配置

# アプリを起動
python main.py
```

## 動作確認済みモデル

| モデル | 量子化 | サイズ | 備考 |
|--------|--------|--------|------|
| Qwen3-14B | Q6_K | ~12GB | 高品質 |
| gemma-3-12b-it | Q4_K_M | ~7GB | バランス良 |
| gpt-oss-20b | Q4_K_M | ~11GB | 大規模 |
| Meta-Llama-3.1-8B-Instruct | Q8_0 | ~8.5GB | 軽量高品質 |
| Mistral-Nemo-Instruct-2407 | Q4_K_M | ~7.5GB | 軽量 |

## 同梱キャラクター

| ID | 名前 | 概要 |
|----|------|------|
| `default_assistant` | アシスタント | 丁寧語の汎用AIアシスタント |
| `tsundere_maid` | ツンデレメイド アリス | ツンデレ口調のメイド |
| `wise_sage` | 賢者マーリン | 古風な口調の賢者 |

キャラクターは `characters/` に JSON ファイルを追加することで自由に増やせます。

## 設定 (`config.json`)

| キー | 説明 | デフォルト |
|------|------|-----------|
| `gpu.n_gpu_layers` | GPUオフロード層数 (`-1` = 全層) | `-1` |
| `inference.context_length` | コンテキスト長 | `4096` |
| `inference.temperature` | 温度パラメータ | `0.7` |
| `inference.max_tokens` | 最大生成トークン数 | `512` |
| `ui.theme` | UIテーマ | `dark` |
| `ui.font_size` | フォントサイズ | `14` |

## ディレクトリ構成

```
CharacterLLM/
├── main.py                  # エントリポイント
├── config.json              # アプリ設定
├── Start.cmd                # ワンクリック起動スクリプト
├── build_llama_cuda.cmd     # llama-cpp-python CUDAビルド用
├── requirements.txt         # Python依存パッケージ
├── characters/              # キャラクター定義JSON
│   ├── default_assistant.json
│   ├── tsundere_maid.json
│   └── wise_sage.json
├── models/                  # GGUFモデル格納先 (.gitignore対象)
├── src/
│   ├── core/
│   │   ├── character.py     # キャラクター読込・管理
│   │   ├── chat_engine.py   # チャットエンジン・推論実行
│   │   ├── model_loader.py  # GGUFモデル管理・HFダウンロード
│   │   └── gpu_utils.py     # GPU情報取得
│   ├── ui/
│   │   ├── main_window.py   # メインウィンドウ
│   │   ├── chat_widget.py   # チャット表示・入力
│   │   ├── character_panel.py # キャラクター選択パネル
│   │   ├── model_panel.py   # モデル選択パネル
│   │   ├── settings_dialog.py # 設定ダイアログ
│   │   └── styles.py        # QSSスタイル定義
│   └── data/
│       └── database.py      # SQLiteチャット履歴
├── logs/                    # ログ出力先 (.gitignore対象)
└── resources/
    └── icons/               # アイコン素材
```

## ライセンス

MIT License

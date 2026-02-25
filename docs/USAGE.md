# CharacterLLM — 使い方マニュアル

## 目次

1. [インストール](#インストール)
2. [初回起動](#初回起動)
3. [モデルの準備](#モデルの準備)
4. [チャットする](#チャットする)
5. [キャラクターの管理](#キャラクターの管理)
6. [音声機能](#音声機能)
7. [設定のカスタマイズ](#設定のカスタマイズ)
8. [トラブルシューティング](#トラブルシューティング)

---

## インストール

### 必要なもの

| 項目 | 要件 |
|------|------|
| OS | Windows 10 / 11 |
| Python | 3.11 以上（3.12 推奨） |
| GPU | NVIDIA CUDA対応GPU（VRAM 8GB以上推奨） |
| CUDA Toolkit | 12.4 以上 |
| ディスク | モデルサイズ + 5GB 程度の空き容量 |

### 方法A: ワンクリック起動（推奨）

`Start.cmd` をダブルクリックするだけで、以下がすべて自動で実行されます。

1. Python 仮想環境 (`venv`) の作成
2. llama-cpp-python (CUDA 12.4版) のインストール
3. その他依存パッケージのインストール
4. アプリケーションの起動

2回目以降は仮想環境の有効化とアプリ起動のみ行われるため高速です。

### 方法B: 手動セットアップ

```bash
# 1. 仮想環境を作成・有効化
python -m venv venv
venv\Scripts\activate

# 2. llama-cpp-python (CUDA 12.4版) をインストール
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl

# 3. PyTorch (CUDA版) をインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. その他の依存パッケージをインストール
pip install -r requirements.txt

# 5. アプリを起動
python main.py
```

### 方法C: llama-cpp-python をソースからビルド

Visual Studio 2022 と CUDA Toolkit 12.8 がある場合、`build_llama_cuda.cmd` を管理者権限で実行するとソースからCUDAビルドできます。カスタムのCUDA Architectureを指定したい場合に有用です。

---

## 初回起動

アプリを起動すると、以下の画面が表示されます。

- **左サイドバー**: キャラクター選択パネル、モデル選択パネル、音声設定パネル
- **メインエリア**: チャット画面（メッセージ表示 + 入力欄）

初回起動時に必要な手順:

1. **モデルを準備する**（次セクション参照）
2. **キャラクターを選択する** — サイドバーのキャラクター一覧からクリック
3. **チャットを開始する** — 入力欄にメッセージを入力してEnterまたは送信ボタン

---

## モデルの準備

### アプリ内からダウンロード（推奨）

1. サイドバーの **モデルパネル** を開く
2. 「ダウンロード」タブを選択
3. おすすめモデル一覧から希望のモデルを選択
4. 「ダウンロード」ボタンをクリック
5. HuggingFace からダウンロードが開始される（進捗表示あり）
6. 完了後、ローカルモデル一覧に自動追加される

### 手動配置

1. HuggingFace などから `.gguf` ファイルをダウンロード
2. `models/` ディレクトリに配置
3. アプリのモデルパネルでリストを更新

### モデルの読み込み

1. モデルパネルの「ローカル」タブでモデルを選択
2. 「読み込み」ボタンをクリック
3. GPU にモデルが読み込まれる（モデルサイズとVRAMに応じて数秒〜数十秒）

### モデル選びの目安

| VRAM | おすすめモデル |
|------|---------------|
| 8GB | Qwen3-8B (Q4_K_M)、Qwen3-Swallow-8B-SFT (Q4_K_M) |
| 12GB | Gemma3-12B-IT (Q4_K_M)、Nemotron-Nano-9B-v2-JP (Q8_0) |
| 16GB | Qwen3-14B (Q4_K_M)、GPT-OSS-20B (Q4_K_M) |
| 24GB | Qwen3-32B (Q4_K_M)、GPT-OSS-Swallow-20B-RL (Q4_K_M) |

---

## チャットする

### 基本操作

1. キャラクターとモデルを選択・読み込み済みであることを確認
2. 画面下部の入力欄にメッセージを入力
3. **Enter** キーまたは送信ボタンでメッセージを送信
4. ストリーミングでリアルタイムに応答が表示される

### 思考モデルの表示

Qwen3、GPT-OSS、Nemotron などの思考モデルを使用している場合:

- 応答前にモデルの思考過程が折りたたみセクションで表示される
- 思考内容は自動的に検出・分離される
- 対応フォーマット:
  - `<think>...</think>` タグ（Qwen3、Gemma）
  - `<|channel|>analysis` チャネル（GPT-OSS）
  - `思考内容` / `応答内容` セクション

### 動的トークン調整

入力内容に応じて、応答の長さが自動調整されます。

| 入力パターン | max_tokens | 応答の長さ |
|-------------|-----------|-----------|
| 短い挨拶（〜20文字） | 128 | 1〜2文の短い返答 |
| 通常の会話（〜80文字） | 256 | 3〜5文程度 |
| 質問・相談 | 512 | 丁寧な応答 |
| 「教えて」「詳しく」等 | 768 | 詳細な説明 |

### セッション管理

- **新規セッション**: メニューまたはショートカットで新しい会話を開始
- **履歴の保存**: チャット履歴はSQLiteデータベースに自動保存される
- **履歴の復元**: 過去のセッションを選択して会話を再開
- **履歴のクリア**: 現在の会話履歴をクリア（データベースからも削除可能）

---

## キャラクターの管理

### キャラクターの選択

サイドバーのキャラクター一覧からクリックして選択します。選択すると即座にシステムプロンプトが切り替わり、そのキャラクターの口調・性格で応答が生成されます。

### キャラクターの追加

`characters/` ディレクトリに以下の形式でJSONファイルを作成します。

```json
{
  "id": "my_character",
  "name": "キャラクター名",
  "avatar": "",
  "personality": {
    "system_prompt": "キャラクターの基本設定を記述",
    "tone": "口調の特徴",
    "speech_style": "具体的な話し方の例",
    "background": "キャラクターの背景設定"
  },
  "generation_params": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "repeat_penalty": 1.1
  },
  "tts": {
    "enabled": true,
    "model_path": "tts_models\\モデル名\\model.safetensors",
    "config_path": "tts_models\\モデル名\\config.json",
    "style_vec_path": "tts_models\\モデル名\\style_vectors.npy",
    "style": "Neutral",
    "speaker_id": 0,
    "sdp_ratio": 0.0,
    "noise": 0.6,
    "noise_w": 0.8,
    "length": 0.95,
    "style_weight": 5.0
  }
}
```

### キャラクター設定の詳細

#### personality セクション

| フィールド | 説明 | 例 |
|-----------|------|-----|
| `system_prompt` | LLMに渡すシステムプロンプト | 「あなたはツンデレなメイドの...」 |
| `tone` | 口調の特徴キーワード | 「ツンデレ、時々デレる、照れ隠し」 |
| `speech_style` | 話し方の具体例 | 「〜なんだからね！」「べ、別に…」 |
| `background` | キャラクターの背景・設定 | 「執事学校を首席で卒業した...」 |

#### generation_params セクション

| パラメータ | 説明 | 範囲 | 推奨値 |
|-----------|------|------|--------|
| `temperature` | ランダム性の度合い | 0.0〜2.0 | 0.7〜0.85 |
| `top_p` | 累積確率サンプリング | 0.0〜1.0 | 0.9 |
| `max_tokens` | 基本の最大トークン数 | 64〜2048 | 512 |
| `repeat_penalty` | 繰り返し抑制 | 1.0〜2.0 | 1.1〜1.15 |

> **ヒント**: `temperature` を上げるほど応答のバリエーションが増え、キャラクターらしい表現が出やすくなります。ただし高すぎると支離滅裂になることがあります。

#### tts セクション

| パラメータ | 説明 |
|-----------|------|
| `enabled` | このキャラクターのTTSを有効にするか |
| `model_path` | TTSモデルファイルのパス |
| `config_path` | TTS設定ファイルのパス |
| `style_vec_path` | スタイルベクトルファイルのパス |
| `style` | 基本音声スタイル（Neutral, Happy, Angry, Sad 等） |
| `speaker_id` | 話者ID |
| `sdp_ratio` | SDP比率（音声の安定性） |
| `noise` | ノイズ値（表現力に影響） |
| `noise_w` | ノイズ重み |
| `length` | 話速（大きいほどゆっくり） |
| `style_weight` | スタイルの強さ |

---

## 音声機能

### 音声入力（STT）

faster-whisper を使用した音声認識機能です。

#### セットアップ

1. サイドバーの **音声パネル** を開く
2. 入力デバイス（マイク）を選択
3. Whisperモデルサイズを選択（`small` / `medium` / `large`）
4. 「モデルをロード」ボタンをクリック

#### 使い方

1. チャット画面の音声ボタンを押して録音開始
2. マイクに向かって話す
3. もう一度ボタンを押して録音停止
4. 認識結果が入力欄に自動入力される
5. `auto_send` が有効な場合は自動送信される

#### Whisperモデルの選び方

| モデル | VRAM追加 | 認識精度 | 速度 |
|--------|---------|---------|------|
| `small` | ~1GB | 良好 | 高速 |
| `medium` | ~2.5GB | 高精度 | 普通 |
| `large` | ~4GB | 最高精度 | 低速 |

> **注意**: LLMモデルとVRAMを共有するため、大きなLLMモデルと `large` Whisperモデルの併用にはVRAMの余裕が必要です。

### 音声出力（TTS）

Style-Bert-VITS2 を使用したキャラクター別音声合成機能です。

#### セットアップ

1. TTSモデルを `tts_models/` ディレクトリに配置
2. キャラクターJSONの `tts` セクションでモデルパスを設定
3. `config.json` の `tts.enabled` を `true` に設定

#### TTSモデルの配置

```
tts_models/
├── jvnv-M1-jp/         # 男性声1（賢者マーリン用）
│   ├── jvnv-M1-jp_e158_s14000.safetensors
│   ├── config.json
│   └── style_vectors.npy
├── jvnv-M2-jp/         # 男性声2（アシスタント用）
│   └── ...
└── jvnv-F2-jp/         # 女性声（ツンデレメイド用）
    └── ...
```

#### ト書き（演出指示）機能

LLMの応答に括弧付きのト書きが含まれている場合、TTSパラメータが自動調整されます。

| ト書き例 | 効果 |
|---------|------|
| （小声で）、（ささやき） | 静かでゆっくりした声 |
| （大声で）、（叫んで） | 大きく力強い声 |
| （怒って）、（キレて） | 怒りのスタイル |
| （笑って）、（嬉しそうに） | 楽しげなスタイル |
| （泣いて）、（悲しそうに） | 悲しみのスタイル |
| （驚いて）、（びっくり） | 驚きのスタイル |
| （照れて）、（恥ずかしそうに） | おとなしい声 |
| （慌てて）、（焦って） | 早口で驚いた声 |

---

## 設定のカスタマイズ

### config.json

アプリのルートにある `config.json` を編集して各種設定を変更できます。

#### GPU設定

```json
{
  "gpu": {
    "n_gpu_layers": -1,
    "main_gpu": 0
  }
}
```

| 設定 | 説明 |
|------|------|
| `n_gpu_layers` | GPUにオフロードするレイヤー数。`-1` で全レイヤーをGPUに載せる |
| `main_gpu` | マルチGPU環境で使用するGPUのインデックス |

#### 推論パラメータ

```json
{
  "inference": {
    "context_length": 4096,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "repeat_penalty": 1.1
  }
}
```

`context_length` はモデルが参照できる会話の長さです。大きくするほどVRAMを消費しますが、長い会話を維持できます。

#### UI設定

```json
{
  "ui": {
    "theme": "dark",
    "window_width": 1920,
    "window_height": 1080,
    "font_size": 14
  }
}
```

#### 音声入力設定

```json
{
  "voice": {
    "input_device": 1,
    "output_device": 11,
    "whisper_model": "medium",
    "device": "cuda",
    "compute_type": "float16",
    "language": "ja",
    "auto_send": false
  }
}
```

| 設定 | 説明 |
|------|------|
| `input_device` / `output_device` | オーディオデバイスのインデックス（アプリ内で選択可能） |
| `whisper_model` | Whisperモデルサイズ（`small` / `medium` / `large`） |
| `device` | 推論デバイス（`cuda` / `cpu`） |
| `compute_type` | 計算精度（`float16` / `float32`） |
| `language` | 認識対象の言語 |
| `auto_send` | 音声認識後に自動送信するか |

#### TTS設定

```json
{
  "tts": {
    "enabled": true,
    "use_gpu": true,
    "auto_play": true,
    "bert_model": "ku-nlp/deberta-v2-large-japanese-char-wwm"
  }
}
```

| 設定 | 説明 |
|------|------|
| `enabled` | TTSのグローバル有効/無効 |
| `use_gpu` | GPU推論を使用するか |
| `auto_play` | 音声合成後に自動再生するか |
| `bert_model` | テキスト解析に使うBERTモデル名 |

---

## トラブルシューティング

### CUDA関連

**CUDA が認識されない**
- NVIDIA ドライバが最新か確認
- CUDA Toolkit 12.4+ がインストールされているか確認
- `nvidia-smi` コマンドでGPUが認識されるか確認

**CUBLAS_STATUS_EXECUTION_FAILED**
- LLM推論とTTS推論がGPU上で競合した場合に発生することがある
- アプリは自動的にCUDA状態をリセットして対処するため、通常は再試行で解決

**VRAMが不足する**
- より小さい量子化モデル（Q4_K_M）を使用
- `config.json` の `context_length` を小さくする
- Whisper / TTSモデルのGPU使用を無効にする

### モデル関連

**モデルがリストに表示されない**
- `.gguf` ファイルが `models/` ディレクトリに配置されているか確認
- ファイルが破損していないか確認（ダウンロードが完了しているか）

**応答がおかしい / 繰り返しが多い**
- `repeat_penalty` を `1.1`〜`1.2` に調整
- `temperature` を下げる（0.5〜0.7）
- 別のモデルを試す

### 音声関連

**マイクが認識されない**
- Windows のサウンド設定でマイクが有効か確認
- アプリの音声パネルでデバイスリストを更新
- `input_device` のインデックスが正しいか確認

**TTS の音声が再生されない**
- `tts_models/` にモデルファイルが配置されているか確認
- キャラクターJSONの `tts.model_path` のパスが正しいか確認
- `config.json` の `tts.enabled` が `true` になっているか確認

### ログの確認

アプリのログは `logs/` ディレクトリに出力されます。

- **アプリケーションログ**: エラーや警告の詳細
- **LLM Raw Output ログ**: モデルの生出力（思考ブロック含む）

ログは24時間で自動ローテーションされます。

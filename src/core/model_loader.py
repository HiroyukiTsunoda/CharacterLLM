"""
ModelLoader: GGUFモデルの読込・アンロード・GPU設定・HuggingFaceダウンロードを管理する。
"""

from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from llama_cpp import Llama

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """ローカルに存在するGGUFモデルの情報。"""
    name: str
    path: str
    size_bytes: int

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    @property
    def estimated_vram_gb(self) -> float:
        """ファイルサイズからVRAM使用量を概算。全レイヤーGPUオフロード前提。"""
        return self.size_gb + 0.5  # KV cache + コンテキストバッファ分の余裕

    def to_dict(self) -> dict:
        return {"name": self.name, "path": self.path, "size_bytes": self.size_bytes}


@dataclass
class HFModelEntry:
    """HuggingFaceからダウンロード可能なモデルの定義。"""
    repo_id: str
    filename: str
    display_name: str
    estimated_vram_gb: float
    supports_thinking: bool = False


# ---------------------------------------------------------------------------
# ダウンロード可能なおすすめモデル一覧
# ---------------------------------------------------------------------------

RECOMMENDED_MODELS: list[HFModelEntry] = [
    # --- GPT-OSS Swallow (日本語強化GPT-OSS・MoE 21B総/3.6Bアクティブ・analysis channel) ---
    HFModelEntry(
        repo_id="mmnga-o/GPT-OSS-Swallow-20B-RL-v0.1-gguf",
        filename="GPT-OSS-Swallow-20B-RL-v0.1-Q4_K_M.gguf",
        display_name="GPT-OSS-Swallow-20B-RL (Q4_K_M) [完全版]",
        estimated_vram_gb=16.5,
        supports_thinking=True,
    ),
    HFModelEntry(
        repo_id="mmnga-o/GPT-OSS-Swallow-20B-RL-v0.1-gguf",
        filename="GPT-OSS-Swallow-20B-RL-v0.1-Q5_K_M.gguf",
        display_name="GPT-OSS-Swallow-20B-RL (Q5_K_M) [完全版]",
        estimated_vram_gb=17.5,
        supports_thinking=True,
    ),
    # --- Nemotron Nano 9B v2 Japanese (NVIDIA・Mamba2ハイブリッド・日本語特化・thinking対応) ---
    HFModelEntry(
        repo_id="mmnga-o/NVIDIA-Nemotron-Nano-9B-v2-Japanese-gguf",
        filename="NVIDIA-Nemotron-Nano-9B-v2-Japanese-Q4_K_M.gguf",
        display_name="Nemotron-Nano-9B-v2-JP (Q4_K_M)",
        estimated_vram_gb=7.0,
        supports_thinking=True,
    ),
    HFModelEntry(
        repo_id="mmnga-o/NVIDIA-Nemotron-Nano-9B-v2-Japanese-gguf",
        filename="NVIDIA-Nemotron-Nano-9B-v2-Japanese-Q8_0.gguf",
        display_name="Nemotron-Nano-9B-v2-JP (Q8_0)",
        estimated_vram_gb=10.0,
        supports_thinking=True,
    ),
    # --- Qwen3 Swallow (日本語強化Qwen3・thinking対応) ---
    HFModelEntry(
        repo_id="mmnga-o/Qwen3-Swallow-8B-SFT-v0.2-gguf",
        filename="Qwen3-Swallow-8B-SFT-v0.2-Q4_K_M.gguf",
        display_name="Qwen3-Swallow-8B-SFT (Q4_K_M)",
        estimated_vram_gb=5.5,
        supports_thinking=True,
    ),
    HFModelEntry(
        repo_id="mmnga-o/Qwen3-Swallow-8B-SFT-v0.2-gguf",
        filename="Qwen3-Swallow-8B-SFT-v0.2-Q8_0.gguf",
        display_name="Qwen3-Swallow-8B-SFT (Q8_0)",
        estimated_vram_gb=9.5,
        supports_thinking=True,
    ),
    HFModelEntry(
        repo_id="mmnga-o/Qwen3-Swallow-30B-A3B-SFT-v0.2-gguf",
        filename="Qwen3-Swallow-30B-A3B-SFT-v0.2-Q4_K_M.gguf",
        display_name="Qwen3-Swallow-30B-A3B-SFT MoE (Q4_K_M)",
        estimated_vram_gb=19.0,
        supports_thinking=True,
    ),
    # --- Qwen3 (日本語最強・thinking対応) ---
    HFModelEntry(
        repo_id="Qwen/Qwen3-8B-GGUF",
        filename="Qwen3-8B-Q4_K_M.gguf",
        display_name="Qwen3-8B (Q4_K_M)",
        estimated_vram_gb=5.0,
        supports_thinking=True,
    ),
    HFModelEntry(
        repo_id="Qwen/Qwen3-14B-GGUF",
        filename="Qwen3-14B-Q4_K_M.gguf",
        display_name="Qwen3-14B (Q4_K_M)",
        estimated_vram_gb=9.0,
        supports_thinking=True,
    ),
    HFModelEntry(
        repo_id="Qwen/Qwen3-32B-GGUF",
        filename="Qwen3-32B-Q4_K_M.gguf",
        display_name="Qwen3-32B (Q4_K_M)",
        estimated_vram_gb=20.5,
        supports_thinking=True,
    ),
    # --- Qwen3-30B-A3B (MoE Shallow・30B総パラメータ/3.3Bアクティブ・VRAM効率◎) ---
    HFModelEntry(
        repo_id="Qwen/Qwen3-30B-A3B-GGUF",
        filename="Qwen3-30B-A3B-Q4_K_M.gguf",
        display_name="Qwen3-30B-A3B MoE (Q4_K_M)",
        estimated_vram_gb=18.0,
        supports_thinking=True,
    ),
    # --- Gemma 3 (日本語◎・感情表現が豊か・プロンプトで<think>誘導) ---
    HFModelEntry(
        repo_id="ggml-org/gemma-3-12b-it-GGUF",
        filename="gemma-3-12b-it-Q4_K_M.gguf",
        display_name="Gemma3-12B-IT (Q4_K_M)",
        estimated_vram_gb=7.5,
        supports_thinking=True,
    ),
    HFModelEntry(
        repo_id="ggml-org/gemma-3-27b-it-GGUF",
        filename="gemma-3-27b-it-Q4_K_M.gguf",
        display_name="Gemma3-27B-IT (Q4_K_M)",
        estimated_vram_gb=17.0,
        supports_thinking=True,
    ),
    # --- gpt-oss (OpenAI・MoE軽量高性能・analysis channel) ---
    HFModelEntry(
        repo_id="unsloth/gpt-oss-20b-GGUF",
        filename="gpt-oss-20b-Q4_K_M.gguf",
        display_name="GPT-OSS-20B (Q4_K_M)",
        estimated_vram_gb=12.0,
        supports_thinking=True,
    ),
    HFModelEntry(
        repo_id="unsloth/gpt-oss-20b-GGUF",
        filename="gpt-oss-20b-Q6_K.gguf",
        display_name="GPT-OSS-20B (Q6_K)",
        estimated_vram_gb=12.5,
        supports_thinking=True,
    ),
    HFModelEntry(
        repo_id="unsloth/gpt-oss-20b-GGUF",
        filename="gpt-oss-20b-Q8_0.gguf",
        display_name="GPT-OSS-20B (Q8_0)",
        estimated_vram_gb=13.0,
        supports_thinking=True,
    ),
    # --- Llama 3.1 (汎用・RP派生モデルが豊富) ---
    HFModelEntry(
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        display_name="Llama3.1-8B-Inst (Q4_K_M)",
        estimated_vram_gb=5.0,
    ),
    HFModelEntry(
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        display_name="Llama3.1-8B-Inst (Q8_0)",
        estimated_vram_gb=9.0,
    ),
    # --- Mistral (RP特化fine-tuneの実績豊富) ---
    HFModelEntry(
        repo_id="bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        filename="Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        display_name="Mistral-Nemo-12B (Q4_K_M)",
        estimated_vram_gb=7.5,
    ),
    HFModelEntry(
        repo_id="bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
        filename="Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
        display_name="Mistral-Small-24B (Q4_K_M)",
        estimated_vram_gb=14.0,
    ),
]


# ---------------------------------------------------------------------------
# ModelLoader 本体
# ---------------------------------------------------------------------------

class ModelLoader:
    """GGUFモデルのライフサイクル管理。"""

    def __init__(self, models_dir: str | Path, config: dict | None = None):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

        self._model: Optional[Llama] = None
        self._current_model_path: Optional[str] = None
        self._supports_thinking: bool = False
        self._uses_think_tags: bool = False
        self._template_inserts_think: bool = False

    # ------------------------------------------------------------------
    # プロパティ
    # ------------------------------------------------------------------

    @property
    def model(self) -> Optional[Llama]:
        """現在ロードされているLlamaモデル。"""
        return self._model

    @property
    def current_model_path(self) -> Optional[str]:
        return self._current_model_path

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def supports_thinking(self) -> bool:
        """ロード中のモデルが内部推論（<think> や analysis channel）を出力するか。"""
        return self._supports_thinking

    @property
    def uses_think_tags(self) -> bool:
        """ロード中のモデルが <think> ブロックを使用するか（ネイティブまたはプロンプト誘導）。"""
        return self._uses_think_tags

    @property
    def template_inserts_think(self) -> bool:
        """ロード中のモデルのチャットテンプレートが <think> を自動挿入するか。"""
        return self._template_inserts_think

    # ------------------------------------------------------------------
    # ローカルモデル一覧
    # ------------------------------------------------------------------

    def list_local_models(self) -> list[ModelInfo]:
        """models/ ディレクトリのGGUFファイルを列挙。"""
        models: list[ModelInfo] = []
        if not self.models_dir.exists():
            return models

        for p in sorted(self.models_dir.glob("*.gguf")):
            models.append(
                ModelInfo(
                    name=p.stem,
                    path=str(p),
                    size_bytes=p.stat().st_size,
                )
            )
        return models

    # ------------------------------------------------------------------
    # ロード / アンロード
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        **kwargs,
    ) -> Llama:
        """GGUFモデルをロードする。既にロード済みなら先にアンロードする。"""
        model_path = str(model_path)

        if self._model is not None:
            self.unload_model()

        gpu_cfg = self.config.get("gpu", {})
        n_gpu_layers = gpu_cfg.get("n_gpu_layers", n_gpu_layers)
        main_gpu = gpu_cfg.get("main_gpu", 0)

        inf_cfg = self.config.get("inference", {})
        n_ctx = inf_cfg.get("context_length", n_ctx)

        n_batch = inf_cfg.get("n_batch", n_ctx)

        logger.info(
            "Loading model: %s (n_ctx=%d, n_batch=%d, n_gpu_layers=%d)",
            model_path, n_ctx, n_batch, n_gpu_layers,
        )

        self._model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            main_gpu=main_gpu,
            verbose=False,
            **kwargs,
        )
        self._current_model_path = model_path
        self._supports_thinking = self._detect_thinking_support(model_path)
        filename_lower = Path(model_path).name.lower()
        self._uses_think_tags = self._supports_thinking and "gpt-oss" not in filename_lower
        self._template_inserts_think = self._detect_template_think_insertion()
        logger.info(
            "Model loaded successfully: %s (thinking=%s, think_tags=%s, template_think=%s)",
            model_path, self._supports_thinking, self._uses_think_tags,
            self._template_inserts_think,
        )
        return self._model

    def unload_model(self) -> None:
        """現在のモデルをアンロードしてVRAMを解放。"""
        if self._model is not None:
            logger.info("Unloading model: %s", self._current_model_path)
            del self._model
            self._model = None
            self._current_model_path = None
            self._supports_thinking = False
            self._uses_think_tags = False
            self._template_inserts_think = False
            gc.collect()

    def _detect_template_think_insertion(self) -> bool:
        """モデルのチャットテンプレートに <think> 自動挿入が含まれるか検出する。"""
        if self._model is None:
            return False
        try:
            template = self._model.metadata.get("tokenizer.chat_template", "")
            if "<think>" in template:
                logger.info("Chat template contains <think> auto-insertion.")
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _detect_thinking_support(model_path: str) -> bool:
        """モデルパスから thinking 対応を判定する。

        RECOMMENDED_MODELS にファイル名が一致するエントリがあればその設定を使い、
        なければファイル名から推定する。
        """
        filename = Path(model_path).name.lower()
        for entry in RECOMMENDED_MODELS:
            if entry.filename.lower() == filename:
                return entry.supports_thinking
        return any(kw in filename for kw in ("qwen", "gemma", "gpt-oss", "swallow", "nemotron"))

    # ------------------------------------------------------------------
    # 推論
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """モデルの内部状態を完全にリセットする。

        Mamba2/Transformer ハイブリッドモデルでは、llama-cpp-python の
        プレフィックスキャッシュが再帰状態と干渉し 2 回目以降の推論で
        llama_decode エラーが発生する。明示的リセットで回避する。
        """
        try:
            self._model.reset()
        except AttributeError:
            try:
                if hasattr(self._model, "n_tokens"):
                    self._model.n_tokens = 0
                if hasattr(self._model, "_ctx"):
                    self._model._ctx.kv_cache_clear()
            except Exception:
                pass

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        repeat_penalty: float = 1.1,
        stream: bool = True,
    ) -> Iterator[str] | str:
        """チャット形式で推論を実行。stream=Trueならトークンを逐次yieldする。"""
        if self._model is None:
            raise RuntimeError("モデルがロードされていません。先にモデルを読み込んでください。")

        self._reset_state()

        if stream:
            return self._generate_stream(messages, temperature, top_p, max_tokens, repeat_penalty)
        else:
            response = self._model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repeat_penalty=repeat_penalty,
            )
            return response["choices"][0]["message"]["content"]

    def _generate_stream(
        self,
        messages: list[dict],
        temperature: float,
        top_p: float,
        max_tokens: int,
        repeat_penalty: float,
    ) -> Iterator[str]:
        """ストリーミング推論。トークンごとにyieldする。"""
        stream = self._model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repeat_penalty=repeat_penalty,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                logger.debug("token: %r", content)
                yield content

    # ------------------------------------------------------------------
    # HuggingFace ダウンロード
    # ------------------------------------------------------------------

    def download_model(
        self,
        repo_id: str,
        filename: str,
        progress_callback=None,
    ) -> str:
        """HuggingFace HubからGGUFモデルをダウンロードし、models/に保存する。

        Args:
            repo_id: HuggingFace リポジトリID (例: "Qwen/Qwen3-8B-GGUF")
            filename: ダウンロードするファイル名 (例: "qwen3-8b-q4_k_m.gguf")
            progress_callback: 進捗コールバック(bytesダウンロード済み, bytes合計)

        Returns:
            ダウンロード先のローカルパス
        """
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(self.models_dir),
            local_dir_use_symlinks=False,
        )
        logger.info("Model downloaded: %s -> %s", filename, local_path)
        return local_path

    def get_recommended_models(self) -> list[HFModelEntry]:
        """おすすめモデル一覧を返す。"""
        return RECOMMENDED_MODELS.copy()

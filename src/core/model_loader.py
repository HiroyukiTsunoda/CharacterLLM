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
# ダウンロード可能なおすすめモデル一覧 (resources/recommended_models.json)
# ---------------------------------------------------------------------------

_CATALOG_PATH = Path(__file__).parent.parent.parent / "resources" / "recommended_models.json"


def _load_recommended_models() -> list[HFModelEntry]:
    """resources/recommended_models.json からモデルカタログを読み込む。"""
    try:
        with open(_CATALOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            HFModelEntry(
                repo_id=m["repo_id"],
                filename=m["filename"],
                display_name=m["display_name"],
                estimated_vram_gb=m["estimated_vram_gb"],
                supports_thinking=m.get("supports_thinking", False),
            )
            for m in data.get("models", [])
        ]
    except Exception as e:
        logger.warning("Failed to load model catalog (%s): %s", _CATALOG_PATH, e)
        return []


RECOMMENDED_MODELS: list[HFModelEntry] = _load_recommended_models()


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
        self._needs_state_reset: bool = False

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
        self._needs_state_reset = "nemotron" in filename_lower
        logger.info(
            "Model loaded successfully: %s (thinking=%s, think_tags=%s, template_think=%s, state_reset=%s)",
            model_path, self._supports_thinking, self._uses_think_tags,
            self._template_inserts_think, self._needs_state_reset,
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
            self._needs_state_reset = False
            gc.collect()

    @staticmethod
    def synchronize_gpu() -> None:
        """CUDA カーネルの完了を待つ。

        llama.cpp (ggml) と PyTorch (TTS等) が同一GPUを共有するため、
        推論完了後・後続のGPU処理前に呼んで競合を防ぐ。
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def _detect_template_think_insertion(self) -> bool:
        """モデルのチャットテンプレートが <think> を自動挿入し、
        モデルが思考内容を出力した後 </think> で閉じることを期待するか検出する。

        Qwen3.5 Small (4B/9B等) はデフォルトで思考無効: テンプレートが
        <think>\\n\\n</think> と閉じた状態で挿入するため、モデルは思考タグを
        出力しない。この場合は False を返す。
        """
        if self._model is None:
            return False
        try:
            template = self._model.metadata.get("tokenizer.chat_template", "")
            if "<think>" not in template:
                return False

            # add_generation_prompt 付近のテンプレートロジックを取得
            gen_idx = template.find("add_generation_prompt")
            if gen_idx == -1:
                logger.info("Chat template contains <think> (no generation prompt section).")
                return True

            gen_section = template[gen_idx:]

            if "enable_thinking" in gen_section:
                # Qwen3.5 系: enable_thinking による条件分岐あり
                # デフォルト（else節）が <think>\n だけなら思考有効、
                # <think>...\n</think> と閉じていたら思考無効
                #
                # 思考有効テンプレート (35B-A3B):
                #   if enable_thinking is false → <think>\n\n</think>
                #   else → <think>\n          ← デフォルト = 思考ON
                #
                # 思考無効テンプレート (4B/9B):
                #   if enable_thinking is true → <think>\n
                #   else → <think>\n\n</think> ← デフォルト = 思考OFF
                #
                # enable_thinking 直後の else節に </think> があるかで判定
                et_idx = gen_section.find("enable_thinking")
                else_idx = gen_section.find("else", et_idx)
                if else_idx != -1:
                    else_section = gen_section[else_idx:else_idx + 200]
                    if "</think>" in else_section:
                        logger.info(
                            "Chat template has conditional thinking "
                            "(default=disabled, Qwen3.5 Small pattern)."
                        )
                        return False
                    else:
                        logger.info(
                            "Chat template has conditional thinking "
                            "(default=enabled, Qwen3.5 MoE pattern)."
                        )
                        return True

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
        """Mamba2/SSM ハイブリッドモデル専用: 内部状態をリセットする。

        Mamba2 (Nemotron等) ではプレフィックスキャッシュが再帰状態と干渉し
        2 回目以降の推論で llama_decode エラーが発生する。
        通常の Transformer モデルでは不要（むしろ KV キャッシュ破壊の原因）。
        """
        if not self._needs_state_reset:
            return
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

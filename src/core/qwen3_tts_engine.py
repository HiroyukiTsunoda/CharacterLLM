"""
Qwen3TTSEngine: Qwen3-TTS によるキャラクター別音声合成エンジン。

qwen-tts パッケージを使用し、CustomVoice / VoiceDesign / Base(音声クローン)
の3モードに対応する。
"""

from __future__ import annotations

import io
import logging
import os
import re
import sys
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.core.tts_base import TTSEngineBase

if TYPE_CHECKING:
    from src.core.character import Character

logger = logging.getLogger(__name__)

# ト書きキーワード → Qwen3-TTS instructions テキスト変換マップ
_DIRECTION_TO_INSTRUCTION: dict[str, str] = {
    "小声": "小声でささやくように話して",
    "ささやき": "小声でささやくように話して",
    "ささやく": "小声でささやくように話して",
    "囁": "小声でささやくように話して",
    "ひそひそ": "小声でささやくように話して",
    "ぼそ": "小声でぼそぼそと話して",
    "そっと": "そっと優しく話して",
    "つぶやき": "小声でつぶやくように話して",
    "つぶやく": "小声でつぶやくように話して",
    "呟": "小声でつぶやくように話して",
    "大声": "大きな声で力強く話して",
    "叫": "叫ぶように大声で話して",
    "絶叫": "絶叫するように話して",
    "怒鳴": "怒鳴るように大声で話して",
    "怒": "怒った口調で話して",
    "キレ": "怒った口調で話して",
    "イライラ": "イライラした口調で話して",
    "激怒": "激怒した口調で話して",
    "笑": "笑いながら楽しそうに話して",
    "嬉し": "嬉しそうに明るく話して",
    "楽し": "楽しそうに明るく話して",
    "ニコニコ": "にこにこと嬉しそうに話して",
    "ウキウキ": "ウキウキと楽しそうに話して",
    "はしゃ": "はしゃぐように元気よく話して",
    "泣": "泣きそうな声で話して",
    "悲し": "悲しそうに話して",
    "切な": "切なそうに話して",
    "しんみり": "しんみりと静かに話して",
    "涙": "涙声で話して",
    "驚": "驚いた声で話して",
    "びっくり": "びっくりした声で話して",
    "恐": "怖がっている声で話して",
    "怖": "怖がっている声で話して",
    "ビクビク": "おびえた声で話して",
    "おびえ": "おびえた声で話して",
    "震え": "震えた声で話して",
    "嫌": "嫌そうな声で話して",
    "うんざり": "うんざりした声で話して",
    "不快": "不快そうな声で話して",
    "照れ": "恥ずかしそうに話して",
    "恥ずかし": "恥ずかしそうに話して",
    "慌て": "慌てた声で早口に話して",
    "焦": "焦った声で話して",
}

_STAGE_DIR_RE = re.compile(r'[（(]([^）)]+)[）)]')


def _parse_instructions_from_stage_directions(text: str) -> tuple[str, str]:
    """テキスト中のト書きを解析し、(クリーンテキスト, instructions) を返す。

    最初に検出されたト書きキーワードを instructions に変換する。
    認識されない括弧テキストはそのまま残す。
    """
    instructions = ""
    cleaned_parts: list[str] = []
    last_end = 0

    for match in _STAGE_DIR_RE.finditer(text):
        content = match.group(1)
        matched_instruction = None
        for keyword, instr in _DIRECTION_TO_INSTRUCTION.items():
            if keyword in content:
                matched_instruction = instr
                break

        if matched_instruction is not None:
            before = text[last_end:match.start()]
            if before.strip():
                cleaned_parts.append(before)
            if not instructions:
                instructions = matched_instruction
            last_end = match.end()
        # 認識されない括弧テキストはそのまま残す

    remaining = text[last_end:]
    if remaining.strip():
        cleaned_parts.append(remaining)

    clean_text = "".join(cleaned_parts).strip() if cleaned_parts else text
    return clean_text, instructions


class Qwen3TTSEngine(TTSEngineBase):
    """Qwen3-TTS を使ったキャラクター別 TTS エンジン。

    qwen-tts パッケージの Qwen3TTSModel を使用する。
    モデルはキャラクターの qwen3_model 設定をキーにキャッシュされる。
    """

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        qwen3_cfg = self._config.get("tts", {}).get("qwen3", {})
        self._default_model: str = qwen3_cfg.get(
            "default_model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        )
        self._dtype: str = qwen3_cfg.get("dtype", "bfloat16")
        self._flash_attention: bool = qwen3_cfg.get("flash_attention", True)
        self._char_model_map: dict[str, str] = {}
        self._available = False
        self._check_availability()

    def _check_availability(self) -> None:
        """qwen-tts パッケージがインストールされているか確認する。

        起動高速化のため、実際の import は行わず find_spec で存在確認のみ行う。
        重い依存 (torch, transformers, pysox) のロードはモデル使用時まで遅延させる。
        """
        import importlib.util
        if importlib.util.find_spec("qwen_tts") is not None:
            self._available = True
        else:
            self._available = False
            logger.warning(
                "qwen-tts パッケージが見つかりません。"
                "Qwen3-TTS を使用するには 'pip install qwen-tts' を実行してください。"
            )

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # モデルロード
    # ------------------------------------------------------------------

    _qwen3_model_cls: type | None = None

    @classmethod
    def _import_qwen3_model(cls) -> type:
        """qwen_tts.Qwen3TTSModel を import して返す。

        初回 import 時に flash-attn / SoX の警告が print/logging で
        出力されるため、stdout/stderr とロガーを一時的に抑制する。
        """
        if cls._qwen3_model_cls is not None:
            return cls._qwen3_model_cls

        sox_logger = logging.getLogger("sox")
        prev_sox_level = sox_logger.level
        sox_logger.setLevel(logging.CRITICAL)

        devnull = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            from qwen_tts import Qwen3TTSModel
            cls._qwen3_model_cls = Qwen3TTSModel
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sox_logger.setLevel(prev_sox_level)

        return cls._qwen3_model_cls

    def _get_torch_dtype(self):
        import torch
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._dtype, torch.bfloat16)

    @staticmethod
    def _is_model_cached(model_id: str) -> bool:
        """HuggingFace キャッシュにモデルが存在するか確認する。"""
        try:
            from huggingface_hub import try_to_load_from_cache
            result = try_to_load_from_cache(model_id, "config.json")
            return result is not None and isinstance(result, str)
        except Exception:
            return False

    def _get_or_load_model(self, model_id: str) -> object:
        """Qwen3TTSModel をキャッシュから取得、なければロードする。"""
        with self._lock:
            if model_id in self._model_cache:
                return self._model_cache[model_id]

        if not self._available:
            raise RuntimeError(
                "qwen-tts パッケージがインストールされていません。"
            )

        Qwen3TTSModel = self._import_qwen3_model()

        device = f"cuda:0" if self.device == "cuda" else "cpu"
        dtype = self._get_torch_dtype()

        kwargs: dict = {
            "device_map": device,
            "dtype": dtype,
        }
        if self._flash_attention and self.device == "cuda":
            try:
                import flash_attn  # noqa: F401
                kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.info("flash-attn not installed; using default attention.")

        cached = self._is_model_cached(model_id)
        if cached:
            kwargs["local_files_only"] = True
            logger.info(
                "Loading Qwen3-TTS model from cache: %s (device=%s, dtype=%s)",
                model_id, device, self._dtype,
            )
        else:
            logger.info(
                "Downloading & loading Qwen3-TTS model: %s (device=%s, dtype=%s)",
                model_id, device, self._dtype,
            )

        try:
            model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
        except Exception:
            if cached:
                logger.warning(
                    "local_files_only load failed for %s; retrying with download.",
                    model_id,
                )
                kwargs.pop("local_files_only", None)
                model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
            else:
                raise

        with self._lock:
            self._model_cache[model_id] = model

        logger.info("Qwen3-TTS model loaded: %s", model_id)
        return model

    def _release_stale_model(self, character_id: str, new_model_id: str) -> None:
        """キャラクターが別モデルへ切り替わる場合、旧モデルを解放する。

        他のキャラクターがまだ同じモデルを参照していれば repo_id キーは残す。
        """
        with self._lock:
            old_model_id = self._char_model_map.get(character_id)
            if not old_model_id or old_model_id == new_model_id:
                return

            self._model_cache.pop(character_id, None)

            still_used = any(
                mid == old_model_id
                for cid, mid in self._char_model_map.items()
                if cid != character_id
            )
            if not still_used:
                self._model_cache.pop(old_model_id, None)
                logger.info(
                    "Released model '%s' (no longer used after char '%s' "
                    "switched to '%s').",
                    old_model_id, character_id, new_model_id,
                )
            del self._char_model_map[character_id]

    def try_register_cached_model(self, character: Character) -> bool:
        """キャラクターの使用モデルが既にキャッシュ済みなら紐付けだけ行う。

        Returns:
            True ならモデルは既に VRAM 上にあり、追加ロード不要。
        """
        tts = character.tts_params
        model_id = tts.qwen3_model or self._default_model
        with self._lock:
            if model_id in self._model_cache:
                old = self._char_model_map.get(character.id)
                if old != model_id:
                    self._model_cache[character.id] = self._model_cache[model_id]
                    self._char_model_map[character.id] = model_id
                    logger.info(
                        "Registered char '%s' with already-cached model '%s'.",
                        character.id, model_id,
                    )
                return True
        return False

    def load_model_for_character(self, character: Character) -> None:
        """キャラクターの Qwen3-TTS モデルを明示的にメモリへロードする。"""
        tts = character.tts_params
        model_id = tts.qwen3_model or self._default_model
        self._release_stale_model(character.id, model_id)
        self._get_or_load_model(model_id)
        with self._lock:
            self._model_cache[character.id] = self._model_cache[model_id]
            self._char_model_map[character.id] = model_id

    def loaded_character_ids(self) -> list[str]:
        with self._lock:
            return list(self._char_model_map.keys())

    def unload_character(self, character_id: str) -> None:
        with self._lock:
            old_model_id = self._char_model_map.pop(character_id, None)
            self._model_cache.pop(character_id, None)

            if old_model_id:
                still_used = any(
                    mid == old_model_id
                    for mid in self._char_model_map.values()
                )
                if not still_used:
                    self._model_cache.pop(old_model_id, None)
                    logger.info(
                        "Released model '%s' (last user '%s' unloaded).",
                        old_model_id, character_id,
                    )
                else:
                    logger.info(
                        "Unloaded char '%s'; model '%s' still in use.",
                        character_id, old_model_id,
                    )

    def unload_all(self) -> None:
        """全リソースを解放する。"""
        with self._lock:
            self._char_model_map.clear()
        self.clear_cache()
        logger.info("Qwen3TTSEngine fully unloaded.")

    # ------------------------------------------------------------------
    # 音声合成
    # ------------------------------------------------------------------

    def synthesize(
        self, text: str, character: Character, *, force: bool = False,
    ) -> tuple[int, np.ndarray]:
        """テキストを音声に変換する。

        Returns:
            (sample_rate, audio_int16) のタプル。
        """
        if not self._available:
            raise RuntimeError(
                "qwen-tts パッケージがインストールされていません。"
            )

        if not force:
            if not self._enabled:
                raise RuntimeError("TTS が無効です。")
            tts_p = character.tts_params
            if not tts_p.enabled:
                raise RuntimeError(
                    f"キャラクター '{character.name}' の TTS が無効です。"
                )

        tts = character.tts_params
        model_id = tts.qwen3_model or self._default_model
        self._release_stale_model(character.id, model_id)
        model = self._get_or_load_model(model_id)

        with self._lock:
            self._model_cache[character.id] = model
            self._char_model_map[character.id] = model_id

        cleaned = self._clean_text_for_tts(text)
        if not cleaned:
            raise RuntimeError("TTS に渡せるテキストがありません。")

        import torch

        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # ト書き解析して instructions を抽出
        clean_text, stage_instructions = _parse_instructions_from_stage_directions(cleaned)
        if not clean_text:
            clean_text = cleaned

        instructions = stage_instructions or tts.qwen3_instructions or ""
        mode = self._resolve_mode(model, tts)

        logger.info(
            "Qwen3-TTS synthesizing (mode=%s): '%s'",
            mode, clean_text[:60],
        )

        with torch.no_grad():
            wavs, sr = self._generate(
                model, clean_text, mode, tts, instructions,
            )

        if not wavs or len(wavs[0]) == 0:
            raise RuntimeError("Qwen3-TTS: 音声生成に失敗しました。")

        audio = wavs[0]
        if isinstance(audio, np.ndarray):
            pass
        else:
            audio = np.array(audio, dtype=np.float32)

        if audio.dtype in (np.float32, np.float64):
            audio = (audio * 32767).astype(np.int16)

        tail_pad = np.zeros(int(sr * 0.15), dtype=np.int16)
        audio = np.concatenate([audio, tail_pad])

        return sr, audio

    @staticmethod
    def _detect_model_mode(model: object) -> str | None:
        """ロード済みモデルの実際のタイプ (サポートするモード) を検出する。"""
        for getter in [
            lambda: getattr(model, "tts_model_type", None),
            lambda: getattr(getattr(model, "config", None), "tts_model_type", None),
        ]:
            try:
                val = getter()
                if val and isinstance(val, str):
                    return val
            except Exception:
                continue
        return None

    def _resolve_mode(self, model: object, tts_params) -> str:
        """ロード済みモデルからモードを導出する。

        優先順位: モデルの tts_model_type → KNOWN_MODELS → tts_params.qwen3_mode
        """
        detected = self._detect_model_mode(model)
        if detected:
            return detected

        repo_id = getattr(tts_params, "qwen3_model", "") or self._default_model
        from src.core.qwen3_tts_model_manager import KNOWN_MODELS
        for m in KNOWN_MODELS:
            if m.repo_id == repo_id:
                return m.mode

        return getattr(tts_params, "qwen3_mode", "") or "custom_voice"

    def _generate(
        self,
        model: object,
        text: str,
        mode: str,
        tts_params,
        instructions: str,
    ) -> tuple[list, int]:
        """モードに応じて適切な生成メソッドを呼び出す。"""

        if mode == "voice_design":
            return model.generate_voice_design(
                text=text,
                language="Japanese",
                instruct=instructions or "自然な日本語の女性の声",
            )
        elif mode == "base":
            ref_audio = tts_params.qwen3_ref_audio
            ref_text = tts_params.qwen3_ref_text
            if not ref_audio:
                raise RuntimeError(
                    "Base モード (音声クローン) には参照音声が必要です。"
                )
            return model.generate_voice_clone(
                text=text,
                language="Japanese",
                ref_audio=ref_audio,
                ref_text=ref_text or "",
            )
        else:
            speaker = tts_params.qwen3_voice or "Vivian"
            kwargs: dict = {
                "text": text,
                "speaker": speaker,
                "language": "Japanese",
            }
            if instructions:
                kwargs["instruct"] = instructions
            return model.generate_custom_voice(**kwargs)

    def get_config(self) -> dict:
        """現在の設定を辞書として返す。"""
        return {
            "enabled": self._enabled,
            "use_gpu": self._use_gpu,
            "auto_play": self._auto_play,
            "qwen3": {
                "default_model": self._default_model,
                "dtype": self._dtype,
                "flash_attention": self._flash_attention,
            },
        }

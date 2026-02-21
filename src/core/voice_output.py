"""
TTSEngine: Style-Bert-VITS2 によるキャラクター別音声合成と再生を管理する。
"""

from __future__ import annotations

import io
import logging
import re
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from src.core.character import Character
    from src.core.tts_model_manager import LocalTTSModel

logger = logging.getLogger(__name__)


@dataclass
class TTSStyleOverride:
    """ト書き（演出指示）から解釈されたTTSパラメータのオーバーライド。

    None のフィールドはキャラクターの既定値を使用する。
    """
    style: Optional[str] = None
    sdp_ratio: Optional[float] = None
    noise: Optional[float] = None
    noise_w: Optional[float] = None
    length: Optional[float] = None
    style_weight: Optional[float] = None


class TTSEngine:
    """Style-Bert-VITS2 を使ったキャラクター別 TTS エンジン。

    BERTモデルは初回合成時に遅延ロードされ、
    TTSモデルはキャラクターIDをキーにキャッシュされる。
    """

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        tts_cfg = self._config.get("tts", {})

        self._enabled: bool = tts_cfg.get("enabled", False)
        self._use_gpu: bool = tts_cfg.get("use_gpu", True)
        self._auto_play: bool = tts_cfg.get("auto_play", True)
        self._bert_model_name: str = tts_cfg.get(
            "bert_model", "ku-nlp/deberta-v2-large-japanese-char-wwm"
        )

        self._bert_loaded = False
        self._model_cache: dict[str, object] = {}
        self._lock = threading.Lock()
        self._playback_stop = threading.Event()

    # ------------------------------------------------------------------
    # プロパティ
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def use_gpu(self) -> bool:
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, value: bool):
        if value != self._use_gpu:
            self._use_gpu = value
            self.clear_cache()

    @property
    def auto_play(self) -> bool:
        return self._auto_play

    @auto_play.setter
    def auto_play(self, value: bool):
        self._auto_play = value

    @property
    def device(self) -> str:
        if self._use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            logger.warning("CUDA is not available; falling back to CPU.")
        return "cpu"

    @property
    def cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @property
    def is_bert_loaded(self) -> bool:
        return self._bert_loaded

    # ------------------------------------------------------------------
    # BERT モデル管理
    # ------------------------------------------------------------------

    def load_bert(self) -> None:
        """SBV2 用の BERT モデル / トークナイザをロードする。"""
        if self._bert_loaded:
            return

        from style_bert_vits2.constants import Languages
        from style_bert_vits2.nlp import bert_models

        logger.info("Loading BERT model: %s", self._bert_model_name)
        bert_models.load_model(Languages.JP, self._bert_model_name)
        bert_models.load_tokenizer(Languages.JP, self._bert_model_name)

        # BERT hidden_states が float16 のまま TTS へ渡ると
        # CUDA カーネル内で dtype assert が発生する。
        # キャッシュ済みモデルを float32 に統一して防止する。
        bert_model = bert_models.load_model(Languages.JP)
        bert_model.float()

        self._bert_loaded = True
        logger.info("BERT model loaded (float32).")

    # ------------------------------------------------------------------
    # TTS モデル管理（キャラクター別キャッシュ）
    # ------------------------------------------------------------------

    def is_character_loaded(self, character_id: str) -> bool:
        """指定キャラクターの TTS モデルがメモリにロード済みか。"""
        with self._lock:
            return character_id in self._model_cache

    def loaded_character_ids(self) -> list[str]:
        """メモリにロード済みの全キャラクター ID を返す。"""
        with self._lock:
            return list(self._model_cache.keys())

    def load_model_for_character(self, character: Character) -> None:
        """キャラクターの TTS モデルを明示的にメモリへロードする。"""
        self._get_or_load_model(character)

    def _get_or_load_model(self, character: Character) -> object:
        """キャラクターに紐づく TTSModel をキャッシュから取得、なければロードする。"""
        from style_bert_vits2.tts_model import TTSModel

        char_id = character.id
        with self._lock:
            if char_id in self._model_cache:
                return self._model_cache[char_id]

        tts = character.tts_params
        model_path = Path(tts.model_path)
        config_path = Path(tts.config_path)
        style_vec_path = Path(tts.style_vec_path)

        for p, label in [
            (model_path, "model_path"),
            (config_path, "config_path"),
            (style_vec_path, "style_vec_path"),
        ]:
            if not p.exists():
                raise FileNotFoundError(
                    f"TTS {label} が見つかりません: {p}"
                )

        if not self._bert_loaded:
            self.load_bert()

        logger.info(
            "Loading TTS model for '%s': %s (device=%s)",
            character.name, model_path.name, self.device,
        )

        import torch
        if torch.cuda.is_available() and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.enabled = False

        model = TTSModel(
            model_path=model_path,
            config_path=config_path,
            style_vec_path=style_vec_path,
            device=self.device,
        )

        # safetensors の重みが float16 で保存されている場合、
        # 推論時に float32 バイアスとの dtype 不一致で CUDA assert が発生する。
        # 明示的にロードして全パラメータを float32 に統一する。
        # ※ weight_norm の除去は不要（フックが float32 の weight_g/weight_v から
        #   weight を再計算するため、自然に float32 になる）。
        model.load()
        net_g = getattr(model, "_TTSModel__net_g", None)
        if net_g is not None:
            net_g.float()
            logger.info("TTS net_g converted to float32.")

        with self._lock:
            self._model_cache[char_id] = model

        logger.info("TTS model loaded for '%s'.", character.name)
        return model

    def clear_cache(self) -> None:
        """全キャラクターの TTS モデルキャッシュをクリアする。"""
        with self._lock:
            self._model_cache.clear()
        logger.info("TTS model cache cleared.")

    def unload_character(self, character_id: str) -> None:
        """特定キャラクターの TTS モデルをアンロードする。"""
        with self._lock:
            if character_id in self._model_cache:
                del self._model_cache[character_id]
                logger.info("TTS model unloaded: %s", character_id)

    def unload_all(self) -> None:
        """全リソースを解放する。"""
        self.clear_cache()
        self._bert_loaded = False
        logger.info("TTSEngine fully unloaded.")

    # ------------------------------------------------------------------
    # 音声合成
    # ------------------------------------------------------------------

    def synthesize(
        self, text: str, character: Character, *, force: bool = False
    ) -> tuple[int, np.ndarray]:
        """テキストを音声に変換する。

        Args:
            force: True の場合、グローバル/キャラクター有効チェックをスキップする（テスト発話用）。

        Returns:
            (sample_rate, audio_int16) のタプル。
        """
        if not force:
            if not self._enabled:
                raise RuntimeError("TTS が無効です。")

            tts_p = character.tts_params
            if not tts_p.enabled:
                raise RuntimeError(
                    f"キャラクター '{character.name}' の TTS が無効です。"
                )

        tts = character.tts_params
        model = self._get_or_load_model(character)

        cleaned = self._clean_text_for_tts(text)
        if not cleaned:
            raise RuntimeError("TTS に渡せるテキストがありません。")

        directed_segments = self._parse_stage_directions(cleaned)

        logger.info(
            "Synthesizing %d directed segment(s): '%s'",
            len(directed_segments), cleaned[:60],
        )

        import torch

        # llama-cpp-python (ggml) と PyTorch が同一 GPU 上で cuBLAS ハンドルを
        # 共有するため、LLM 推論直後に TTS を実行すると
        # CUBLAS_STATUS_EXECUTION_FAILED が発生する。
        # 事前に同期・キャッシュクリアで CUDA 状態をリセットする。
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        speaker_id = tts.speaker_id
        n_speakers = self._get_n_speakers(model)
        if n_speakers is not None and speaker_id >= n_speakers:
            logger.warning(
                "speaker_id=%d exceeds n_speakers=%d; clamping to 0.",
                speaker_id, n_speakers,
            )
            speaker_id = 0

        all_audio: list[np.ndarray] = []
        sr = 44100

        for seg_text, override in directed_segments:
            seg_text = seg_text.strip()
            if not seg_text:
                continue

            style = (
                override.style if override and override.style
                else tts.style
            )
            sdp_ratio = (
                override.sdp_ratio if override and override.sdp_ratio is not None
                else tts.sdp_ratio
            )
            noise = (
                override.noise if override and override.noise is not None
                else tts.noise
            )
            noise_w = (
                override.noise_w if override and override.noise_w is not None
                else tts.noise_w
            )
            length = (
                override.length if override and override.length is not None
                else tts.length
            )
            style_weight = (
                override.style_weight
                if override and override.style_weight is not None
                else tts.style_weight
            )

            if override:
                logger.info(
                    "  Stage direction applied: style=%s, noise=%.2f, "
                    "length=%.2f, text='%s'",
                    style, noise, length, seg_text[:30],
                )

            sub_segments = self._split_text(seg_text, max_len=100)
            for sub_seg in sub_segments:
                with torch.no_grad():
                    seg_sr, seg_audio = model.infer(
                        text=sub_seg,
                        speaker_id=speaker_id,
                        style=style,
                        sdp_ratio=sdp_ratio,
                        noise=noise,
                        noise_w=noise_w,
                        length=length,
                        style_weight=style_weight,
                    )
                sr = seg_sr
                all_audio.append(seg_audio)

        if not all_audio:
            raise RuntimeError("TTS に渡せるテキストがありません。")

        # 末尾に無音を付加して最後の音が途切れるのを防ぐ
        tail_pad = np.zeros(int(sr * 0.15), dtype=all_audio[0].dtype)

        if len(all_audio) == 1:
            return sr, np.concatenate([all_audio[0], tail_pad])

        pause = np.zeros(int(sr * 0.3), dtype=all_audio[0].dtype)
        parts: list[np.ndarray] = []
        for i, a in enumerate(all_audio):
            parts.append(a)
            if i < len(all_audio) - 1:
                parts.append(pause)
        parts.append(tail_pad)
        return sr, np.concatenate(parts)

    @staticmethod
    def _get_n_speakers(model: object) -> int | None:
        """TTSModel の config から n_speakers を取得する。取得できなければ None。"""
        try:
            config = getattr(model, "config", None) or getattr(model, "hps", None)
            if config is not None:
                data = getattr(config, "data", config)
                n = getattr(data, "n_speakers", None)
                if isinstance(n, int):
                    return n
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # テキスト前処理
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text_for_tts(text: str) -> str:
        """LLM 応答テキストから TTS に不適切な要素を除去する。"""
        import re
        t = text.strip()
        t = re.sub(r"```[\s\S]*?```", "", t)
        t = re.sub(r"`[^`]+`", "", t)
        t = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", t)
        t = re.sub(r"#{1,6}\s*", "", t)
        t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
        t = re.sub(r"[*_~|>]+", "", t)
        t = re.sub(r"[^\u3000-\u9FFF\u30A0-\u30FF\u3040-\u309F"
                    r"\uFF00-\uFFEF\u4E00-\u9FAF"
                    r"a-zA-Za-zA-Z0-90-9"
                    r"、。！？!?.,;:…ー〜（）()「」『』【】\s\n]", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _split_text(text: str, max_len: int = 100) -> list[str]:
        """テキストを句読点で区切り、max_len 以下のセグメントに分割する。"""
        import re
        sentences = re.split(r"(?<=[。！？!?\n])", text)
        segments: list[str] = []
        buf = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(buf) + len(s) > max_len and buf:
                segments.append(buf)
                buf = s
            else:
                buf += s
        if buf:
            segments.append(buf)
        return segments if segments else [text[:max_len]]

    # ------------------------------------------------------------------
    # ト書き（演出指示）解析
    # ------------------------------------------------------------------

    _STAGE_DIR_RE = re.compile(r'[（(]([^）)]+)[）)]')

    # パラメータオーバーライドのプリセット
    _OV_WHISPER = TTSStyleOverride(noise=0.3, noise_w=0.4, length=1.2)
    _OV_SHOUT = TTSStyleOverride(
        style="Angry", noise=0.9, noise_w=1.0, length=0.85, style_weight=8.0,
    )
    _OV_LOUD = TTSStyleOverride(
        noise=0.9, noise_w=1.0, length=0.85, style_weight=8.0,
    )
    _OV_ANGRY = TTSStyleOverride(style="Angry", style_weight=8.0)
    _OV_HAPPY = TTSStyleOverride(style="Happy", style_weight=7.0)
    _OV_SAD = TTSStyleOverride(style="Sad", style_weight=7.0)
    _OV_SURPRISE = TTSStyleOverride(style="Surprise", style_weight=7.0)
    _OV_FEAR = TTSStyleOverride(style="Fear", style_weight=7.0)
    _OV_DISGUST = TTSStyleOverride(style="Disgust", style_weight=6.0)
    _OV_SHY = TTSStyleOverride(noise=0.4, noise_w=0.6, length=1.1)
    _OV_FLUSTERED = TTSStyleOverride(
        style="Surprise", noise=0.8, length=0.85,
    )

    _DIRECTION_RULES: list[tuple[str, TTSStyleOverride]] = [
        # 小声・囁き系
        ("小声", _OV_WHISPER),
        ("ささやき", _OV_WHISPER),
        ("ささやく", _OV_WHISPER),
        ("囁", _OV_WHISPER),
        ("ひそひそ", _OV_WHISPER),
        ("ぼそ", _OV_WHISPER),
        ("そっと", _OV_WHISPER),
        ("つぶやき", _OV_WHISPER),
        ("つぶやく", _OV_WHISPER),
        ("呟", _OV_WHISPER),
        # 大声・叫び系
        ("大声", _OV_LOUD),
        ("叫", _OV_SHOUT),
        ("絶叫", _OV_SHOUT),
        ("怒鳴", _OV_SHOUT),
        # 怒り
        ("怒", _OV_ANGRY),
        ("キレ", _OV_ANGRY),
        ("イライラ", _OV_ANGRY),
        ("激怒", _OV_ANGRY),
        # 喜び・笑い
        ("笑", _OV_HAPPY),
        ("嬉し", _OV_HAPPY),
        ("楽し", _OV_HAPPY),
        ("ニコニコ", _OV_HAPPY),
        ("ウキウキ", _OV_HAPPY),
        ("はしゃ", _OV_HAPPY),
        # 悲しみ
        ("泣", _OV_SAD),
        ("悲し", _OV_SAD),
        ("切な", _OV_SAD),
        ("しんみり", _OV_SAD),
        ("涙", _OV_SAD),
        # 驚き
        ("驚", _OV_SURPRISE),
        ("びっくり", _OV_SURPRISE),
        # 恐怖
        ("恐", _OV_FEAR),
        ("怖", _OV_FEAR),
        ("ビクビク", _OV_FEAR),
        ("おびえ", _OV_FEAR),
        ("震え", _OV_FEAR),
        # 嫌悪
        ("嫌", _OV_DISGUST),
        ("うんざり", _OV_DISGUST),
        ("不快", _OV_DISGUST),
        # 照れ・恥ずかしさ
        ("照れ", _OV_SHY),
        ("恥ずかし", _OV_SHY),
        # 慌て・焦り
        ("慌て", _OV_FLUSTERED),
        ("焦", _OV_FLUSTERED),
    ]

    @classmethod
    def _match_direction(cls, content: str) -> Optional[TTSStyleOverride]:
        """括弧内テキストを既知のト書きキーワードと照合する。"""
        for keyword, override in cls._DIRECTION_RULES:
            if keyword in content:
                return override
        return None

    @classmethod
    def _parse_stage_directions(
        cls, text: str,
    ) -> list[tuple[str, Optional[TTSStyleOverride]]]:
        """テキスト中のト書き（演出指示）を解析し、セグメントとオーバーライドの組を返す。

        認識されたト書き（例: （小声で）、（怒って））は読み上げテキストから除去され、
        直後のテキストセグメントに対応するスタイルオーバーライドが付与される。
        認識されない括弧テキストはそのまま読み上げ対象として残る。
        """
        segments: list[tuple[str, Optional[TTSStyleOverride]]] = []
        current_override: Optional[TTSStyleOverride] = None
        last_end = 0

        for match in cls._STAGE_DIR_RE.finditer(text):
            content = match.group(1)
            override = cls._match_direction(content)
            if override is not None:
                before = text[last_end:match.start()]
                if before.strip():
                    segments.append((before, current_override))
                current_override = override
                last_end = match.end()
                logger.debug(
                    "Stage direction detected: '(%s)' → %s",
                    content, override,
                )

        remaining = text[last_end:]
        if remaining.strip():
            segments.append((remaining, current_override))

        if not segments:
            return [(text, None)]
        return segments

    def synthesize_to_wav(
        self, text: str, character: Character
    ) -> bytes:
        """テキストを WAV バイト列に変換する。"""
        sr, audio = self.synthesize(text, character)
        return self._audio_to_wav(sr, audio)

    # ------------------------------------------------------------------
    # 音声再生
    # ------------------------------------------------------------------

    def play_audio(
        self,
        sr: int,
        audio: np.ndarray,
        output_device: int | None = None,
    ) -> None:
        """sounddevice で音声を再生する（ブロッキング）。"""
        import sounddevice as sd

        self._playback_stop.clear()

        if audio.dtype != np.int16:
            if audio.dtype in (np.float32, np.float64):
                audio = (audio * 32767).astype(np.int16)

        logger.info(
            "Playing audio: %d samples, sr=%d, device=%s",
            len(audio), sr, output_device,
        )

        sd.play(audio, samplerate=sr, device=output_device)
        sd.wait()

    def stop_playback(self) -> None:
        """再生中の音声を停止する。"""
        try:
            import sounddevice as sd
            sd.stop()
        except Exception as e:
            logger.warning("Failed to stop playback: %s", e)
        self._playback_stop.set()

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    @staticmethod
    def _audio_to_wav(sr: int, audio: np.ndarray) -> bytes:
        """numpy 音声データを WAV バイト列に変換する。"""
        if audio.dtype in (np.float32, np.float64):
            audio = (audio * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.tobytes())
        buf.seek(0)
        return buf.read()

    def get_config(self) -> dict:
        """現在の設定を辞書として返す。"""
        return {
            "enabled": self._enabled,
            "use_gpu": self._use_gpu,
            "auto_play": self._auto_play,
            "bert_model": self._bert_model_name,
        }

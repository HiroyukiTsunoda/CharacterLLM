"""
CharacterSystem: キャラクター定義JSONの読込・保存・一覧管理。
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class Personality:
    """キャラクターの性格設定。"""
    system_prompt: str = ""
    tone: str = ""
    speech_style: str = ""
    background: str = ""

    def build_system_message(self) -> str:
        """各フィールドを結合して完全なシステムプロンプトを組み立てる。"""
        parts: list[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        if self.tone:
            parts.append(f"口調: {self.tone}")
        if self.speech_style:
            parts.append(f"話し方: {self.speech_style}")
        if self.background:
            parts.append(f"背景: {self.background}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "tone": self.tone,
            "speech_style": self.speech_style,
            "background": self.background,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Personality:
        return cls(
            system_prompt=data.get("system_prompt", ""),
            tone=data.get("tone", ""),
            speech_style=data.get("speech_style", ""),
            background=data.get("background", ""),
        )


@dataclass
class GenerationParams:
    """キャラクターごとの生成パラメータ。"""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    repeat_penalty: float = 1.1

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "repeat_penalty": self.repeat_penalty,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GenerationParams:
        return cls(
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
            max_tokens=data.get("max_tokens", 512),
            repeat_penalty=data.get("repeat_penalty", 1.1),
        )


@dataclass
class TTSParams:
    """キャラクターごとの音声合成パラメータ (Qwen3-TTS)。"""
    enabled: bool = False
    engine: str = "qwen3"

    # --- Qwen3-TTS ---
    qwen3_model: str = ""
    qwen3_mode: str = "custom_voice"
    qwen3_voice: str = ""
    qwen3_instructions: str = ""
    qwen3_ref_audio: str = ""
    qwen3_ref_text: str = ""

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "engine": self.engine,
            "qwen3_model": self.qwen3_model,
            "qwen3_mode": self.qwen3_mode,
            "qwen3_voice": self.qwen3_voice,
            "qwen3_instructions": self.qwen3_instructions,
            "qwen3_ref_audio": self.qwen3_ref_audio,
            "qwen3_ref_text": self.qwen3_ref_text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TTSParams:
        return cls(
            enabled=data.get("enabled", False),
            engine=data.get("engine", "qwen3"),
            qwen3_model=data.get("qwen3_model", ""),
            qwen3_mode=data.get("qwen3_mode", "custom_voice"),
            qwen3_voice=data.get("qwen3_voice", ""),
            qwen3_instructions=data.get("qwen3_instructions", ""),
            qwen3_ref_audio=data.get("qwen3_ref_audio", ""),
            qwen3_ref_text=data.get("qwen3_ref_text", ""),
        )


@dataclass
class Character:
    """キャラクター定義。"""
    id: str
    name: str
    avatar: str = ""
    personality: Personality = field(default_factory=Personality)
    generation_params: GenerationParams = field(default_factory=GenerationParams)
    tts_params: TTSParams = field(default_factory=TTSParams)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "avatar": self.avatar,
            "personality": self.personality.to_dict(),
            "generation_params": self.generation_params.to_dict(),
            "tts": self.tts_params.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Character:
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Unknown"),
            avatar=data.get("avatar", ""),
            personality=Personality.from_dict(data.get("personality", {})),
            generation_params=GenerationParams.from_dict(data.get("generation_params", {})),
            tts_params=TTSParams.from_dict(data.get("tts", {})),
        )

    def get_system_message(self) -> dict:
        """LLM用のsystemメッセージを返す。"""
        return {
            "role": "system",
            "content": self.personality.build_system_message(),
        }


# ---------------------------------------------------------------------------
# CharacterManager
# ---------------------------------------------------------------------------

class CharacterManager:
    """characters/ ディレクトリ内のキャラクターJSONを管理。"""

    def __init__(self, characters_dir: str | Path):
        self.characters_dir = Path(characters_dir)
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        self._characters: dict[str, Character] = {}
        self.reload()

    # ------------------------------------------------------------------
    # 読込
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """characters/ ディレクトリから全JSONを再読込。"""
        self._characters.clear()
        if not self.characters_dir.exists():
            return

        for p in sorted(self.characters_dir.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                char = Character.from_dict(data)
                self._characters[char.id] = char
                logger.debug("Loaded character: %s (%s)", char.name, char.id)
            except Exception as e:
                logger.warning("Failed to load character from %s: %s", p, e)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list_characters(self) -> list[Character]:
        """全キャラクターをリストで返す。"""
        return list(self._characters.values())

    def get_character(self, character_id: str) -> Optional[Character]:
        """IDでキャラクターを取得。"""
        return self._characters.get(character_id)

    def save_character(self, character: Character) -> Path:
        """キャラクターをJSONファイルに保存。"""
        filepath = self.characters_dir / f"{character.id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(character.to_dict(), f, ensure_ascii=False, indent=2)
        self._characters[character.id] = character
        logger.info("Saved character: %s -> %s", character.name, filepath)
        return filepath

    def delete_character(self, character_id: str) -> bool:
        """キャラクターを削除。"""
        if character_id not in self._characters:
            return False
        filepath = self.characters_dir / f"{character_id}.json"
        if filepath.exists():
            filepath.unlink()
        del self._characters[character_id]
        logger.info("Deleted character: %s", character_id)
        return True

    def create_character(
        self,
        name: str,
        system_prompt: str = "",
        tone: str = "",
        speech_style: str = "",
        background: str = "",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        repeat_penalty: float = 1.1,
        avatar: str = "",
    ) -> Character:
        """新しいキャラクターを作成して保存。"""
        char_id = str(uuid.uuid4())[:8]
        character = Character(
            id=char_id,
            name=name,
            avatar=avatar,
            personality=Personality(
                system_prompt=system_prompt,
                tone=tone,
                speech_style=speech_style,
                background=background,
            ),
            generation_params=GenerationParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repeat_penalty=repeat_penalty,
            ),
        )
        self.save_character(character)
        return character

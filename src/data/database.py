"""
Database: SQLiteによるチャット履歴の保存・読込・セッション管理。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# データ型
# ---------------------------------------------------------------------------

class ChatSession:
    """1つのチャットセッション。"""

    def __init__(
        self,
        session_id: str,
        character_id: str,
        character_name: str,
        model_name: str,
        created_at: str,
        updated_at: str,
        messages: list[dict] | None = None,
    ):
        self.session_id = session_id
        self.character_id = character_id
        self.character_name = character_name
        self.model_name = model_name
        self.created_at = created_at
        self.updated_at = updated_at
        self.messages = messages or []


# ---------------------------------------------------------------------------
# ChatDatabase
# ---------------------------------------------------------------------------

class ChatDatabase:
    """SQLiteベースのチャット履歴データベース。"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------

    def _init_db(self):
        """データベースとテーブルを初期化。"""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                character_name TEXT NOT NULL DEFAULT '',
                model_name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                character_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id);
        """)
        self._conn.commit()
        logger.info("Database initialized: %s", self.db_path)

    # ------------------------------------------------------------------
    # セッション操作
    # ------------------------------------------------------------------

    def create_session(
        self,
        character_id: str,
        character_name: str = "",
        model_name: str = "",
    ) -> ChatSession:
        """新しいチャットセッションを作成。"""
        session_id = str(uuid.uuid4())[:12]
        now = datetime.now().isoformat()

        self._conn.execute(
            """INSERT INTO sessions (session_id, character_id, character_name, model_name, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, character_id, character_name, model_name, now, now),
        )
        self._conn.commit()
        logger.info("Created session: %s", session_id)

        return ChatSession(
            session_id=session_id,
            character_id=character_id,
            character_name=character_name,
            model_name=model_name,
            created_at=now,
            updated_at=now,
        )

    def list_sessions(self, limit: int = 50) -> list[ChatSession]:
        """セッション一覧を新しい順に返す。"""
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

        return [
            ChatSession(
                session_id=r["session_id"],
                character_id=r["character_id"],
                character_name=r["character_name"],
                model_name=r["model_name"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        """セッションとそのメッセージを削除。"""
        self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        result = self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self._conn.commit()
        deleted = result.rowcount > 0
        if deleted:
            logger.info("Deleted session: %s", session_id)
        return deleted

    # ------------------------------------------------------------------
    # メッセージ操作
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        character_id: str = "",
    ) -> int:
        """メッセージを追加。"""
        now = datetime.now().isoformat()

        cursor = self._conn.execute(
            """INSERT INTO messages (session_id, role, content, character_id, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, role, content, character_id, now),
        )

        # セッションの updated_at を更新
        self._conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )
        self._conn.commit()

        return cursor.lastrowid

    def get_messages(self, session_id: str) -> list[dict]:
        """セッションの全メッセージを取得。"""
        rows = self._conn.execute(
            "SELECT role, content, character_id, created_at FROM messages WHERE session_id = ? ORDER BY message_id",
            (session_id,),
        ).fetchall()

        return [
            {
                "role": r["role"],
                "content": r["content"],
                "character_id": r["character_id"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def save_chat_history(
        self,
        session_id: str,
        messages: list[dict],
    ) -> None:
        """チャット履歴を一括保存（既存メッセージは削除して再挿入）。"""
        now = datetime.now().isoformat()

        self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

        for msg in messages:
            self._conn.execute(
                """INSERT INTO messages (session_id, role, content, character_id, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    session_id,
                    msg["role"],
                    msg["content"],
                    msg.get("character_id", ""),
                    msg.get("created_at", now),
                ),
            )

        self._conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # クリーンアップ
    # ------------------------------------------------------------------

    def close(self):
        """データベース接続を閉じる。"""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Database closed.")

    def __del__(self):
        self.close()

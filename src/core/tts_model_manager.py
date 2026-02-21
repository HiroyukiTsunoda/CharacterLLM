"""
TTSModelManager: SBV2 TTS モデルのダウンロード・一覧・パス解決を管理する。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

JVNV_REPO = "litagin/style_bert_vits2_jvnv"

KNOWN_MODELS: list[dict] = [
    {
        "id": "jvnv-F1-jp",
        "name": "JVNV 女性1",
        "repo": JVNV_REPO,
        "files": {
            "model": "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors",
            "config": "jvnv-F1-jp/config.json",
            "style": "jvnv-F1-jp/style_vectors.npy",
        },
    },
    {
        "id": "jvnv-F2-jp",
        "name": "JVNV 女性2",
        "repo": JVNV_REPO,
        "files": {
            "model": "jvnv-F2-jp/jvnv-F2_e166_s20000.safetensors",
            "config": "jvnv-F2-jp/config.json",
            "style": "jvnv-F2-jp/style_vectors.npy",
        },
    },
    {
        "id": "jvnv-M1-jp",
        "name": "JVNV 男性1",
        "repo": JVNV_REPO,
        "files": {
            "model": "jvnv-M1-jp/jvnv-M1-jp_e158_s14000.safetensors",
            "config": "jvnv-M1-jp/config.json",
            "style": "jvnv-M1-jp/style_vectors.npy",
        },
    },
    {
        "id": "jvnv-M2-jp",
        "name": "JVNV 男性2",
        "repo": JVNV_REPO,
        "files": {
            "model": "jvnv-M2-jp/jvnv-M2-jp_e159_s17000.safetensors",
            "config": "jvnv-M2-jp/config.json",
            "style": "jvnv-M2-jp/style_vectors.npy",
        },
    },
]


@dataclass
class LocalTTSModel:
    """ローカルに存在する TTS モデルの情報。"""
    id: str
    name: str
    dir_path: Path
    model_path: Path
    config_path: Path
    style_vec_path: Path

    def is_complete(self) -> bool:
        return (
            self.model_path.exists()
            and self.config_path.exists()
            and self.style_vec_path.exists()
        )


class TTSModelManager:
    """tts_models/ ディレクトリ内の TTS モデルを管理する。"""

    META_FILE = "model_info.json"

    def __init__(self, models_dir: str | Path = "tts_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # ローカルモデル一覧
    # ------------------------------------------------------------------

    def list_local_models(self) -> list[LocalTTSModel]:
        """tts_models/ 内のダウンロード済みモデルを列挙する。"""
        results: list[LocalTTSModel] = []
        if not self.models_dir.exists():
            return results

        for d in sorted(self.models_dir.iterdir()):
            if not d.is_dir():
                continue
            model = self._scan_model_dir(d)
            if model and model.is_complete():
                results.append(model)

        return results

    def get_local_model(self, model_id: str) -> Optional[LocalTTSModel]:
        """ID を指定してローカルモデルを取得する。"""
        d = self.models_dir / model_id
        if d.is_dir():
            m = self._scan_model_dir(d)
            if m and m.is_complete():
                return m
        return None

    def _scan_model_dir(self, d: Path) -> Optional[LocalTTSModel]:
        """ディレクトリ内からモデル3ファイルを検出する。"""
        safetensors = list(d.glob("*.safetensors"))
        configs = list(d.glob("config.json"))
        styles = list(d.glob("style_vectors.npy"))

        if not safetensors or not configs or not styles:
            return None

        meta_path = d / self.META_FILE
        name = d.name
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                name = meta.get("name", d.name)
            except Exception:
                pass

        return LocalTTSModel(
            id=d.name,
            name=name,
            dir_path=d,
            model_path=safetensors[0],
            config_path=configs[0],
            style_vec_path=styles[0],
        )

    # ------------------------------------------------------------------
    # ダウンロード可能モデル一覧
    # ------------------------------------------------------------------

    def list_downloadable(self) -> list[dict]:
        """ダウンロード可能な既知モデルの一覧を返す。

        ローカルに既に存在するものには downloaded=True が付く。
        """
        local_ids = {m.id for m in self.list_local_models()}
        result = []
        for entry in KNOWN_MODELS:
            info = dict(entry)
            info["downloaded"] = (entry["id"] in local_ids)
            result.append(info)
        return result

    # ------------------------------------------------------------------
    # ダウンロード
    # ------------------------------------------------------------------

    def download_model(self, model_id: str) -> LocalTTSModel:
        """既知モデルを HuggingFace からダウンロードする。

        ブロッキング呼び出し。UI 側では別スレッドで呼ぶこと。
        """
        entry = None
        for m in KNOWN_MODELS:
            if m["id"] == model_id:
                entry = m
                break

        if entry is None:
            raise ValueError(f"不明なモデル ID: {model_id}")

        from huggingface_hub import hf_hub_download

        dest_dir = self.models_dir / model_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        repo = entry["repo"]
        files = entry["files"]

        logger.info("Downloading TTS model '%s' from %s ...", model_id, repo)

        for key, remote_path in files.items():
            logger.info("  Downloading %s: %s", key, remote_path)
            hf_hub_download(
                repo,
                remote_path,
                local_dir=str(dest_dir),
            )

        # hf_hub_download はサブディレクトリ構造を維持するため、
        # ファイルを model_id/ 直下にフラット化する
        sub = dest_dir / model_id
        if sub.is_dir():
            for f in sub.iterdir():
                target = dest_dir / f.name
                if not target.exists():
                    f.rename(target)
            try:
                sub.rmdir()
            except OSError:
                pass

        meta_path = dest_dir / self.META_FILE
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"name": entry["name"], "repo": repo}, f, ensure_ascii=False, indent=2)

        logger.info("TTS model '%s' downloaded to %s", model_id, dest_dir)

        model = self._scan_model_dir(dest_dir)
        if model is None or not model.is_complete():
            raise RuntimeError(f"ダウンロード後のモデル検証に失敗: {dest_dir}")
        return model

    # ------------------------------------------------------------------
    # カスタムモデル登録
    # ------------------------------------------------------------------

    def register_custom(
        self,
        name: str,
        model_path: str | Path,
        config_path: str | Path,
        style_vec_path: str | Path,
    ) -> LocalTTSModel:
        """外部パスのモデルをシンボリックリンクまたはコピーで登録する。"""
        import shutil

        model_path = Path(model_path)
        config_path = Path(config_path)
        style_vec_path = Path(style_vec_path)

        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        dest = self.models_dir / safe_name
        dest.mkdir(parents=True, exist_ok=True)

        shutil.copy2(model_path, dest / model_path.name)
        shutil.copy2(config_path, dest / config_path.name)
        shutil.copy2(style_vec_path, dest / style_vec_path.name)

        meta_path = dest / self.META_FILE
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"name": name, "repo": "local"}, f, ensure_ascii=False, indent=2)

        model = self._scan_model_dir(dest)
        if model is None:
            raise RuntimeError(f"カスタムモデルの登録に失敗: {dest}")
        return model

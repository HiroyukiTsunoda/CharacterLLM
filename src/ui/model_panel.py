"""
ModelPanel: ローカルモデル一覧表示、HuggingFaceからのDL機能、モデル切替UI。
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.gpu_utils import check_vram_fit, get_primary_gpu_info
from src.core.model_loader import HFModelEntry, ModelLoader
from src.ui.styles import COLORS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ダウンロードワーカー
# ---------------------------------------------------------------------------

class DownloadWorker(QThread):
    """HuggingFaceからモデルをダウンロードするワーカー。"""

    progress_signal = Signal(int)   # パーセント
    finished_signal = Signal(str)   # ローカルパス
    error_signal = Signal(str)

    def __init__(self, model_loader: ModelLoader, repo_id: str, filename: str, parent=None):
        super().__init__(parent)
        self.model_loader = model_loader
        self.repo_id = repo_id
        self.filename = filename

    def run(self):
        try:
            local_path = self.model_loader.download_model(
                repo_id=self.repo_id,
                filename=self.filename,
            )
            self.finished_signal.emit(local_path)
        except Exception as e:
            logger.error("Download failed: %s", e)
            self.error_signal.emit(str(e))


# ---------------------------------------------------------------------------
# モデルロードワーカー
# ---------------------------------------------------------------------------

class LoadModelWorker(QThread):
    """モデルロードを別スレッドで実行。"""

    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, model_loader: ModelLoader, model_path: str, parent=None):
        super().__init__(parent)
        self.model_loader = model_loader
        self.model_path = model_path

    def run(self):
        try:
            self.model_loader.load_model(self.model_path)
            self.finished_signal.emit()
        except Exception as e:
            logger.error("Model load failed: %s", e)
            self.error_signal.emit(str(e))


# ---------------------------------------------------------------------------
# ModelPanel
# ---------------------------------------------------------------------------

class ModelPanel(QWidget):
    """サイドバーに配置するモデル選択・管理パネル。"""

    model_loaded = Signal(str)    # model_path
    model_unloaded = Signal()

    def __init__(self, model_loader: ModelLoader, parent=None):
        super().__init__(parent)
        self.model_loader = model_loader
        self._download_worker = None
        self._load_worker = None
        self._gpu_info: GpuInfo | None = None
        self._refresh_gpu_info()
        self._setup_ui()
        self.refresh_local_models()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ---- ヘッダー ----
        header = QLabel("モデル")
        header.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 15px; font-weight: bold;")
        layout.addWidget(header)

        # ---- VRAM 情報 ----
        self.vram_label = QLabel()
        self.vram_label.setWordWrap(True)
        self._update_vram_label()
        layout.addWidget(self.vram_label)

        # ---- 現在のモデル ----
        self.current_model_label = QLabel("未ロード")
        self.current_model_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        self.current_model_label.setWordWrap(True)
        layout.addWidget(self.current_model_label)

        # ---- ローカルモデル一覧 ----
        local_group = QGroupBox("ローカルモデル")
        local_layout = QVBoxLayout()

        self.model_list = QListWidget()
        self.model_list.setMaximumHeight(150)
        local_layout.addWidget(self.model_list)

        model_btn_layout = QHBoxLayout()

        self.load_btn = QPushButton("ロード")
        self.load_btn.clicked.connect(self._on_load)
        model_btn_layout.addWidget(self.load_btn)

        self.unload_btn = QPushButton("アンロード")
        self.unload_btn.setProperty("secondary", True)
        self.unload_btn.clicked.connect(self._on_unload)
        self.unload_btn.setEnabled(False)
        model_btn_layout.addWidget(self.unload_btn)

        self.delete_btn = QPushButton("削除")
        self.delete_btn.setProperty("secondary", True)
        self.delete_btn.clicked.connect(self._on_delete)
        model_btn_layout.addWidget(self.delete_btn)

        self.refresh_btn = QPushButton("更新")
        self.refresh_btn.setProperty("secondary", True)
        self.refresh_btn.clicked.connect(self.refresh_local_models)
        model_btn_layout.addWidget(self.refresh_btn)

        local_layout.addLayout(model_btn_layout)
        local_group.setLayout(local_layout)
        layout.addWidget(local_group)

        # ---- ダウンロード ----
        dl_group = QGroupBox("モデルダウンロード")
        dl_layout = QVBoxLayout()

        self.dl_combo = QComboBox()
        self._populate_download_combo()
        dl_layout.addWidget(self.dl_combo)

        self.dl_btn = QPushButton("ダウンロード")
        self.dl_btn.clicked.connect(self._on_download)
        dl_layout.addWidget(self.dl_btn)

        self.dl_progress = QProgressBar()
        self.dl_progress.setVisible(False)
        dl_layout.addWidget(self.dl_progress)

        self.dl_status = QLabel("")
        self.dl_status.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        self.dl_status.setWordWrap(True)
        dl_layout.addWidget(self.dl_status)

        dl_group.setLayout(dl_layout)
        layout.addWidget(dl_group)

        layout.addStretch()

    # ------------------------------------------------------------------
    # 状態ヘルパー
    # ------------------------------------------------------------------

    def _is_downloading(self) -> bool:
        return self._download_worker is not None and self._download_worker.isRunning()

    def _is_loading(self) -> bool:
        return self._load_worker is not None and self._load_worker.isRunning()

    def _is_busy(self) -> bool:
        return self._is_downloading() or self._is_loading()

    # ------------------------------------------------------------------
    # ローカルモデル一覧
    # ------------------------------------------------------------------

    def refresh_local_models(self):
        """ローカルモデル一覧を更新。VRAM 判定結果も反映。"""
        self.model_list.clear()
        self._refresh_gpu_info()
        self._update_vram_label()

        models = self.model_loader.list_local_models()
        if not models:
            item = QListWidgetItem("(モデルなし — models/ にGGUFファイルを配置)")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            item.setForeground(Qt.GlobalColor.gray)
            self.model_list.addItem(item)
            return

        free_gb = self._gpu_info.free_vram_gb if self._gpu_info else -1

        for m in models:
            fit = check_vram_fit(m.estimated_vram_gb, free_gb)
            indicator = self._fit_indicator(fit)
            label = f"{indicator} {m.name}  ({m.size_gb:.1f} GB)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, m.path)
            color = self._fit_color(fit)
            if color:
                item.setForeground(QColor(color))
            self.model_list.addItem(item)

    # ------------------------------------------------------------------
    # ロード / アンロード
    # ------------------------------------------------------------------

    def _on_load(self):
        """選択されたモデルをロードする。"""
        current = self.model_list.currentItem()
        if not current:
            return
        model_path = current.data(Qt.ItemDataRole.UserRole)
        if not model_path:
            return

        if self._is_downloading():
            QMessageBox.warning(
                self, "操作不可",
                "ダウンロード中はモデルをロードできません。\n"
                "ダウンロード完了後に再度お試しください。",
            )
            return
        if self._is_loading():
            return

        # 別モデルがロード済みなら先にアンロード（VRAM解放 + シグナル通知）
        if self.model_loader.current_model_path:
            self._on_unload()

        self.load_btn.setEnabled(False)
        self.load_btn.setText("ロード中...")
        self.dl_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.current_model_label.setText("ロード中...")

        self._load_worker = LoadModelWorker(self.model_loader, model_path, self)
        self._load_worker.finished_signal.connect(lambda: self._on_model_loaded(model_path))
        self._load_worker.error_signal.connect(self._on_model_load_error)
        self._load_worker.start()

    def _on_model_loaded(self, model_path: str):
        """モデルロード完了。"""
        name = model_path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        self.current_model_label.setText(f"ロード済み: {name}")
        self.current_model_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px;")
        self.load_btn.setEnabled(True)
        self.load_btn.setText("ロード")
        self.unload_btn.setEnabled(True)
        self.dl_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self._refresh_gpu_info()
        self._update_vram_label()
        self.model_loaded.emit(model_path)

    def _on_model_load_error(self, error: str):
        """モデルロードエラー。"""
        self.current_model_label.setText(f"エラー: {error}")
        self.current_model_label.setStyleSheet(f"color: {COLORS['error']}; font-size: 12px;")
        self.load_btn.setEnabled(True)
        self.load_btn.setText("ロード")
        self.dl_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        QMessageBox.critical(self, "モデルロードエラー", f"モデルのロードに失敗しました:\n{error}")

    def _on_unload(self):
        """モデルをアンロード。"""
        if self._is_loading():
            QMessageBox.warning(self, "操作不可", "モデルロード中はアンロードできません。")
            return
        self.model_loader.unload_model()
        self.current_model_label.setText("未ロード")
        self.current_model_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        self.unload_btn.setEnabled(False)
        self._refresh_gpu_info()
        self._update_vram_label()
        self.model_unloaded.emit()

    def _on_delete(self):
        """選択されたモデルファイルを削除する。"""
        if self._is_busy():
            QMessageBox.warning(self, "操作不可", "ダウンロードまたはロード中はモデルを削除できません。")
            return

        current = self.model_list.currentItem()
        if not current:
            return
        model_path = current.data(Qt.ItemDataRole.UserRole)
        if not model_path:
            return

        p = Path(model_path)
        size_gb = p.stat().st_size / (1024 ** 3) if p.exists() else 0

        reply = QMessageBox.question(
            self,
            "モデル削除確認",
            f"以下のモデルファイルを削除しますか？\n\n"
            f"{p.name}\n"
            f"サイズ: {size_gb:.1f} GB\n\n"
            f"この操作は元に戻せません。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # ロード中のモデルなら先にアンロード
        if self.model_loader.current_model_path and Path(self.model_loader.current_model_path) == p:
            self._on_unload()

        try:
            p.unlink()
            logger.info("Deleted model: %s", p)
            self.refresh_local_models()
        except Exception as e:
            logger.error("Failed to delete model: %s", e)
            QMessageBox.critical(self, "削除エラー", f"モデルの削除に失敗しました:\n{e}")

    # ------------------------------------------------------------------
    # ダウンロード
    # ------------------------------------------------------------------

    def _on_download(self):
        """おすすめモデルをダウンロード。"""
        entry: HFModelEntry = self.dl_combo.currentData()
        if not entry:
            return

        if self._is_loading():
            QMessageBox.warning(
                self, "操作不可",
                "モデルロード中はダウンロードできません。\n"
                "ロード完了後に再度お試しください。",
            )
            return
        if self._is_downloading():
            return

        reply = QMessageBox.question(
            self,
            "ダウンロード確認",
            f"{entry.display_name} をダウンロードしますか？\n推定VRAM: {entry.estimated_vram_gb:.1f} GB",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.dl_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.dl_progress.setVisible(True)
        self.dl_progress.setRange(0, 0)  # indeterminate
        self.dl_status.setText(f"ダウンロード中: {entry.filename} ...")

        self._download_worker = DownloadWorker(self.model_loader, entry.repo_id, entry.filename, self)
        self._download_worker.finished_signal.connect(self._on_download_finished)
        self._download_worker.error_signal.connect(self._on_download_error)
        self._download_worker.start()

    def _on_download_finished(self, local_path: str):
        """ダウンロード完了。"""
        self.dl_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self.dl_progress.setVisible(False)
        self.dl_status.setText(f"完了: {local_path}")
        self.dl_status.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px;")
        self.refresh_local_models()

    def _on_download_error(self, error: str):
        """ダウンロードエラー。"""
        self.dl_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self.dl_progress.setVisible(False)
        self.dl_status.setText(f"エラー: {error}")
        self.dl_status.setStyleSheet(f"color: {COLORS['error']}; font-size: 12px;")
        QMessageBox.critical(self, "ダウンロードエラー", f"ダウンロードに失敗しました:\n{error}")

    # ------------------------------------------------------------------
    # VRAM 判定ヘルパー
    # ------------------------------------------------------------------

    def _refresh_gpu_info(self):
        """GPU 情報を取得・更新する。"""
        gpu_index = self.model_loader.config.get("gpu", {}).get("main_gpu", 0)
        self._gpu_info = get_primary_gpu_info(gpu_index)

    def _update_vram_label(self):
        """VRAM ステータスラベルを更新。"""
        gpu = self._gpu_info
        if gpu is None:
            self.vram_label.setText("GPU: 検出できません")
            self.vram_label.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-size: 12px;"
            )
            return

        self.vram_label.setText(
            f"GPU: {gpu.name}\n"
            f"VRAM: {gpu.free_vram_gb:.1f} GB 空き / {gpu.total_vram_gb:.1f} GB 合計"
        )
        self.vram_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px;"
        )

    def _populate_download_combo(self):
        """おすすめモデルのコンボボックスを VRAM 判定付きで構築。"""
        self.dl_combo.clear()
        free_gb = self._gpu_info.free_vram_gb if self._gpu_info else -1

        for entry in self.model_loader.get_recommended_models():
            fit = check_vram_fit(entry.estimated_vram_gb, free_gb)
            indicator = self._fit_indicator(fit)
            label = f"{indicator} {entry.display_name} (~{entry.estimated_vram_gb:.0f}GB)"
            self.dl_combo.addItem(label, entry)

    @staticmethod
    def _fit_indicator(fit: str) -> str:
        """VRAM 判定結果に対応するテキストインジケータ。"""
        return {
            "ok": "[OK]",
            "tight": "[!!]",
            "over": "[NG]",
            "unknown": "",
        }.get(fit, "")

    @staticmethod
    def _fit_color(fit: str) -> str | None:
        """VRAM 判定結果に対応する色。"""
        return {
            "ok": COLORS["success"],
            "tight": COLORS["warning"],
            "over": COLORS["error"],
        }.get(fit)

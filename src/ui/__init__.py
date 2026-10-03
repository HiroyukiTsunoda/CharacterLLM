from PySide6.QtCore import QEvent, QObject, Qt


class WheelGuard(QObject):
    """フォーカスされていないウィジェットのホイールイベントを無視する。

    QComboBox / QSpinBox 等をスクロール領域内に配置すると、
    マウスホイールで値が勝手に変わりスクロールが止まる問題を防ぐ。
    """

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)


def install_wheel_guard(parent: QObject, *widgets) -> None:
    """widgets に WheelGuard をまとめてインストールする。"""
    guard = WheelGuard(parent)
    for w in widgets:
        w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        w.installEventFilter(guard)

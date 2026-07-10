"""Reusable UI widgets for ScriptedElite."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget


def badge(initial: str = "—") -> QLabel:
    lab = QLabel(initial)
    lab.setObjectName("valueBadge")
    lab.setAlignment(Qt.AlignCenter)
    return lab


def slider_row(title: str, tip: str, slider: QSlider, value_badge: QLabel) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 2, 0, 2)
    lay.setSpacing(3)
    top = QHBoxLayout()
    t = QLabel(title)
    t.setStyleSheet("font-weight: 700;")
    t.setToolTip(tip)
    top.addWidget(t)
    top.addStretch()
    top.addWidget(value_badge)
    lay.addLayout(top)
    slider.setToolTip(tip)
    lay.addWidget(slider)
    return w


def mini_slider(slider: QSlider, value_badge: QLabel) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(slider, 1)
    h.addWidget(value_badge)
    return w

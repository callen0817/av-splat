"""AV Scan / scanAR glass tokens, with legible desktop surface opacity."""
from pathlib import Path
import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import QSplashScreen

GOLD='#BAA164'
BRIGHT='#E5C983'
WHITE='#FCFBF8'
STYLE='''
QMainWindow, QDialog { background: #15171c; }
QWidget { color: #FCFBF8; font-family: "Inter", "Segoe UI", sans-serif; font-size: 13px; }
QFrame#glass, QGroupBox { background: rgba(42, 44, 50, 238); border: 1px solid rgba(186,161,100,130); border-radius: 16px; }
QLabel#eyebrow { color: #E5C983; font-size: 11px; font-weight: 600; }
QLabel#title { font-size: 24px; font-weight: 600; }
QLabel#muted { color: #D0CCC2; }
QPushButton, QToolButton { background: rgba(255,255,255,18); border: 1px solid rgba(255,255,255,55); border-radius: 9px; padding: 9px 12px; }
QPushButton:hover, QToolButton:hover { background: rgba(186,161,100,55); border-color: #BAA164; }
QPushButton:focus, QToolButton:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus { border: 2px solid #E5C983; }
QPushButton:disabled { color: #807e78; border-color: #48484a; background: #25262b; }
QPushButton#primary { background: #BAA164; color: #15171c; font-weight: 600; }
QPushButton#primary:hover { background: #E5C983; }
QPushButton#primary:disabled { background: #3c3a32; color: #9e9785; border-color: #554e3e; }
QTextEdit, QPlainTextEdit, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QListWidget, QTableWidget { background: #202228; border: 1px solid #555047; border-radius: 7px; padding: 6px; selection-background-color: #75643e; }
QHeaderView::section { background: #292b32; color: #E5C983; padding: 6px; }
QProgressBar { border: 1px solid #555047; border-radius: 5px; background: #202228; text-align: center; }
QProgressBar::chunk { background: #BAA164; border-radius: 4px; }
QSplitter::handle { background: transparent; width: 8px; height: 8px; }
QCheckBox { spacing: 7px; padding: 3px; }
QScrollArea { border: none; background: transparent; }
QComboBox QAbstractItemView { background: #202228; color: #FCFBF8; selection-background-color: #75643e; border: 1px solid #BAA164; }
QScrollBar:vertical { background: #202228; width: 9px; border: none; }
QScrollBar::handle:vertical { background: #625b49; min-height: 24px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QToolTip { background: #292b32; color: #FCFBF8; border: 1px solid #BAA164; }
'''


def asset(name='av_splat.png'):
    root=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent.parent))
    return root/'logos'/name


def icon():
    return QIcon(str(asset()))


class BrandSplash(QSplashScreen):
    def __init__(self):
        from PySide6.QtGui import QColor, QPainter, QPen, QFont
        pixmap=QPixmap(560,360); pixmap.fill(Qt.transparent)
        painter=QPainter(pixmap); painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(27,29,35,248)); painter.setPen(QPen(QColor(GOLD),1))
        painter.drawRoundedRect(1,1,558,358,22,22)
        logo=QPixmap(str(asset()))
        if not logo.isNull():
            painter.setBrush(QColor(WHITE)); painter.setPen(QPen(QColor(GOLD),1))
            painter.drawRoundedRect(184,24,192,184,16,16)
            logo=logo.scaled(170,170,Qt.KeepAspectRatio,Qt.SmoothTransformation)
            painter.drawPixmap((560-logo.width())//2,31+(170-logo.height())//2,logo)
        painter.setPen(QColor(WHITE)); painter.setFont(QFont('Segoe UI',24))
        painter.drawText(0,210,560,40,Qt.AlignCenter,'AV Splat')
        painter.setFont(QFont('Segoe UI',11)); painter.setPen(QColor(BRIGHT))
        painter.drawText(0,256,560,30,Qt.AlignCenter,'An Artificial Vision Product.')
        painter.end()
        super().__init__(pixmap,Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowIcon(icon())

    def status(self,message):
        from PySide6.QtGui import QColor
        self.showMessage(message,Qt.AlignBottom|Qt.AlignHCenter,QColor(WHITE))
        self.repaint()

#!/usr/bin/env python3
"""
YOLO Model Inspector
Üç YOLO modelini yan yana karşılaştırarak fotoğraf inceleme ve hata işaretleme aracı.
Özel Renk Paleti, Kutu Sayacı, Kısayol Kontrollü ROI Görünürlüğü, Boşluk ile Silme 
ve 'Arada Kalanlar (4)' Sınıflandırması. Toast mesajı en üste taşınmıştır.
"""

import sys
import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QFileDialog, QSlider, QStatusBar,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QDoubleSpinBox,
    QFrame, QMessageBox, QShortcut
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QPixmap, QImage, QKeySequence, QPainter, QColor, QWheelEvent
)

import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("UYARI: ultralytics yüklü değil. Model inference devre dışı.")


# ─────────────────────────── Data Model ───────────────────────────

@dataclass
class PhotoInfo:
    """Tek bir fotoğrafın metadata bilgisi."""
    file_path: str
    camera_id: str
    event_id: str
    year: int = 0
    month: int = 0
    day: int = 0
    hour: int = 0
    minute: int = 0
    second: int = 0
    millisecond: int = 0
    frame_id: str = ""

    @property
    def time_tuple(self) -> Tuple:
        return (self.hour, self.minute, self.second, self.millisecond, self.camera_id)

    @property
    def time_str(self) -> str:
        if self.year > 0:
            return f"{self.hour:02d}:{self.minute:02d}:{self.second:02d}.{self.millisecond:03d}"
        return "Zaman Bilgisi Yok"

    @property
    def filename(self) -> str:
        return os.path.basename(self.file_path)


# ─────────────────────────── Photo Scanner ───────────────────────────

class PhotoScanner:
    """Ana klasörü tarar, farklı uygulama ve klasör yapılarına uyum sağlar."""

    @staticmethod
    def parse_filename(filename: str) -> Optional[dict]:
        name = os.path.splitext(filename)[0]
        parts = name.split('_')
        if len(parts) < 8:
            return None
        try:
            return {
                'year': int(parts[0]),
                'month': int(parts[1]),
                'day': int(parts[2]),
                'hour': int(parts[3]),
                'minute': int(parts[4]),
                'second': int(parts[5]),
                'millisecond': int(parts[6]),
                'frame_id': '_'.join(parts[7:])
            }
        except (ValueError, IndexError):
            return None

    @staticmethod
    def scan_folder(main_folder: str) -> List[PhotoInfo]:
        photos = []
        main_path = Path(main_folder)
        if not main_path.exists():
            print(f"[HATA] Klasör bulunamadı: {main_path}")
            return photos

        print(f"[TARAMA] Klasör taranıyor: {main_path}")
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        img_files = [f for f in main_path.rglob('*') if f.is_file() and f.suffix.lower() in valid_extensions]

        for file_path in img_files:
            parsed = PhotoScanner.parse_filename(file_path.name)
            
            rel_parts = file_path.relative_to(main_path).parts
            camera_id = rel_parts[0] if len(rel_parts) > 1 else "Genel"
            event_id = rel_parts[1] if len(rel_parts) > 2 else "Varsayilan"

            if parsed:
                photos.append(PhotoInfo(
                    file_path=str(file_path),
                    camera_id=camera_id,
                    event_id=event_id,
                    **parsed
                ))
            else:
                photos.append(PhotoInfo(
                    file_path=str(file_path),
                    camera_id=camera_id,
                    event_id=event_id,
                    frame_id=file_path.stem
                ))

        photos.sort(key=lambda p: p.time_tuple if p.year > 0 else (0, p.filename))
        print(f"[SONUÇ] Toplam {len(photos)} fotoğraf bulundu ve sıralandı.")
        return photos


# ─────────────────────────── Toast Notification ───────────────────────────

class ToastNotification(QLabel):
    """Ekranda kısa süre görünen belirgin bildirim mesajı."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        self.hide()

    def show_message(self, message: str, color: str = "#64ffda", duration_ms: int = 2500):
        self.setText(message)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: rgba(10, 12, 30, 230);
                color: {color};
                border: 2px solid {color};
                border-radius: 12px;
                padding: 14px 28px;
                font-size: 15px;
                font-weight: bold;
            }}
        """)
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()
        self._timer.start(duration_ms)

    def _reposition(self):
        if self.parent():
            pr = self.parent().rect()
            self.move((pr.width() - self.width()) // 2, 10)


# ─────────────────────────── Sync Graphics View ───────────────────────────

class SyncGraphicsView(QGraphicsView):
    """Zoom/pan destekli ve senkronize edilebilir görüntü alanı."""
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom = 1.0
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("background-color: #0d1020;")

    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._zoom = 1.0

    def set_absolute_zoom(self, zoom_level: float):
        if abs(self._zoom - zoom_level) > 0.001:
            factor = zoom_level / self._zoom
            self._zoom = zoom_level
            self.scale(factor, factor)

    def wheelEvent(self, event: QWheelEvent):
        pass

    def sync_zoom(self, zoom_level: float):
        if abs(self._zoom - zoom_level) > 0.001:
            factor = zoom_level / self._zoom
            self._zoom = zoom_level
            self.scale(factor, factor)

    def reset_zoom(self):
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._zoom = 1.0


# ─────────────────────────── Model Panel ───────────────────────────

class ModelPanel(QWidget):
    """Tek bir YOLO modeli paneli: model yükleme + custom inference gösterimi."""

    def __init__(self, panel_id: int, parent=None):
        super().__init__(parent)
        self.panel_id = panel_id
        self.model = None
        self.model_path = ""
        self.confidence = 0.25
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()

        self.load_btn = QPushButton(f"📦 Model {self.panel_id}")
        self.load_btn.setFixedHeight(32)
        self.load_btn.setCursor(Qt.PointingHandCursor)
        self.load_btn.clicked.connect(self.load_model)
        header.addWidget(self.load_btn)

        self.model_label = QLabel("Model yüklenmedi")
        self.model_label.setStyleSheet("color: #8892b0; font-size: 11px;")
        self.model_label.setWordWrap(True)
        header.addWidget(self.model_label, 1)

        conf_lbl = QLabel("Conf:")
        conf_lbl.setStyleSheet("color: #ccd6f6; font-size: 11px;")
        header.addWidget(conf_lbl)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setFixedWidth(55)
        self.conf_spin.valueChanged.connect(self._on_conf_changed)
        header.addWidget(self.conf_spin)

        layout.addLayout(header)

        self.view = SyncGraphicsView()
        layout.addWidget(self.view, 1)

        # Sağ alta yerleşecek Kutu Sayacı çubuğu
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()
        self.box_count_lbl = QLabel("Kutu Sayısı: 0")
        self.box_count_lbl.setStyleSheet("""
            color: #64ffda; 
            font-size: 11px; 
            font-weight: bold; 
            background-color: #1a1a3e;
            padding: 4px 8px;
            border-radius: 4px;
        """)
        bottom_bar.addWidget(self.box_count_lbl)
        layout.addLayout(bottom_bar)

    def _on_conf_changed(self, value):
        self.confidence = value

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Model {self.panel_id} Seç", "",
            "YOLO Model (*.pt);;Tüm Dosyalar (*)"
        )
        if path:
            self.model_path = path
            self.model_label.setText(os.path.basename(path))
            self.model_label.setToolTip(path)
            self.model_label.setStyleSheet("color: #64ffda; font-size: 11px;")
            if YOLO_AVAILABLE:
                try:
                    self.model = YOLO(path)
                    self.load_btn.setStyleSheet(
                        "background-color: #0d2818; color: #64ffda; border: 1px solid #2dd4a0;"
                    )
                except Exception as e:
                    QMessageBox.warning(self, "Hata", f"Model yüklenemedi:\n{e}")
                    self.model = None

    def run_inference(self, image_path: str, run_model: bool = True, show_flags: dict = None) -> Tuple[QPixmap, int]:
        if show_flags is None:
            show_flags = {"kapakli": True, "sivi-yok": True, "sivi-var": True}

        if self.model is None or not run_model or not os.path.exists(image_path):
            pix = QPixmap(image_path) if image_path else self._empty_pixmap()
            return (pix if not pix.isNull() else self._empty_pixmap()), 0

        try:
            results = self.model(image_path, conf=self.confidence, verbose=False)
            result = results[0]
            
            img = result.orig_img.copy()
            boxes = result.boxes
            names = result.names
            drawn_box_count = 0

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = names[cls_id].lower()

                    if "kapakli" in cls_name:
                        if not show_flags.get("kapakli", True): continue
                        color = (255, 0, 0)       
                        text_color = (255, 255, 255) 
                    elif "sivi-yok" in cls_name or "siviyok" in cls_name:
                        if not show_flags.get("sivi-yok", True): continue
                        color = (0, 0, 255)       
                        text_color = (255, 255, 255) 
                    elif "sivi-var" in cls_name or "sivivar" in cls_name:
                        if not show_flags.get("sivi-var", True): continue
                        color = (0, 255, 255)     
                        text_color = (0, 0, 0)       
                    else:
                        color = (0, 255, 0)       
                        text_color = (0, 0, 0)       
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    
                    drawn_box_count += 1

            annotated = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_img, w_img, ch = annotated.shape
            qimg = QImage(annotated.data, w_img, h_img, ch * w_img, QImage.Format_RGB888)
            
            return QPixmap.fromImage(qimg), drawn_box_count
            
        except Exception as e:
            print(f"Inference hatası: {e}")
            pix = QPixmap(image_path)
            return (pix if not pix.isNull() else self._empty_pixmap()), 0

    def display_image(self, image_path: str, run_model: bool = True, show_flags: dict = None):
        pixmap, box_count = self.run_inference(image_path, run_model, show_flags)
        self.view.set_pixmap(pixmap)
        
        if run_model and self.model is not None and image_path:
            self.box_count_lbl.setText(f"Kutu Sayısı: {box_count}")
        else:
            self.box_count_lbl.setText("Kutu Sayısı: 0")

    @staticmethod
    def _empty_pixmap() -> QPixmap:
        p = QPixmap(640, 480)
        p.fill(QColor("#0d1020"))
        return p


# ─────────────────────────── Main Window ───────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.photos: List[PhotoInfo] = []
        self.current_index = 0
        self.main_folder = ""
        
        self.show_kapakli = True
        self.show_sivi_yok = True
        self.show_sivi_var = True

        self.setWindowTitle("YOLO Model Inspector")
        self.setMinimumSize(1400, 800)
        self._build_ui()
        self._bind_shortcuts()
        self._apply_theme()
        self.showMaximized()
        self.last_action_path = None
        self.inference_enabled = True

    # ── UI Construction ──

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        top = QHBoxLayout()
        self.folder_btn = QPushButton("📁 Klasör Seç")
        self.folder_btn.setFixedHeight(36)
        self.folder_btn.setCursor(Qt.PointingHandCursor)
        self.folder_btn.clicked.connect(self.select_folder)
        top.addWidget(self.folder_btn)

        self.folder_label = QLabel("Klasör seçilmedi")
        self.folder_label.setStyleSheet("color: #8892b0; font-size: 12px;")
        top.addWidget(self.folder_label)

        top.addSpacing(30)
        self.zoom_lbl = QLabel("🔍 Zoom:")
        self.zoom_lbl.setStyleSheet("color: #ccd6f6; font-size: 12px; font-weight: bold;")
        top.addWidget(self.zoom_lbl)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(150)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        top.addWidget(self.zoom_slider)

        self.zoom_val_lbl = QLabel("%100")
        self.zoom_val_lbl.setStyleSheet("color: #64ffda; font-size: 12px;")
        self.zoom_val_lbl.setFixedWidth(45)
        top.addWidget(self.zoom_val_lbl)
        top.addStretch(1)
        
        self.shortcuts_lbl = QLabel("Kısayollar: (K) Kapaklı | (Y) Sıvı Yok | (V) Sıvı Var | (Boşluk) Sil")
        self.shortcuts_lbl.setStyleSheet("color: #8892b0; font-size: 11px; margin-right: 15px;")
        top.addWidget(self.shortcuts_lbl)

        self.model_status_lbl = QLabel("🧠 Model: AÇIK")
        self.model_status_lbl.setStyleSheet("color: #64ffda; font-size: 12px; font-weight: bold; margin-right: 15px;")
        top.addWidget(self.model_status_lbl)
        self.exit_btn = QPushButton("Çıkış")
        self.exit_btn.setFixedHeight(30)
        self.exit_btn.setCursor(Qt.PointingHandCursor)
        self.exit_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #1a1a3e; 
                        color: #ff6b6b; 
                        border: 1px solid #303060;
                        border-radius: 4px; 
                        padding: 4px 12px; 
                        font-size: 12px; 
                        font-weight: bold;
                    }
                    QPushButton:hover { 
                        background-color: #2a2a5e; 
                        border-color: #ff6b6b; 
                        color: #ff8b8b;
                    }
                    QPushButton:pressed { 
                        background-color: #3a3a7e; 
                    }
                """)

        self.exit_btn.clicked.connect(self.close)
        top.addWidget(self.exit_btn)
        top.addSpacing(10)

        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: #64ffda; font-size: 13px; font-weight: bold;")
        top.addWidget(self.count_label)
        root.addLayout(top)

        self.splitter = QSplitter(Qt.Horizontal)
        self.panel1 = ModelPanel(1)
        self.panel2 = ModelPanel(2)
        self.panel3 = ModelPanel(3) 

        self.splitter.addWidget(self.panel1)
        self.splitter.addWidget(self.panel2)
        self.splitter.addWidget(self.panel3)
        self.splitter.setSizes([450, 450, 450])

        for p_source in (self.panel1, self.panel2, self.panel3):
            for p_target in (self.panel1, self.panel2, self.panel3):
                if p_source != p_target:
                    p_source.view.horizontalScrollBar().valueChanged.connect(
                        p_target.view.horizontalScrollBar().setValue
                    )
                    p_source.view.verticalScrollBar().valueChanged.connect(
                        p_target.view.verticalScrollBar().setValue
                    )
                    
        self.panel1.view.zoom_changed.connect(self.panel2.view.sync_zoom)
        self.panel1.view.zoom_changed.connect(self.panel3.view.sync_zoom)
        self.panel2.view.zoom_changed.connect(self.panel1.view.sync_zoom)
        self.panel2.view.zoom_changed.connect(self.panel3.view.sync_zoom)
        self.panel3.view.zoom_changed.connect(self.panel1.view.sync_zoom)
        self.panel3.view.zoom_changed.connect(self.panel2.view.sync_zoom)
        
        root.addWidget(self.splitter, 1)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Önceki")
        self.prev_btn.setFixedSize(100, 36)
        self.prev_btn.setCursor(Qt.PointingHandCursor)
        self.prev_btn.clicked.connect(self.prev_photo)
        nav.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Sonraki ▶")
        self.next_btn.setFixedSize(100, 36)
        self.next_btn.setCursor(Qt.PointingHandCursor)
        self.next_btn.clicked.connect(self.next_photo)
        nav.addWidget(self.next_btn)

        nav.addSpacing(16)
        self.idx_label = QLabel("0 / 0")
        self.idx_label.setStyleSheet("color: #ccd6f6; font-size: 14px; font-weight: bold;")
        self.idx_label.setFixedWidth(140)
        self.idx_label.setAlignment(Qt.AlignCenter)
        nav.addWidget(self.idx_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._slider_moved)
        nav.addWidget(self.slider, 1)
        root.addLayout(nav)

        info = QHBoxLayout()
        self.cam_lbl = QLabel("📷 Kategori 1: -")
        self.time_lbl = QLabel("⏰ Zaman: -")
        self.frame_lbl = QLabel("🆔 Dosya: -")
        self.event_lbl = QLabel("📂 Kategori 2: -")
        for lbl in (self.cam_lbl, self.time_lbl, self.frame_lbl, self.event_lbl):
            lbl.setStyleSheet("color: #e6f1ff; font-size: 12px;")
            info.addWidget(lbl)
        info.addStretch()
        self.fname_lbl = QLabel("")
        self.fname_lbl.setStyleSheet("color: #8892b0; font-size: 11px;")
        info.addWidget(self.fname_lbl)
        root.addLayout(info)

        err = QHBoxLayout()
        self.day_btn = QPushButton("❌ Hatalı (1)")
        self.day_btn.setFixedHeight(40)
        self.day_btn.setCursor(Qt.PointingHandCursor)
        self.day_btn.setObjectName("dayBtn")
        self.day_btn.clicked.connect(lambda: self.mark_error("hatali"))
        err.addWidget(self.day_btn)

        self.night_btn = QPushButton("⚠️ Yarı Hatalı (2)")
        self.night_btn.setFixedHeight(40)
        self.night_btn.setCursor(Qt.PointingHandCursor)
        self.night_btn.setObjectName("nightBtn")
        self.night_btn.clicked.connect(lambda: self.mark_error("yari_hatali"))
        err.addWidget(self.night_btn)

        self.human_btn = QPushButton("✅ Hatasız (3)")
        self.human_btn.setFixedHeight(40)
        self.human_btn.setCursor(Qt.PointingHandCursor)
        self.human_btn.setObjectName("humanBtn")
        self.human_btn.clicked.connect(lambda: self.mark_error("hatasiz"))
        err.addWidget(self.human_btn)

        self.unknown_btn = QPushButton("❔ Arada Kalanlar (4)")
        self.unknown_btn.setFixedHeight(40)
        self.unknown_btn.setCursor(Qt.PointingHandCursor)
        self.unknown_btn.setObjectName("unknownBtn")
        self.unknown_btn.clicked.connect(lambda: self.mark_error("arada_kalanlar"))
        err.addWidget(self.unknown_btn)

        root.addLayout(err)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Hazır — Klasör seçerek başlayın")

        self.toast = ToastNotification(central)

    # ── Shortcuts ──

    def _bind_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_photo)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_photo)
        QShortcut(QKeySequence(Qt.Key_Home), self, self.go_first)
        QShortcut(QKeySequence(Qt.Key_End), self, self.go_last)
        
        QShortcut(QKeySequence(Qt.Key_1), self, lambda: self.mark_error("hatali"))
        QShortcut(QKeySequence(Qt.Key_2), self, lambda: self.mark_error("yari_hatali"))
        QShortcut(QKeySequence(Qt.Key_3), self, lambda: self.mark_error("hatasiz"))
        QShortcut(QKeySequence(Qt.Key_4), self, lambda: self.mark_error("arada_kalanlar"))
        
        QShortcut(QKeySequence(Qt.Key_Z), self, self.undo_last_action)
        QShortcut(QKeySequence(Qt.Key_M), self, self.toggle_inference)
        QShortcut(QKeySequence("Ctrl+O"), self, self.select_folder)
        QShortcut(QKeySequence("Ctrl+1"), self, self.panel1.load_model)
        QShortcut(QKeySequence("Ctrl+2"), self, self.panel2.load_model)
        QShortcut(QKeySequence("Ctrl+3"), self, self.panel3.load_model)
        QShortcut(QKeySequence("Ctrl+0"), self, self._reset_zoom)
        
        QShortcut(QKeySequence(Qt.Key_K), self, self.toggle_kapakli)
        QShortcut(QKeySequence(Qt.Key_Y), self, self.toggle_sivi_yok)
        QShortcut(QKeySequence(Qt.Key_V), self, self.toggle_sivi_var)
        
        QShortcut(QKeySequence(Qt.Key_Space), self, self.delete_current_photo)

    # ── Photo Deletion ──

    def delete_current_photo(self):
        if not self.photos:
            return

        photo_to_delete = self.photos[self.current_index]
        file_path = photo_to_delete.file_path

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            
            self.photos.pop(self.current_index)
            
            n = len(self.photos)
            self.count_label.setText(f"📸 {n} fotoğraf")
            self.slider.setMaximum(max(0, n - 1))
            
            self.toast.show_message(f"🗑️ Fotoğraf Silindi:\n{photo_to_delete.filename}", color="#ff5555")
            self.status.showMessage(f"🗑️ Silindi: {photo_to_delete.filename}")

            if not self.photos:
                self.current_index = 0
                self._clear_info()
                self.panel1.display_image("", False)
                self.panel2.display_image("", False)
                self.panel3.display_image("", False)
            else:
                if self.current_index >= len(self.photos):
                    self.current_index = len(self.photos) - 1
                self.show_current()

        except Exception as e:
            self.toast.show_message(f"❌ Silme hatası:\n{e}", color="#ff5555")
            self.status.showMessage(f"❌ Silme hatası: {e}")

    # ── ROI Toggle Methods ──
    
    def toggle_kapakli(self):
        self.show_kapakli = not self.show_kapakli
        durum = "AÇIK" if self.show_kapakli else "KAPALI"
        self.toast.show_message(f"Kapaklı Kutuları: {durum}", color="#64ffda")
        self.show_current()

    def toggle_sivi_yok(self):
        self.show_sivi_yok = not self.show_sivi_yok
        durum = "AÇIK" if self.show_sivi_yok else "KAPALI"
        self.toast.show_message(f"Sıvı Yok Kutuları: {durum}", color="#64ffda")
        self.show_current()

    def toggle_sivi_var(self):
        self.show_sivi_var = not self.show_sivi_var
        durum = "AÇIK" if self.show_sivi_var else "KAPALI"
        self.toast.show_message(f"Sıvı Var Kutuları: {durum}", color="#64ffda")
        self.show_current()

    # ── Theme ──

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#0a0a1a; color:#ccd6f6;
                font-family:'Segoe UI','Ubuntu',sans-serif; }
            QPushButton { background:#1a1a3e; color:#ccd6f6; border:1px solid #303060;
                border-radius:6px; padding:6px 14px; font-size:12px; font-weight:bold; }
            QPushButton:hover { background:#2a2a5e; border-color:#64ffda; color:#64ffda; }
            QPushButton:pressed { background:#3a3a7e; }
            QSlider::groove:horizontal { border:1px solid #303060; height:6px;
                background:#1a1a3e; border-radius:3px; }
            QSlider::handle:horizontal { background:#64ffda; width:16px; height:16px;
                margin:-5px 0; border-radius:8px; }
            QSlider::sub-page:horizontal { background:#233554; border-radius:3px; }
            QDoubleSpinBox { background:#1a1a3e; color:#ccd6f6;
                border:1px solid #303060; border-radius:4px; padding:2px 6px; }
            QStatusBar { background:#0d1117; color:#8892b0;
                border-top:1px solid #1a1a3e; font-size:12px; }
            QSplitter::handle { background:#303060; width:2px; }
            
            QPushButton#dayBtn { background:#2e1a1a; border:1px solid #8c3f3f; color:#ce6b6b; }
            QPushButton#dayBtn:hover { background:#4e2a2a; border-color:#ce6b6b; color:#f09c9c; }
            QPushButton#nightBtn { background:#2a2a1a; border:1px solid #8c8c3f; color:#cece6b; }
            QPushButton#nightBtn:hover { background:#4e4e2a; border-color:#cece6b; color:#f0f09c; }
            QPushButton#humanBtn { background:#1a2e1a; border:1px solid #3f8c5f; color:#6bce8c; }
            QPushButton#humanBtn:hover { background:#2a4e2a; border-color:#6bce8c; color:#9cf0af; }
            
            QPushButton#unknownBtn { background:#2e2a1a; border:1px solid #8c7a3f; color:#ceb96b; }
            QPushButton#unknownBtn:hover { background:#4e462a; border-color:#ceb96b; color:#f0de9c; }
        """)

    # ── Folder & Navigation ──

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Fotoğraf Klasörü Seç", "")
        if not folder:
            return
        self.main_folder = folder
        self.folder_label.setText(folder)
        self.folder_label.setStyleSheet("color: #64ffda; font-size: 12px;")
        self.status.showMessage("Klasör taranıyor...")
        QApplication.processEvents()

        self.photos = PhotoScanner.scan_folder(folder)
        self.current_index = 0
        n = len(self.photos)
        self.count_label.setText(f"📸 {n} fotoğraf")
        self.slider.setMaximum(max(0, n - 1))
        self.slider.setValue(0)

        if n > 0:
            self.status.showMessage(f"{n} fotoğraf bulundu ve sıralandı.")
            self.show_current()
        else:
            self.status.showMessage("Bu klasörde uygun fotoğraf bulunamadı.")
            self._clear_info()

    def show_current(self):
        if not self.photos:
            return
        p = self.photos[self.current_index]
        
        flags = {
            "kapakli": self.show_kapakli,
            "sivi-yok": self.show_sivi_yok,
            "sivi-var": self.show_sivi_var
        }
        
        self.panel1.display_image(p.file_path, self.inference_enabled, flags)
        self.panel2.display_image(p.file_path, self.inference_enabled, flags)
        self.panel3.display_image(p.file_path, self.inference_enabled, flags)
        
        self.last_action_path = None
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(100)
        self.zoom_val_lbl.setText("%100")
        self.zoom_slider.blockSignals(False)
        
        self.idx_label.setText(f"{self.current_index + 1} / {len(self.photos)}")
        self.cam_lbl.setText(f"📂 Kategori 1: {p.camera_id}")
        self.time_lbl.setText(f"⏰ Zaman: {p.time_str}")
        self.frame_lbl.setText(f"🆔 Dosya: {p.frame_id}")
        self.event_lbl.setText(f"📂 Kategori 2: {p.event_id}")
        self.fname_lbl.setText(p.filename)
        
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_index)
        self.slider.blockSignals(False)

    def _clear_info(self):
        self.idx_label.setText("0 / 0")
        for lbl, txt in [(self.cam_lbl, "📂 Kategori 1: -"), (self.time_lbl, "⏰ Zaman: -"),
                          (self.frame_lbl, "🆔 Dosya: -"), (self.event_lbl, "📂 Kategori 2: -")]:
            lbl.setText(txt)
        self.fname_lbl.setText("")

    def next_photo(self):
        if self.photos and self.current_index < len(self.photos) - 1:
            self.current_index += 1
            self.show_current()

    def prev_photo(self):
        if self.photos and self.current_index > 0:
            self.current_index -= 1
            self.show_current()

    def go_first(self):
        if self.photos:
            self.current_index = 0
            self.show_current()

    def go_last(self):
        if self.photos:
            self.current_index = len(self.photos) - 1
            self.show_current()

    def _slider_moved(self, value):
        if self.photos and 0 <= value < len(self.photos):
            self.current_index = value
            self.show_current()

    def _reset_zoom(self):
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(100)
        self.zoom_val_lbl.setText("%100")
        self.zoom_slider.blockSignals(False)

        self.panel1.view.reset_zoom()
        self.panel2.view.reset_zoom()
        self.panel3.view.reset_zoom()

    def _on_zoom_slider_changed(self, value):
        self.zoom_val_lbl.setText(f"%{value}")
        zoom_factor = value / 100.0
        self.panel1.view.set_absolute_zoom(zoom_factor)
        self.panel2.view.set_absolute_zoom(zoom_factor)
        self.panel3.view.set_absolute_zoom(zoom_factor)

    # ── Error Marking ──

    def mark_error(self, error_type: str):
        if not self.photos or not self.main_folder:
            self.toast.show_message("⚠️ Önce klasör seçin ve fotoğraf yükleyin.", color="#f0a500")
            self.status.showMessage("Önce klasör seçin ve fotoğraf yükleyin.")
            return

        photo = self.photos[self.current_index]
        main_name = os.path.basename(self.main_folder.rstrip('/\\'))
        parent_dir = os.path.dirname(self.main_folder.rstrip('/\\'))
        error_root = os.path.join(parent_dir, f"{main_name}_siniflandirma")
        dest_dir = os.path.join(error_root, error_type, photo.camera_id)

        os.makedirs(dest_dir, exist_ok=True)

        # Yeni dosya ismine klasör ismini (error_type) ekliyoruz
        base_name, ext = os.path.splitext(photo.filename)
        new_filename = f"{base_name}_{error_type}{ext}"
        dest_path = os.path.join(dest_dir, new_filename)

        if os.path.exists(dest_path):
            msg = f"⚠️ Zaten kopyalanmış:\n{new_filename}"
            self.toast.show_message(msg, color="#f0a500")
            self.status.showMessage(f"⚠️ Bu fotoğraf zaten kopyalanmış: {new_filename}")
            return

        _label_map = {
            "hatali": ("❌ Hatalı", "#ce6b6b"),
            "yari_hatali": ("⚠️ Yarı Hatalı", "#cece6b"),
            "hatasiz": ("✅ Hatasız", "#6bce8c"),
            "arada_kalanlar": ("❔ Arada Kalanlar", "#f0a500")
        }
                      
        label, color = _label_map.get(error_type, (error_type, "#64ffda"))

        try:
            shutil.copy2(photo.file_path, dest_path)
            self.last_action_path = dest_path
            toast_msg = f"{label}\n{new_filename}\n→ {photo.camera_id}/"
            self.toast.show_message(toast_msg, color=color)
            self.status.showMessage(
                f"✅ [{label}] {new_filename} → {photo.camera_id}/ klasörüne kopyalandı"
            )
        except Exception as e:
            self.toast.show_message(f"❌ Kopyalama hatası:\n{e}", color="#ff5555")
            self.status.showMessage(f"❌ Kopyalama hatası: {e}")

    def undo_last_action(self):
        if not self.last_action_path:
            self.toast.show_message("Bu fotoğraf için geri alınacak işlem yok.", color="#8892b0")
            self.status.showMessage("Bu fotoğraf için geri alınacak işlem yok.")
            return

        if os.path.exists(self.last_action_path):
            try:
                os.remove(self.last_action_path)
                filename = os.path.basename(self.last_action_path)
                self.toast.show_message(f"↩️ Geri Alındı\n{filename} klasörden silindi.", color="#f0a500")
                self.status.showMessage(f"↩️ Geri alma başarılı: {filename} hedef klasörden silindi.")
                self.last_action_path = None
            except Exception as e:
                self.toast.show_message(f"❌ Silme hatası:\n{e}", color="#ff5555")
                self.status.showMessage(f"❌ Geri alma sırasında silme hatası: {e}")
        else:
            self.toast.show_message("⚠️ Dosya zaten silinmiş veya bulunamadı.", color="#f0a500")
            self.status.showMessage("⚠️ Dosya bulunamadığı için geri alma işlemi atlandı.")
            self.last_action_path = None

    def toggle_inference(self):
        self.inference_enabled = not self.inference_enabled

        if self.inference_enabled:
            durum = "AÇIK"
            renk = "#64ffda"
            mesaj = "🧠 Model Çıkarımı: AÇIK"
            self.model_status_lbl.setText("🧠 Model: AÇIK")
            self.model_status_lbl.setStyleSheet(
                "color: #64ffda; font-size: 12px; font-weight: bold; margin-right: 15px;")
        else:
            durum = "KAPALI (Güç Tasarrufu)"
            renk = "#f0a500"
            mesaj = "⚡ Model Çıkarımı: KAPALI\n(Sadece ham görüntüler gösteriliyor)"
            self.model_status_lbl.setText("⚡ Model: KAPALI")
            self.model_status_lbl.setStyleSheet(
                "color: #f0a500; font-size: 12px; font-weight: bold; margin-right: 15px;")

        self.toast.show_message(mesaj, color=renk)
        self.status.showMessage(f"Model durumu: {durum}")

        if self.photos:
            self.show_current()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "toast") and self.toast.isVisible():
            self.toast._reposition()


# ─────────────────────────── Entry Point ───────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

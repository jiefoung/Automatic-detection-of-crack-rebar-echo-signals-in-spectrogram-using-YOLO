# -*- coding: utf-8 -*-
"""
YOLOv4 一鍵預測 GUI（打包友善版 / 可攜式支援）
-------------------------------------------------
- 單張預測：OpenCV DNN（GUI 內即時顯示，並輸出 YOLO txt: class conf cx cy w h）
- 批次預測：外部 darknet.exe（可隨包，或由使用者指定）
- 內建：resource_path() 支援 PyInstaller _MEIPASS，user_data_dir() 將輸出寫入可寫位置
- 可攜式模式：EXE 同層放置 portable.flag，輸出與預設模型/可執行檔以 EXE 同層為主

必要套件：
  pip install PySide6 chardet opencv-python matplotlib
建議打包（範例）：
  pyinstaller -y -D -w YOLOv4_IE_GUI_packed.py \
    --name YOLOv4_IE_GUI \
    --collect-all PySide6 \
    --add-data "models/*.*;models" \
    --add-data "darknet/*.*;darknet" \
    --hidden-import cv2 --hidden-import numpy
"""

import os
import sys
import json
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog, QHBoxLayout, QVBoxLayout, QGridLayout, QGroupBox, QTextEdit,
    QCheckBox, QMessageBox, QSpinBox
)

try:
    import chardet
except Exception:
    chardet = None

APP_TITLE = "基於YOLOV4-自動檢測鋼筋與裂縫回波"
APP_NAME = "YOLOv4IE"

# ----------------------------
# 打包/路徑工具
# ----------------------------

def resource_path(*paths: str) -> Path:
    """取得隨程式打包的唯讀資源路徑（支援 PyInstaller）。"""
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent
    return (base.joinpath(*paths)).resolve()


def user_data_dir(app_name: str = APP_NAME) -> Path:
    """使用者可寫的資料夾（Windows：LOCALAPPDATA／APPDATA）。"""
    base = (Path(os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or (Path.home() / "AppData" / "Local")))
    d = base / app_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_portable_mode() -> bool:
    """若 EXE 同層有 portable.flag，改用相對資料夾寫入。"""
    flag = resource_path("portable.flag")
    return flag.exists()


def writable_runs_dir() -> Path:
    if is_portable_mode():
        d = resource_path("runs")
    else:
        d = user_data_dir() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_timestamp_dir() -> Path:
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    d = writable_runs_dir() / 'predictions' / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def last_state_path() -> Path:
    return (user_data_dir() / "last_state.json")


def load_last_state() -> dict:
    p = last_state_path()
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            return {}
    return {}


def save_last_state(d: dict):
    try:
        last_state_path().write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def setup_darknet_env() -> str:
    """尋找可用的 darknet.exe（支援隨包、EXE 同層、CWD），會遞迴搜索子資料夾。找到後把其資料夾加到 PATH 並回傳路徑。"""
    roots = []
    # 1) 隨包資源根（_MEIPASS 或專案目錄）
    roots.append(resource_path("darknet"))
    # 2) EXE 同層的 darknet 目錄
    exe_dir = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    roots.append(exe_dir / "darknet")
    # 3) 目前工作目錄的 darknet 目錄
    roots.append(Path.cwd() / "darknet")

    # 先嘗試常見扁平位置
    for root in roots:
        cand = root / "darknet.exe"
        if cand.is_file():
            os.environ["PATH"] = str(cand.parent) + os.pathsep + os.environ.get("PATH", "")
            return str(cand)

    # 再遞迴找（支援 build/Release 等巢狀）
    for root in roots:
        if root.is_dir():
            try:
                cand = next(root.rglob("darknet.exe"))
                os.environ["PATH"] = str(cand.parent) + os.pathsep + os.environ.get("PATH", "")
                return str(cand)
            except StopIteration:
                pass

    # 最後掃 PATH
    for p in os.getenv("PATH", "").split(os.pathsep):
        cand = Path(p) / "darknet.exe"
        if cand.is_file():
            os.environ["PATH"] = str(cand.parent) + os.pathsep + os.environ.get("PATH", "")
            return str(cand)
    return ""


def guess_model_paths() -> Tuple[str, str, str]:
    """優先使用：GUI(稍後覆寫) -> last_state -> 內建資源。此函式僅提供預設值。"""
    st = load_last_state()

    def _prefer(*cands):
        for c in cands:
            if c and Path(c).is_file():
                return str(Path(c).resolve())
        return ""

    cfg = _prefer(st.get("cfg"), resource_path("models", "yolov4-obj.cfg"))
    weights = _prefer(st.get("weights"), resource_path("models", "yolov4-obj.weights"))
    names = _prefer(st.get("names"), resource_path("models", "obj.names"))
    return cfg, weights, names


# ----------------------------
# 一般工具
# ----------------------------

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def detect_encoding(path: str, default="utf-8") -> str:
    if not chardet:
        return default
    try:
        with open(path, 'rb') as f:
            data = f.read(8192)
        enc = chardet.detect(data).get('encoding')
        return enc or default
    except Exception:
        return default


def read_text_lines(path: str) -> list:
    enc = detect_encoding(path)
    with open(path, 'r', encoding=enc, errors='ignore') as f:
        return [line.strip() for line in f if line.strip()]


def parse_data_file(path: str) -> dict:
    out = {}
    if not path or not os.path.isfile(path):
        return out
    enc = detect_encoding(path)
    with open(path, 'r', encoding=enc, errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                out[k.strip()] = v.strip()
    return out


def count_names(names_path: str) -> Tuple[int, List[str]]:
    if not names_path or not os.path.isfile(names_path):
        return 0, []
    enc = detect_encoding(names_path)
    with open(names_path, 'r', encoding=enc, errors='ignore') as f:
        items = [ln.strip() for ln in f if ln.strip()]
    return len(items), items


def parse_cfg_classes(cfg_path: str) -> List[int]:
    vals = []
    if not cfg_path or not os.path.isfile(cfg_path):
        return vals
    enc = detect_encoding(cfg_path)
    with open(cfg_path, 'r', encoding=enc, errors='ignore') as f:
        lines = [ln.strip() for ln in f]
    for ln in lines:
        if ln.lower().startswith('classes') and '=' in ln:
            try:
                v = int(ln.split('=', 1)[1].strip())
                vals.append(v)
            except Exception:
                pass
    return vals


def render_cls_info(data_dict: dict, cfg_path: str, names_path: str) -> Tuple[str, List[str]]:
    data_classes = None
    try:
        if 'classes' in data_dict:
            data_classes = int(str(data_dict.get('classes')).strip())
    except Exception:
        data_classes = None

    cfg_classes_list = parse_cfg_classes(cfg_path)
    cfg_classes_set = set(cfg_classes_list) if cfg_classes_list else set()

    n_names, names_list = count_names(names_path)

    msg_lines = []
    if data_classes is not None:
        msg_lines.append(f".data classes = {data_classes}")
    if cfg_classes_list:
        msg_lines.append(f".cfg classes (all yolo heads) = {cfg_classes_list}")
    if n_names:
        msg_lines.append(f".names count = {n_names}")

    warnings = []
    targets = []
    if data_classes is not None:
        targets.append(data_classes)
    if cfg_classes_set:
        if len(cfg_classes_set) == 1:
            targets.append(next(iter(cfg_classes_set)))
        else:
            warnings.append("警告：你的 .cfg 內不同 yolo head 的 classes 不一致！")
    if n_names:
        targets.append(n_names)

    if targets and len(set(targets)) > 1:
        warnings.append("⚠ 類別數不一致（.data / .cfg / .names）請確認！")

    summary = "\n".join(msg_lines + (warnings if warnings else [])) or "尚未提供足夠檔案以判定類別數"
    return summary, names_list


# ----------------------------
# Darknet (批次) 後台執行
# ----------------------------
class DarknetRunner(QThread):
    log_sig = Signal(str)
    done_sig = Signal(bool, str)  # success, run_dir

    def __init__(self, exe_path: str, data_path: str, cfg_path: str, weights_path: str,
                 thresh: float, test_list: str | None, single_image: str | None,
                 run_dir: str):
        super().__init__()
        self.exe_path = exe_path
        self.data_path = data_path
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.thresh = thresh
        self.test_list = test_list
        self.single_image = single_image
        self.run_dir = run_dir

    def _write_log(self, msg: str):
        self.log_sig.emit(msg)
        try:
            with open(os.path.join(self.run_dir, 'run_log.txt'), 'a', encoding='utf-8') as f:
                f.write(msg.rstrip('\n') + '\n')
        except Exception:
            pass

    def _spawn(self, args: list, stdin_data: bytes | None = None):
        self._write_log(f"[CMD] {' '.join(args)}")
        try:
            p = subprocess.Popen(
                args,
                stdin=subprocess.PIPE if stdin_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.run_dir,
                shell=False,
            )
            out = p.communicate(input=stdin_data)[0].decode('utf-8', errors='ignore')
            self._write_log(out)
            return p.returncode, out
        except FileNotFoundError:
            self._write_log("❌ 找不到 darknet.exe，請確認路徑。")
            return 1, ""
        except Exception as e:
            self._write_log(f"❌ 執行失敗：{e}")
            return 1, ""

    def run(self):
        ok = False
        try:
            pred_dir = os.path.join(self.run_dir, 'predictions'); ensure_dir(pred_dir)
            base_args = [
                self.exe_path, 'detector', 'test',
                self.data_path, self.cfg_path, self.weights_path,
                '-thresh', str(self.thresh),
                '-ext_output', '-dont_show',
                '-out', os.path.join(self.run_dir, 'out.json')
            ]
            if self.test_list and os.path.isfile(self.test_list):
                images = read_text_lines(self.test_list)
                if not images:
                    self._write_log("❌ test.txt 內容為空。")
                    self.done_sig.emit(False, self.run_dir); return
                stdin_payload = ("\n".join(images) + "\n").encode('utf-8')
                rc, _ = self._spawn(base_args, stdin_data=stdin_payload)
                ok = (rc == 0)
            else:
                self._write_log("❌ 未提供 test.txt（批次模式）。")
                self.done_sig.emit(False, self.run_dir); return
        finally:
            self.done_sig.emit(ok, self.run_dir)


# ----------------------------
# 單張：OpenCV DNN 推論 + 顯示
# ----------------------------

def _draw_box_with_label(img: np.ndarray, x: int, y: int, w: int, h: int, label: str, color: tuple):
    H, W = img.shape[:2]
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(W - 1, x + w); y2 = min(H - 1, y + h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    pad = 4
    bx = min(max(0, x1), max(0, W - (tw + 2*pad)))
    by_top = y1 - (th + 2*pad + 2)
    by_bot = y2 + 2
    if by_top >= 0:
        by = by_top
    elif by_bot + th + 2*pad <= H:
        by = by_bot
    else:
        by = min(max(0, y1 + 2), max(0, H - (th + 2*pad)))
    cv2.rectangle(img, (bx, by), (bx + tw + 2*pad, by + th + 2*pad), color, thickness=-1)
    cv2.putText(img, label, (bx + pad, by + th + pad), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(img, label, (bx + pad, by + th + pad), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)


def run_single_opencv(image_path: str, cfg_path: str, weights_path: str, names: List[str],
                      conf_th: float = 0.25, nms_th: float = 0.45,
                      input_size: int = 608) -> Tuple[np.ndarray, List[Tuple[int,float,Tuple[int,int,int,int]]]]:
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"無法讀取影像：{image_path}")
    H, W = img.shape[:2]

    net = cv2.dnn.readNet(weights_path, cfg_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf > conf_th:
                cx, cy, w, h = det[0:4]
                cx *= W; cy *= H; w *= W; h *= H
                x = int(cx - w/2); y = int(cy - h/2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(conf)
                class_ids.append(cid)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_th, nms_th)
    results = []
    used_positions = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            cid = class_ids[i]
            conf = confidences[i]
            color = (0, 255, 255) if cid < len(names) and names[cid] == 'r_echo' else (255, 0, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            label = f"{names[cid] if cid < len(names) else cid}: {conf:.2f}"
            text_x = max(0, x + 30)
            text_y = y + h - 27
            while any(abs(text_y - uy) < 18 for uy in used_positions):
                text_y += 7
            text_y = min(max(0, text_y), img.shape[0] - 5)
            used_positions.append(text_y)
            cv2.putText(img, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            results.append((cid, conf, (x, y, w, h)))
    return img, results


def save_txt_for_single(results, out_txt: str, img_w: int, img_h: int):
    lines = []
    for cid, conf, (x, y, w, h) in results:
        cx = (x + w/2) / img_w
        cy = (y + h/2) / img_h
        rw = w / img_w
        rh = h / img_h
        lines.append(f"{cid} {conf:.6f} {cx:.6f} {cy:.6f} {rw:.6f} {rh:.6f}")
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def write_test_list_from_images(img_paths: List[str]) -> Path:
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = writable_runs_dir() / "_lists"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / f'drop_test_{ts}.txt'
    out_txt.write_text("\n".join(img_paths) + "\n", encoding='utf-8')
    return out_txt


# ----------------------------
# 主視窗
# ----------------------------
class DropLineEdit(QLineEdit):
    def __init__(self, on_drop=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_drop = on_drop
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        if not self.on_drop:
            return
        urls = e.mimeData().urls()
        paths = [u.toLocalFile() for u in urls]
        self.on_drop(paths)


class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 800)
        self.setAcceptDrops(True)

        self.run_dir = None
        self.runner: DarknetRunner | None = None

        # 欄位（全部支援直接拖拉到該欄位）
        self.ed_darknet = DropLineEdit(on_drop=lambda ps: self._drop_to_field(self.ed_darknet, ps, exts={'.exe'}))
        self.ed_data = DropLineEdit(on_drop=lambda ps: self._drop_to_field(self.ed_data, ps, exts={'.data'}))
        self.ed_cfg = DropLineEdit(on_drop=lambda ps: self._drop_to_field(self.ed_cfg, ps, exts={'.cfg'}))
        self.ed_weights = DropLineEdit(on_drop=lambda ps: self._drop_to_field(self.ed_weights, ps, exts={'.weights'}))
        self.ed_names = DropLineEdit(on_drop=lambda ps: self._drop_to_field(self.ed_names, ps, exts={'.names','.txt'}))
        self.ed_train = DropLineEdit(on_drop=lambda ps: self._drop_to_field(self.ed_train, ps, exts={'.txt'}))
        self.ed_test  = DropLineEdit(on_drop=lambda ps: self._drop_to_field(self.ed_test,  ps, exts={'.txt'}, allow_dir_as_images=True))
        self.ed_single= DropLineEdit(on_drop=lambda ps: self._drop_to_field(self.ed_single, ps, image_field=True))

        self.sb_thresh = QSpinBox(); self.sb_thresh.setRange(1, 99); self.sb_thresh.setValue(25); self.sb_thresh.setSuffix(" %")
        self.sb_nms = QSpinBox(); self.sb_nms.setRange(1, 99); self.sb_nms.setValue(45); self.sb_nms.setSuffix(" %")
        self.sb_input = QSpinBox(); self.sb_input.setRange(320, 1024); self.sb_input.setSingleStep(32); self.sb_input.setValue(608)

        self.cb_use_data_lists = QCheckBox("優先使用 .data 內的 train/valid/test"); self.cb_use_data_lists.setChecked(True)
        self.cb_show_preview = QCheckBox("單張預測顯示預覽（OpenCV DNN）"); self.cb_show_preview.setChecked(True)
        self.cb_save_txt = QCheckBox("單張輸出 YOLO txt"); self.cb_save_txt.setChecked(True)

        # 類別資訊
        self.lbl_cls_info = QLabel("類別資訊：尚未載入")
        self.txt_cls_list = QTextEdit(); self.txt_cls_list.setReadOnly(True)

        # 日誌與預覽
        self.log_view = QTextEdit(); self.log_view.setReadOnly(True)
        self.preview = QLabel("（預覽區）"); self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("QLabel{border:1px solid #666; background:#111; color:#aaa}")

        # 便利功能
        self.btn_gen_test_from_dir = QPushButton("從資料夾產生 test.txt")

        self._build_layout(); self._wire_events()

        # 預設：嘗試隨包 darknet 及模型
        dk_guess = setup_darknet_env()
        if dk_guess and not self.ed_darknet.text().strip():
            self.ed_darknet.setText(dk_guess)
        cfg0, wts0, names0 = guess_model_paths()
        if cfg0: self.ed_cfg.setText(cfg0)
        if wts0: self.ed_weights.setText(wts0)
        if names0: self.ed_names.setText(names0)
        self._update_class_panel()

        # 記憶上次自訂的 darknet.exe
        st = load_last_state()
        if st.get("darknet") and Path(st["darknet"]).is_file():
            self.ed_darknet.setText(st["darknet"])  # 使用者覆寫優先

    def _build_layout(self):
        def row(lbl: str, edit: QLineEdit):
            h = QHBoxLayout(); h.addWidget(QLabel(lbl)); h.addWidget(edit); btn = QPushButton("選擇…"); h.addWidget(btn); return h, btn

        g = QGridLayout()
        g.addWidget(QLabel("darknet.exe"), 0, 0); g.addWidget(self.ed_darknet, 0, 1); self.btn_darknet = QPushButton("選擇…"); g.addWidget(self.btn_darknet, 0, 2)

        h1, b1 = row(".data", self.ed_data); h2, b2 = row(".cfg", self.ed_cfg); h3, b3 = row(".weights", self.ed_weights); h4, b4 = row(".names", self.ed_names)
        box_files = QVBoxLayout(); box_files.addLayout(h1); box_files.addLayout(h2); box_files.addLayout(h3); box_files.addLayout(h4)
        self.btn_data, self.btn_cfg, self.btn_weights, self.btn_names = b1, b2, b3, b4

        h5, b5 = row("train.txt (選填)", self.ed_train); h6, b6 = row("test.txt (可批次)", self.ed_test); h7, b7 = row("單張影像 (可選)", self.ed_single)
        self.btn_train, self.btn_test, self.btn_single = b5, b6, b7

        grp_left = QGroupBox("模型/資料設定"); vleft = QVBoxLayout(); vleft.addLayout(box_files)
        cls_box = QGroupBox("類別資訊（自動從 .data/.cfg/.names 解析）"); vcls = QVBoxLayout(); vcls.addWidget(self.lbl_cls_info); vcls.addWidget(self.txt_cls_list); cls_box.setLayout(vcls); vleft.addWidget(cls_box)
        vleft.addLayout(h5); vleft.addLayout(h6); vleft.addLayout(h7)

        # 推論參數
        hp = QGroupBox("推論參數（單張預覽使用）"); vh = QHBoxLayout()
        vh.addWidget(QLabel("thresh")); vh.addWidget(self.sb_thresh)
        vh.addWidget(QLabel("nms")); vh.addWidget(self.sb_nms)
        vh.addWidget(QLabel("input")); vh.addWidget(self.sb_input)
        vh.addWidget(self.cb_show_preview); vh.addWidget(self.cb_save_txt)
        hp.setLayout(vh); vleft.addWidget(hp)

        vleft.addWidget(self.cb_use_data_lists)
        grp_left.setLayout(vleft)

        grp_right = QGroupBox("執行 / 預覽 / 日誌"); vright = QVBoxLayout()
        self.btn_predict_single = QPushButton("▶ 單張預測（OpenCV DNN）")
        self.btn_predict_batch = QPushButton("▶ 批次預測（darknet.exe）")
        self.btn_open_run = QPushButton("開啟輸出資料夾")
        vright.addWidget(self.btn_predict_single); vright.addWidget(self.btn_predict_batch); vright.addWidget(self.btn_gen_test_from_dir); vright.addWidget(self.btn_open_run)
        vright.addWidget(self.preview, 4)
        vright.addWidget(QLabel("執行日誌：")); vright.addWidget(self.log_view, 2)
        grp_right.setLayout(vright)

        main = QHBoxLayout(); main.addWidget(grp_left, 2); main.addWidget(grp_right, 3)
        root = QVBoxLayout(); root.addLayout(g); root.addLayout(main); self.setLayout(root)

    def _wire_events(self):
        self.btn_darknet.clicked.connect(lambda: self._pick_file(self.ed_darknet, 'darknet.exe', ["darknet.exe"]))
        self.btn_data.clicked.connect(lambda: self._pick_file(self.ed_data, 'obj.data', ['*.data']))
        self.btn_cfg.clicked.connect(lambda: self._pick_file(self.ed_cfg, 'yolov4.cfg', ['*.cfg']))
        self.btn_weights.clicked.connect(lambda: self._pick_file(self.ed_weights, 'weights', ['*.weights']))
        self.btn_names.clicked.connect(lambda: self._pick_file(self.ed_names, 'obj.names', ['*.names', '*.txt']))
        self.btn_train.clicked.connect(lambda: self._pick_file(self.ed_train, 'train.txt', ['*.txt']))
        self.btn_test.clicked.connect(lambda: self._pick_file(self.ed_test, 'test.txt', ['*.txt']))
        self.btn_single.clicked.connect(lambda: self._pick_file(self.ed_single, '影像', ['*.jpg', '*.png', '*.jpeg', '*.bmp']))

        self.ed_data.editingFinished.connect(self._on_data_changed)
        self.ed_cfg.editingFinished.connect(self._update_class_panel)
        self.ed_names.editingFinished.connect(self._update_class_panel)

        self.btn_predict_single.clicked.connect(self._run_single_opencv)
        self.btn_predict_batch.clicked.connect(self._run_batch_darknet)
        self.btn_open_run.clicked.connect(self._open_run_dir)
        self.btn_gen_test_from_dir.clicked.connect(self._gen_test_from_dir)

    def _pick_file(self, edit: QLineEdit, caption: str, patterns: List[str]):
        flt = ";;".join([f"{p} ({p})" for p in patterns])
        path, _ = QFileDialog.getOpenFileName(self, f"選擇 {caption}", str(Path.home()), flt)
        if path:
            edit.setText(path)
            if edit in (self.ed_data, self.ed_cfg, self.ed_names):
                self._update_class_panel()

    def _pick_dir(self, caption: str = "選擇資料夾") -> str | None:
        path = QFileDialog.getExistingDirectory(self, caption, str(Path.home()))
        return path or None

    def _on_data_changed(self):
        data_path = self.ed_data.text().strip()
        if not data_path:
            return
        info = parse_data_file(data_path)
        names_path = info.get('names') or info.get('names_path')
        if names_path and not self.ed_names.text().strip():
            if not os.path.isabs(names_path):
                names_path = os.path.join(os.path.dirname(data_path), names_path)
            if os.path.isfile(names_path):
                self.ed_names.setText(os.path.abspath(names_path))
        if self.cb_use_data_lists.isChecked():
            for key, target in [('train', self.ed_train), ('valid', self.ed_test), ('test', self.ed_test)]:
                p = info.get(key)
                if p and not target.text().strip():
                    if not os.path.isabs(p):
                        p = os.path.join(os.path.dirname(data_path), p)
                    if os.path.isfile(p):
                        target.setText(os.path.abspath(p))
        self._update_class_panel()

    def _update_class_panel(self):
        data_info = parse_data_file(self.ed_data.text().strip()) if self.ed_data.text().strip() else {}
        summary, names_list = render_cls_info(
            data_info, self.ed_cfg.text().strip(), self.ed_names.text().strip()
        )
        self.lbl_cls_info.setText(summary)
        self.txt_cls_list.setPlainText("\n".join(names_list))

    def _gen_test_from_dir(self):
        folder = self._pick_dir("選擇影像資料夾（將掃描產生 test.txt）")
        if not folder:
            return
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        paths = []
        for root, _, files in os.walk(folder):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in exts:
                    paths.append(os.path.abspath(os.path.join(root, fn)))
        if not paths:
            QMessageBox.information(self, "沒有影像", "此資料夾沒有 .jpg/.png/.bmp 影像。")
            return
        out_txt = write_test_list_from_images(paths)
        self.ed_test.setText(str(out_txt))
        self.log_view.append(f"已產生 test.txt：{out_txt}（共 {len(paths)} 張）")

    # ---------- 單張：OpenCV DNN ----------
    def _run_single_opencv(self):
        # 儲存當前狀態（先）
        save_last_state({
            "cfg": self.ed_cfg.text().strip(),
            "weights": self.ed_weights.text().strip(),
            "names": self.ed_names.text().strip(),
            "darknet": self.ed_darknet.text().strip()
        })

        # 路徑：GUI 欄位 > 內建資源
        cfg = self.ed_cfg.text().strip() or str(resource_path("models", "yolov4-obj.cfg"))
        wts = self.ed_weights.text().strip() or str(resource_path("models", "yolov4-obj.weights"))
        names_p = self.ed_names.text().strip() or str(resource_path("models", "obj.names"))
        img_p = self.ed_single.text().strip()
        if not (os.path.isfile(cfg) and os.path.isfile(wts) and os.path.isfile(img_p)):
            QMessageBox.warning(self, "缺少設定", "請選擇 .cfg / .weights 與 單張影像。若有 .names 請一併指定或使用隨包模型。")
            return
        _, names = count_names(names_p)
        conf_th = self.sb_thresh.value() / 100.0
        nms_th = self.sb_nms.value() / 100.0
        inp = int(self.sb_input.value())

        # 輸出位置
        run_dir = new_timestamp_dir(); self.run_dir = str(run_dir)
        self._append_log(f"輸出資料夾：{run_dir}")
        try:
            vis, results = run_single_opencv(img_p, cfg, wts, names, conf_th, nms_th, inp)
        except Exception as e:
            QMessageBox.critical(self, "推論失敗", str(e)); return

        H, W = vis.shape[:2]
        # 預覽
        if self.cb_show_preview.isChecked():
            qimg = QImage(vis.data, W, H, vis.strides[0], QImage.Format_BGR888)
            self.preview.setPixmap(QPixmap.fromImage(qimg).scaled(self.preview.width(), self.preview.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # 儲存輸出影像與 txt
        base = os.path.splitext(os.path.basename(img_p))[0]
        out_img = run_dir / f"{base}_pred.png"; cv2.imwrite(str(out_img), vis)
        if self.cb_save_txt.isChecked():
            out_txt = run_dir / f"{base}.txt"; save_txt_for_single(results, str(out_txt), W, H)
        self._append_log(f"單張完成：{out_img}")

    # ---------- 批次：darknet.exe ----------
    def _run_batch_darknet(self):
        # 儲存當前狀態（先）
        save_last_state({
            "cfg": self.ed_cfg.text().strip(),
            "weights": self.ed_weights.text().strip(),
            "names": self.ed_names.text().strip(),
            "darknet": self.ed_darknet.text().strip()
        })

        ok, msg = self._validate_common_for_batch()
        if not ok:
            QMessageBox.warning(self, "缺少設定", msg); return
        test = self.ed_test.text().strip()
        if not os.path.isfile(test):
            QMessageBox.warning(self, "找不到檔案", f"test.txt 不存在：\n{test}"); return
        run_dir = new_timestamp_dir(); self.run_dir = str(run_dir)
        self._append_log(f"輸出資料夾：{run_dir}")
        self.runner = DarknetRunner(
            exe_path=self.ed_darknet.text().strip(),
            data_path=self.ed_data.text().strip(),
            cfg_path=self.ed_cfg.text().strip(),
            weights_path=self.ed_weights.text().strip(),
            thresh=self.sb_thresh.value()/100.0,
            test_list=test,
            single_image=None,
            run_dir=str(run_dir),
        )
        self.runner.log_sig.connect(self._append_log)
        self.runner.done_sig.connect(self._on_done)
        self._toggle_ui(False); self.runner.start()

    # 共用
    def _validate_common_for_batch(self) -> Tuple[bool, str]:
        dk = self.ed_darknet.text().strip(); data = self.ed_data.text().strip(); cfg = self.ed_cfg.text().strip(); wts = self.ed_weights.text().strip()
        if not os.path.isfile(dk):
            # 若 GUI 沒填但隨包有，就再嘗試一次
            dk_guess = setup_darknet_env()
            if dk_guess and Path(dk_guess).is_file():
                self.ed_darknet.setText(dk_guess)
                dk = dk_guess
        if not os.path.isfile(dk):
            return False, "請指定正確的 darknet.exe 路徑（僅批次模式需要）。"
        for pth, name in [(data, '.data'), (cfg, '.cfg'), (wts, '.weights')]:
            if not os.path.isfile(pth):
                return False, f"請指定正確的 {name} 路徑。"
        return True, "OK"

    def _append_log(self, msg: str):
        self.log_view.append(msg.rstrip('\n'))

    def _on_done(self, success: bool, run_dir: str):
        self._toggle_ui(True)
        self._append_log("✅ 完成！" if success else "❌ 執行失敗，請檢查日誌與路徑設定。")

    def _toggle_ui(self, enabled: bool):
        for w in [self.btn_darknet, self.btn_data, self.btn_cfg, self.btn_weights, self.btn_names,
                  self.btn_train, self.btn_test, self.btn_single, self.btn_predict_single,
                  self.btn_predict_batch, self.btn_gen_test_from_dir]:
            w.setEnabled(enabled)

    def _open_run_dir(self):
        if not self.run_dir or not os.path.isdir(self.run_dir):
            QMessageBox.information(self, "尚未執行", "請先執行一次預測後再開啟輸出資料夾。"); return
        if sys.platform.startswith('win'):
            os.startfile(self.run_dir)
        else:
            subprocess.call(['open' if sys.platform=='darwin' else 'xdg-open', self.run_dir])

    # 視窗整體拖放支援
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
        else: e.ignore()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if not urls: return
        paths = [u.toLocalFile() for u in urls]
        self._auto_route_drop(paths)

    # 欄位拖放與自動分流
    def _drop_to_field(self, edit: QLineEdit, paths: List[str], exts: set[str] = None,
                       image_field: bool = False, allow_dir_as_images: bool = False):
        exts = exts or set()
        img_exts = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
        picked = None
        images = []
        for p in paths:
            if os.path.isdir(p):
                if allow_dir_as_images or image_field:
                    for root, _, files in os.walk(p):
                        for fn in files:
                            if os.path.splitext(fn)[1].lower() in img_exts:
                                images.append(os.path.abspath(os.path.join(root, fn)))
            else:
                ext = os.path.splitext(p)[1].lower()
                if image_field and ext in img_exts:
                    picked = os.path.abspath(p)
                elif ext in exts:
                    picked = os.path.abspath(p)
        if images and (allow_dir_as_images or (image_field and not picked)):
            out_txt = write_test_list_from_images(images)
            if edit is self.ed_test:
                edit.setText(str(out_txt))
            else:
                self.ed_test.setText(str(out_txt))
            self.log_view.append(f"拖放建立 test.txt：{out_txt}（{len(images)} 張）")
        if picked:
            edit.setText(picked)
            if edit in (self.ed_data, self.ed_cfg, self.ed_names):
                self._update_class_panel()

    def _auto_route_drop(self, paths: List[str]):
        img_exts = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
        images = []
        for p in paths:
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for fn in files:
                        if os.path.splitext(fn)[1].lower() in img_exts:
                            images.append(os.path.abspath(os.path.join(root, fn)))
            else:
                ext = os.path.splitext(p)[1].lower()
                base = os.path.basename(p).lower()
                if ext == '.exe' and 'darknet' in base:
                    self.ed_darknet.setText(os.path.abspath(p))
                elif ext == '.data':
                    self.ed_data.setText(os.path.abspath(p))
                    self._on_data_changed()
                elif ext == '.cfg': self.ed_cfg.setText(os.path.abspath(p))
                elif ext == '.weights': self.ed_weights.setText(os.path.abspath(p))
                elif ext == '.names': self.ed_names.setText(os.path.abspath(p))
                elif ext == '.txt': self.ed_test.setText(os.path.abspath(p))
                elif ext in img_exts: images.append(os.path.abspath(p))
        if images:
            out_txt = write_test_list_from_images(images)
            self.ed_test.setText(str(out_txt))
            self.log_view.append(f"拖放建立 test.txt：{out_txt}（{len(images)} 張）")


def main():
    app = QApplication(sys.argv)
    app.setApplicationDisplayName(APP_TITLE)
    w = MainUI(); w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

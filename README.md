# YOLOv4 自動檢測 GUI 

**YOLOv4_GUI** 是一款以 PySide6 製作的可視化圖形介面，用於自動執行 **YOLOv4 模型的影像或批次檢測**。此專案整合了 **OpenCV DNN 單張推論** 與 **Darknet 批次推論**，支援 **可攜式模式 (Portable Mode)**，可於 Windows 上單檔執行。

---

##  專案簡介

本專案旨在讓使用者能夠**以最少設定、快速進行 YOLOv4 推論**。不論是單張圖片或整批影像，只需透過拖放即可開始偵測。程式支援自動偵測模型、配置檔，並可輸出結果影像與 YOLO txt 格式標註檔。

###  主要特點

* **單張預測模式 (OpenCV DNN)**：

  * 即時顯示結果於 GUI 中。
  * 自動輸出 YOLO txt (`class conf cx cy w h`)
* **批次預測模式 (Darknet)**：

  * 呼叫外部 `darknet.exe`，支援整批影像輸入。
* **自動路徑管理**：

  * 內建 `resource_path()` 與 `user_data_dir()`，支援 PyInstaller 打包。
* **Portable 模式**：

  * 若存在 `portable.flag`，則所有輸出與模型皆以 EXE 同層目錄為主。
* **拖放支援**：

  * 可直接拖入 `.cfg`, `.weights`, `.names`, `.data`, `.txt`, 或影像資料夾。
* **可視化預覽**：

  * OpenCV 即時顯示標框與信心分數。
* **自動產生 test.txt**：

  * 從資料夾快速生成批次測試清單。

---

##  系統需求

* **作業系統**：Windows 10 / 11
* **Python 版本**：3.10 ~ 3.12
* **建議硬體**：

  * GPU (CUDA) 用於批次模式 Darknet.exe
  * CPU 即可運行 OpenCV DNN 模式

---

##  安裝與執行

###  安裝依賴

若以 Python 原始碼運行：

```bash
pip install PySide6 chardet opencv-python matplotlib numpy
```

或使用 `requirements.txt`：

```bash
pip install -r requirements.txt
```

###  執行方式

```bash
python YOLOV4_GUI_packed.py
```

或打包為 EXE：

```bash
pyinstaller -y -D -w YOLOV4_GUI_packed.py ^
  --name YOLOv4_IE_GUI ^
  --collect-all PySide6 ^
  --add-data "models/*.*;models" ^
  --add-data "darknet/*.*;darknet" ^
  --hidden-import cv2 --hidden-import numpy
```

---

##  專案結構建議

```bash
YOLOv4_IE_GUI/
│
├─ YOLOV4_GUI_packed.py
├─ models/
│   ├─ yolov4-obj.cfg
│   ├─ yolov4-obj.weights
│   └─ obj.names
│
├─ darknet/
│   └─ darknet.exe
│
├─ portable.flag       ← 若存在則啟用可攜式模式
└─ runs/               ← 預測輸出目錄
```

---

##  使用教學

###  單張推論 (OpenCV DNN)

1. 啟動 GUI。
2. 填入或拖入 `.cfg`、`.weights`、`.names`、影像路徑。
3. 點選 **▶ 單張預測（OpenCV DNN）**。
4. GUI 內即時顯示結果，並於 `runs/predictions/<timestamp>/` 儲存輸出：

   * `xxx_pred.png`
   * `xxx.txt`

###  批次推論 (Darknet.exe)

1. 指定 `darknet.exe` 與 `.data`, `.cfg`, `.weights`, `test.txt`。
2. 點選 **▶ 批次預測（darknet.exe）**。
3. 系統會自動執行推論並生成：

   * `out.json`
   * `run_log.txt`
   * 各張預測結果影像與標註檔。

###  自動產生 test.txt

* 點擊「從資料夾產生 test.txt」按鈕。
* 系統會自動掃描資料夾內的 `.jpg / .png / .bmp` 檔案，生成 `test.txt`。

---

##  requirements.txt 範例

```txt
PySide6>=6.6.0
opencv-python>=4.10.0
matplotlib>=3.8.0
chardet>=5.2.0
numpy>=1.26.0
```

---

##  可攜式 (Portable) 模式

若 EXE 同層存在 `portable.flag` 檔案：

* 輸出與暫存資料夾將在 EXE 同層建立。
* 無需安裝即可於 USB 或隨身碟中直接運行。
* 不會於系統 AppData 留存任何設定或快取。

---

##  設定說明

| 欄位          | 說明                                     |
| ----------- | -------------------------------------- |
| darknet.exe | Darknet 執行檔（批次推論用）                     |
| .data       | YOLO 資料定義檔，包含 train/test 路徑與 names 檔路徑 |
| .cfg        | 模型結構設定檔                                |
| .weights    | 模型權重檔案                                 |
| .names      | 類別名稱定義檔                                |
| test.txt    | 影像清單檔案（每行一張影像路徑）                       |
| 單張影像        | 用於單張推論（OpenCV DNN）                     |

---

##  執行結果範例

**單張推論輸出檔案：**

```
runs/predictions/20251101-120530/
├─ sample_pred.png   ← 偵測結果影像
└─ sample.txt        ← YOLO txt 標註檔
```

**批次推論輸出檔案：**

```
runs/predictions/20251101-121000/
├─ out.json
├─ run_log.txt
├─ img1_pred.png
├─ img1.txt
└─ img2_pred.png
```

---

##  作者資訊

* **開發者**：jiefoung
* **版本**：YOLOv4_GUI v1.0
* **領域**：非破壞檢測 (NDT) 回波訊號自動識別
* **說明**：以 YOLOv4 + OpenCV DNN 為核心，設計出友善的 GUI 工具，用於自動化裂縫回波訊號辨識。

---

##  授權條款

本專案僅供**研究與教學用途**，禁止未經授權的再散布、修改與商業使用。
如需將本軟體整合進其他系統或產品，請先取得作者授權。

---

>  本專案旨在推廣深度學習於工程影像檢測的應用，期望能降低模型操作門檻，讓更多研究者與工程師能快速進行 YOLOv4 應用實驗。



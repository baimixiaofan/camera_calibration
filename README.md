# 相机标定工具 (张正友标定法 · 多轮平均)

基于 **张正友平面标定法** 的单文件 GUI 相机标定工具，支持 ESP32 网络相机 / USB 相机 / 本地图像目录三种图像源，内置 **多轮随机子集标定取平均** 机制，输出结果可直接用于 ROS / OpenCV / C++ 项目。

## 功能特性

- **多种图像源**：ESP32 相机（HTTP 快照接口）/ USB 相机（OpenCV VideoCapture）/ 本地图像目录离线标定
- **实时预览 + 角点叠加**：采集时实时显示角点检测状态（绿色连线 = 检测成功），避免无效采图
- **多轮平均标定**：每轮随机抽取 80% 图像独立标定，统计各参数的均值与标准差，量化结果可信度
- **一键保存 YAML**：OpenCV FileStorage 格式，字段命名与 ROS 相机标定规范兼容
- **去畸变预览**：标定完成后左右对比原图与校正结果，直观验证
- **参数记忆**：所有配置自动保存，换相机标定无需重新配置
- **WSL 友好**：自动探测中文字体，缺字体时给出安装指引

## 标定原理

1. 打印一张棋盘格标定板，其上每个内角点的世界坐标已知（Z=0，XY 由网格索引 × 格边长推出，无需逐点测量）
2. 从不同姿态拍摄标定板，角点检测得到各角点的像素坐标，构成 3D↔2D 点对
3. 每张图可解出平面单应矩阵 `H = K·[r1 r2 t]`，利用旋转列向量的正交单位性建立内参约束（≥3 张不同姿态的图即可解出 K 初值）
4. OpenCV `calibrateCamera` 内部完成「解 H → 分解初值 → LM 非线性联合优化（内参 + 畸变 + 外参）」

## 多轮平均标定机制

单次标定结果受个别图像的姿态/噪声影响，且无法体现结果对图像选取的敏感度。本工具每轮从全部图像中**不放回随机抽取 80%** 独立标定一次（默认 10 轮），对 fx/fy/cx/cy/畸变系数统计均值与标准差：

- **标准差小** → 结果稳定可信
- **标准差大** → 该参数不可靠：姿态多样性不足，或该参数在当前分辨率/视场下不敏感，提示补图

实测（320×240 合成数据闭环验证，真值 fx=600, k1=-0.2）：

```
fx  = 606.8 ± 5.0        (偏差 ~1%)
k1  = -0.193 ± 0.008     (真值 -0.2)
RMS = 0.277 ± 0.003 px
```

## 环境要求

- Python 3.8+
- OpenCV ≥ 4.5、NumPy（见 `requirements.txt`）
- tkinter（Windows/macOS 自带；Ubuntu/WSL 需 `sudo apt install python3-tk`）
- WSL 下界面中文显示为方块时，安装中文字体：`sudo apt install fonts-noto-cjk`

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 准备标定板

```bash
# 项目已附带可打印的 chessboard_print.png (10x7 格)
# 打印后用尺子量一格的实际边长, 在 GUI 中填入 (单位 mm)
```

### 2. 采集与标定（GUI）

```bash
python3 calibration_gui.py
```

操作流程：

1. **选择图像来源**：填 ESP32 的 IP（默认 `10.68.84.221`）、USB 索引（0/1/2...）或图像目录
2. **配置棋盘参数**：内角点列/行数（例：10×7 格的板子 → 9×6 内角点）、每格边长
3. **开始预览**：画面出现绿色角点连线说明检测成功
4. **抓取**：手持标定板不断改变**位置 / 角度 / 距离**，采集 15~25 张有效图；姿态越丰富结果越好（建议覆盖画面的各个区域，并包含一定倾斜角）
5. **开始标定**：查看多轮统计结果，重点看 RMS（建议 < 0.5 px）与各参数标准差
6. **保存 YAML** / **去畸变预览** 验证效果

### 3. 使用结果

```python
import cv2
fs = cv2.FileStorage("calib_result.yaml", cv2.FILE_STORAGE_READ)
K    = fs.getNode("camera_matrix").mat()      # 3x3 内参矩阵
dist = fs.getNode("dist_coeffs").mat()        # 畸变系数 (k1,k2,p1,p2,k3)
img  = cv2.undistort(img, K, dist, None, K)   # 去畸变
```

## 输出 YAML 字段

| 字段 | 说明 |
|------|------|
| `image_width` / `image_height` | 标定图像分辨率 |
| `camera_matrix` | 3×3 内参矩阵（多轮平均值） |
| `dist_coeffs` | 1×5 畸变系数 (k1, k2, p1, p2, k3) |
| `rms_reproj_error_mean` / `_std` | 多轮重投影误差均值/标准差 (px) |
| `n_images` / `n_runs` | 有效图像数 / 成功轮数 |
| `chessboard_cols` / `chessboard_rows` / `square_size_m` | 棋盘参数备忘 |

> 字段命名与 ROS `camera_calibration_parsers` 兼容，可直接用于 ROS 相机驱动。

## 注意事项

- **小分辨率/小视场相机**（如 320×240 的 ESP32 模组）：k2、k3 的成像效应低于亚像素检测噪声，不可观测，程序默认固定为 0（`CALIB_FLAGS = CALIB_FIX_K2 | CALIB_FIX_K3`）。高分辨率/广角相机请改回全模型 `CALIB_FLAGS = 0`
- **标定质量判读**：RMS < 0.5 px 为良好；某参数标准差偏大 → 按提示补充不同姿态的图像后重新标定
- 标定板的格子数以**内角点**为准（黑白格交界处的顶点），不是格子数

## 项目结构

```
├── calibration_gui.py     # 主程序 (GUI + 张正友标定核心, 单文件)
├── chessboard_print.png   # 可打印标定板 (10x7 格, 打印后量边长填入 GUI)
├── requirements.txt
└── .gitignore
```

运行时生成（已忽略，不入库）：`gui_settings.json`（参数记忆）、标定图像目录、`calib_result.yaml`（标定结果）。

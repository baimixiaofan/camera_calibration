#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机标定 GUI 工具 (张正友标定法, 单文件自包含)
================================================

功能:
    1. 三种图像源: ESP32 相机 (HTTP /capture) / USB 相机 (cv2.VideoCapture) /
       本地图像目录 (标定板放不下大视场或离线补图时用);
    2. 实时预览 + 角点自动检测叠加 (绿色连线=检测成功), 一键抓取入队;
    3. 多轮平均标定: 每轮随机抽取 80% 图像独立标定, 统计 fx/fy/cx/cy/畸变的
       均值与标准差 —— 标准差小 = 结果稳定可信; 某参数标准差大 = 姿态多样性
       不足或该参数在此分辨率下不敏感, 提示补图;
    4. 结果一键保存 YAML (FileStorage 格式, ROS/C++ 可直接读取),
       并支持去畸变预览直观验证;
    5. 所有参数自动记忆 (gui_settings.json), 下次给其他相机标定无需重配。

标定原理 (张正友标定法):
    棋盘格上角点世界坐标已知 (Z=0, XY 由网格索引 x 格边长推出), 角点检测得到
    其像素坐标, 构成 3D<->2D 点对; 每张图可解出平面单应矩阵 H = K·[r1 r2 t],
    利用旋转列向量正交性得到内参约束 (>=3 张图可解 K 初值), 再 LM 非线性联合
    优化内参/畸变/外参 —— OpenCV calibrateCamera 已封装此流程。

运行: python3 calibration_gui.py
"""

import json
import os
import queue
import threading
import time
import urllib.request
import numpy as np
import cv2
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

# ======================================================================
#                          参数"宏"定义
# ======================================================================
SETTINGS_FILE = "gui_settings.json"   # 参数记忆文件
PREVIEW_MAX_W = 360                   # 预览画布最大宽 (像素, 4:3 匹配 320x240)
PREVIEW_MAX_H = 270                   # 预览画布最大高 (像素)
PREVIEW_POLL_MS = 100                 # 主线程刷新预览的周期 (ms, 10fps 足够标定用)
ESP32_FETCH_GAP = 0.15                # ESP32 抓图间隔 (秒), 避免请求过密
USB_RETRY_GAP = 1.0                   # USB 出错重试间隔 (秒)
MIN_OK_IMAGES = 6                     # 开始标定的最少有效图像数
DEFAULT_ESP32_IP = "10.68.84.221"     # ESP32 默认地址
ESP32_SNAPSHOT_API = "/capture"       # ESP32 快照 HTTP 接口
IMAGE_EXT = ("jpg", "jpeg", "png", "bmp")   # 支持的图像扩展名

# ---------- 标定算法参数 ----------
CORNER_WIN = 11              # 亚像素精化窗口边长 (像素), 奇数
CORNER_MAX_ITER = 30         # 亚像素迭代上限
CORNER_EPS = 0.001          # 亚像素收敛阈值 (像素)
# 畸变模型: 固定 k2=k3=0, 只估计 K + (k1,p1,p2)。
# 原因: 320x240 小视场下 k2 的成像效应仅约 0.4px, 低于亚像素检测噪声,
# 参与优化会与 k1 强耦合导致符号翻转/数值发散 (实测 k2 -> -4.6, k3 -> +47)。
# 若换高分辨率/广角镜头相机, 请改用全模型: CALIB_FLAGS = 0
CALIB_FLAGS = cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
TERM_MAX_ITER = 100          # LM 优化迭代上限
TERM_EPS = 1e-6              # LM 收敛阈值


# ======================================================================
#                    标定核心算法 (张正友标定法)
# ======================================================================

def _board_corners_3d(cols, rows, square):
    """构造棋盘全部内角点的世界坐标, Z=0 (标定板所在平面)。

    cols/rows: 内角点列/行数; square: 格边长(米)。
    返回: (N,3) float32, 行优先遍历, 与 findChessboardCorners 输出顺序对应。
    """
    i, j = np.mgrid[0:cols, 0:rows]
    grid = np.stack([i.T, j.T], axis=-1).reshape(-1, 2)
    return np.hstack([grid * square, np.zeros((grid.shape[0], 1))]).astype(np.float32)


def detect_corners(gray, pattern):
    """在灰度图中检测棋盘格内角点, 并做亚像素精化。

    pattern: (内角点列数, 内角点行数)。
    优先使用 findChessboardCornersSB (基于棋盘能量图 + 径向排序,
    亚像素精度更高, 实测 RMS 0.28px vs 经典法 0.42px);
    SB 失败时回退到经典 自适应阈值 + cornerSubPix 流程 (对低质图像更鲁棒)。
    返回: (N,1,2) float32 亚像素角点, 检测失败返回 None。
    """
    # SB 检测器 (内部自带亚像素精化)
    ret = cv2.findChessboardCornersSB(gray, pattern)
    if ret is not None:
        ok, corners = ret
        if ok:
            return np.ascontiguousarray(corners.reshape(-1, 1, 2), np.float32)
    # 经典检测流程
    # ADAPTIVE_THRESH: 自适应阈值应对光照不均; NORMALIZE_IMAGE: 直方图均衡
    ok, corners = cv2.findChessboardCorners(
        gray, pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    # 亚像素精化: 在原像素位置附近的小窗口内, 沿梯度方向找角点精确位置
    corners = cv2.cornerSubPix(
        gray, corners, (CORNER_WIN, CORNER_WIN), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
         CORNER_MAX_ITER, CORNER_EPS))
    return corners


def calibrate(obj_points, img_points, size, flags=CALIB_FLAGS):
    """调用 calibrateCamera 完成单次张正友标定。

    内部: 逐图解单应矩阵 H -> 利用旋转正交性解 K 初值 ->
          LM 非线性联合优化 (K + 畸变 + 全部外参)。
    返回: (RMS 重投影误差, 内参矩阵 K, 畸变向量, 外参 rvecs/tvecs)
    """
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            TERM_MAX_ITER, TERM_EPS)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, size, None, None, flags=flags, criteria=term)
    return rms, K, np.array(dist).ravel(), rvecs, tvecs


def calibrate_multi(obj_points, img_points, size, n_runs=10, subset_ratio=0.8, seed=None):
    """多次标定取平均: 每轮随机抽取一部分图像独立标定, 最后统计均值与标准差。

    机制说明:
        - 单次标定结果受个别图像的姿态/噪声影响较大, 直接用全部图像只得到一个值,
          无法知道结果对图像选取的敏感程度;
        - 本函数每轮从全部图像中不放回随机抽取 subset_ratio 比例 (最少 3 张),
          独立跑一次标定, 共 n_runs 轮;
        - 对各轮的 fx/fy/cx/cy 及畸变系数求均值与标准差:
          标准差小 -> 结果稳定可信; 某参数标准差大 -> 说明采集的图像姿态
          多样性不足或分辨率/视场对该参数不敏感, 需要补图。

    返回 dict:
        K_mean/dist_mean : 各轮平均的内参与畸变 (最终推荐结果)
        K_std            : (fx, fy, cx, cy) 各轮标准差
        dist_std         : 各轮畸变系数标准差
        rms_list         : 各轮 RMS 重投影误差 (px)
        runs_ok/runs_total : 成功轮数 / 总轮数 (个别子集标定失败会被跳过)
    """
    n = len(obj_points)
    k = max(3, int(round(n * subset_ratio)))     # 每轮子集大小 (至少 3 张才能解内参)
    if k > n:
        k = n
    rng = np.random.default_rng(seed)
    Ks, dists, rms_list = [], [], []
    for _ in range(n_runs):
        idx = rng.choice(n, k, replace=False)
        try:
            rms, K, dist, _, _ = calibrate([obj_points[i] for i in idx],
                                           [img_points[i] for i in idx], size)
        except cv2.error:
            continue                              # 个别退化子集跳过
        Ks.append([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])   # fx, fy, cx, cy
        dists.append(dist[:5])                        # k1,k2,p1,p2,k3
        rms_list.append(float(rms))
    runs_ok = len(Ks)
    if runs_ok == 0:
        raise RuntimeError("所有子集标定均失败, 请检查图像质量与棋盘参数")
    Ks, dists = np.array(Ks), np.array(dists)
    K_mean = np.array([[Ks[:, 0].mean(), 0, Ks[:, 2].mean()],
                       [0, Ks[:, 1].mean(), Ks[:, 3].mean()],
                       [0, 0, 1]])
    return {
        "K_mean": K_mean,
        "K_std": Ks.std(axis=0),                  # (fx, fy, cx, cy) 标准差
        "dist_mean": dists.mean(axis=0),
        "dist_std": dists.std(axis=0),
        "rms_list": rms_list,
        "runs_ok": runs_ok,
        "runs_total": n_runs,
    }


class CalibGUI:
    """主窗口: 负责 UI 构建、线程调度与结果展示; 算法见上方标定核心函数。"""

    # ------------------------------------------------------------------
    #                          初始化与 UI 构建
    # ------------------------------------------------------------------
    def __init__(self, root):
        self.root = root
        root.title("相机标定工具 - 张正友标定法 (多轮平均)")
        root.geometry("900x760")          # 初始窗口尺寸
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        # ---- 中文字体: WSL/精简系统常无中文字体 (界面全是方块), 自动探测并应用 ----
        reg, mono = _pick_cjk_fonts()
        if reg:
            root.option_add("*Font", (reg, 10))          # Tk 传统控件默认字体
            ttk.Style(root).configure(".", font=(reg, 10))  # ttk 主题控件字体
        self._fixed_font = (mono or "TkFixedFont", 10)   # 列表/结果区字体
        if reg is None:
            print("[提示] 未找到中文字体, 界面可能显示为方块。\n"
                  "       WSL 下可执行: sudo apt install fonts-noto-cjk")

        # ---- 运行时状态 ----
        self.images = []          # 已采集图像 [{"frame":np, "corners":np, "ok":bool}]
        self._latest = None       # 预览线程缓存的最新一帧 (disp, corners)
        self._latest_orig = None  # 最新一帧的原始图像 (无叠加线, 供去畸变预览)
        self._latest_lock = threading.Lock()
        self.q = queue.Queue()    # 工作线程 -> 主线程 的消息队列 (tkinter 非线程安全)
        self._preview_run = threading.Event()   # 预览线程运行标志
        self._worker = None       # 预览线程句柄
        self._cap = None          # USB 摄像头句柄
        self._result = None       # 最近一次标定结果 (dict)
        self._size = None         # 标定图像尺寸 (w, h)
        self._photo = None        # PhotoImage 引用 (防止被 GC 回收导致白图)
        self._seq = 0             # 图像序号

        self._build_ui()
        self._load_settings()
        root.after(PREVIEW_POLL_MS, self._poll_queue)

    def _build_ui(self):
        """构建全部控件 (左列: 控制/列表/结果, 右列: 预览画布)。"""
        pad = {"padx": 4, "pady": 2}
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        # ================= 左列: 参数与操作 =================
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True, **pad)

        # ---- 相机源 ----
        box_src = ttk.LabelFrame(left, text=" 图像来源 ")
        box_src.pack(fill="x", **pad)
        self.var_src = tk.StringVar(value="esp32")
        for val, txt in [("esp32", "ESP32 (IP)"), ("usb", "USB 相机"), ("folder", "图像目录")]:
            ttk.Radiobutton(box_src, text=txt, value=val, variable=self.var_src,
                            command=self._on_src_change).pack(side="left", **pad)
        row = ttk.Frame(box_src); row.pack(fill="x", **pad)
        self.lbl_src = ttk.Label(row, text="IP:")
        self.lbl_src.pack(side="left")
        self.ent_src = ttk.Entry(row, width=24)
        self.ent_src.insert(0, DEFAULT_ESP32_IP)
        self.ent_src.pack(side="left", fill="x", expand=True, **pad)
        self.btn_browse = ttk.Button(row, text="浏览...", width=8,
                                     command=self._browse_folder)
        ttk.Label(box_src, text="(USB: 填索引号 0/1/2...; 目录: 填路径或点浏览)",
                  foreground="#777").pack(anchor="w", **pad)

        # ---- 棋盘格参数 ----
        box_cb = ttk.LabelFrame(left, text=" 棋盘格参数 (内角点) ")
        box_cb.pack(fill="x", **pad)
        defaults = {"cols": "9", "rows": "6", "square_mm": "25"}   # 默认 9x6 / 25mm
        for i, (label, key, width) in enumerate(
                [("列", "cols", 5), ("行", "rows", 5), ("边长mm", "square_mm", 7)]):
            ttk.Label(box_cb, text=label).grid(row=0, column=i * 2, padx=(8, 2), pady=3)
            ent = ttk.Entry(box_cb, width=width)
            ent.insert(0, defaults[key])
            setattr(self, f"ent_{key}", ent)
            ent.grid(row=0, column=i * 2 + 1, padx=(0, 6), pady=3)

        # ---- 预览与采集 ----
        box_cap = ttk.LabelFrame(left, text=" 预览与采集 ")
        box_cap.pack(fill="x", **pad)
        row = ttk.Frame(box_cap); row.pack(fill="x", **pad)
        self.btn_preview = ttk.Button(row, text="开始预览", width=10,
                                      command=self.toggle_preview)
        self.btn_preview.pack(side="left", **pad)
        ttk.Button(row, text="抓取一张", width=10,
                   command=self.capture_one).pack(side="left", **pad)
        ttk.Button(row, text="加载目录全部", width=12,
                   command=self.load_folder_images).pack(side="left", **pad)
        ttk.Button(row, text="生成棋盘格", width=10,
                   command=self.gen_board_image).pack(side="left", **pad)
        self.var_cap_status = tk.StringVar(value="预览未启动")
        ttk.Label(box_cap, textvariable=self.var_cap_status,
                  foreground="#0a7a0a").pack(anchor="w", **pad)

        # ---- 已采集列表 ----
        box_list = ttk.LabelFrame(left, text=" 已采集图像 (有效帧) ")
        box_list.pack(fill="both", expand=True, **pad)
        frame_list = ttk.Frame(box_list); frame_list.pack(fill="both", expand=True)
        sb = ttk.Scrollbar(frame_list)
        self.listbox = tk.Listbox(frame_list, height=6, width=44, yscrollcommand=sb.set,
                                  font=self._fixed_font)
        sb.config(command=self.listbox.yview)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        row = ttk.Frame(box_list); row.pack(fill="x", **pad)
        ttk.Button(row, text="删除选中", width=10,
                   command=self.delete_selected).pack(side="left", **pad)
        ttk.Button(row, text="清空全部", width=10,
                   command=self.clear_all).pack(side="left", **pad)
        ttk.Button(row, text="导出图像", width=10,
                   command=self.export_images).pack(side="left", **pad)
        self.var_count = tk.StringVar(value="0 张")
        ttk.Label(row, textvariable=self.var_count).pack(side="right", **pad)

        # ---- 多轮平均标定 ----
        box_cal = ttk.LabelFrame(left, text=" 多轮平均标定 ")
        box_cal.pack(fill="x", **pad)
        ttk.Label(box_cal, text="轮数").pack(side="left", **pad)
        self.ent_runs = ttk.Entry(box_cal, width=5)
        self.ent_runs.insert(0, "10")
        self.ent_runs.pack(side="left", **pad)
        ttk.Label(box_cal, text="每轮子集比例").pack(side="left", **pad)
        self.ent_ratio = ttk.Entry(box_cal, width=5)
        self.ent_ratio.insert(0, "0.8")
        self.ent_ratio.pack(side="left", **pad)
        self.btn_cal = ttk.Button(box_cal, text="开始标定", width=10,
                                  command=self.start_calibration)
        self.btn_cal.pack(side="right", **pad)

        # ---- 结果显示 ----
        box_res = ttk.LabelFrame(left, text=" 标定结果 ")
        box_res.pack(fill="both", expand=True, **pad)
        # 操作按钮放在文本框上方, 避免小屏幕/缩放时被窗口底部裁掉看不见
        row = ttk.Frame(box_res); row.pack(fill="x", **pad)
        ttk.Button(row, text="保存 YAML", width=10,
                   command=self.save_yaml).pack(side="left", **pad)
        ttk.Button(row, text="去畸变预览", width=10,
                   command=self.show_undistorted).pack(side="left", **pad)
        self.txt = tk.Text(box_res, height=9, width=48, font=self._fixed_font,
                           state="disabled", wrap="none")
        self.txt.pack(fill="both", expand=True, **pad)

        # ================= 右列: 预览画布 =================
        right = ttk.LabelFrame(main, text=" 预览 (绿线=角点检测成功) ")
        right.pack(side="right", fill="y", **pad)
        self.lbl_preview = ttk.Label(right)
        self.lbl_preview.pack(**pad)
        self.var_cam_status = tk.StringVar(value="—")
        ttk.Label(right, textvariable=self.var_cam_status,
                  font=self._fixed_font, foreground="#555").pack(**pad)

    # ------------------------------------------------------------------
    #                        参数记忆 (json)
    # ------------------------------------------------------------------
    def _collect_settings(self):
        return {key: w.get() for key, w in self._setting_widgets().items()}

    def _setting_widgets(self):
        return {"src": self.var_src, "src_val": self.ent_src, "cols": self.ent_cols,
                "rows": self.ent_rows, "square_mm": self.ent_square_mm,
                "runs": self.ent_runs, "ratio": self.ent_ratio}

    def _load_settings(self):
        """启动时恢复上次使用的参数, 方便换相机时直接用。"""
        try:
            with open(SETTINGS_FILE) as f:
                s = json.load(f)
            widgets = self._setting_widgets()
            for key, val in s.items():
                w = widgets.get(key)
                if w is None:
                    continue
                if isinstance(w, tk.StringVar):      # 单选按钮变量
                    w.set(str(val))
                else:                                # 输入框
                    w.delete(0, "end")
                    w.insert(0, str(val))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        self._on_src_change()

    def _save_settings(self):
        try:
            
            with open(SETTINGS_FILE, "w") as f:
                json.dump(self._collect_settings(), f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    # ------------------------------------------------------------------
    #                        预览线程 (采集 + 检测)
    # ------------------------------------------------------------------
    def _src_kwargs(self):
        """解析当前图像源配置, 返回 (类型, 值)。"""
        val = self.ent_src.get().strip()
        return self.var_src.get(), val

    def _on_src_change(self):
        """切换图像源时, 更新输入框标签/内容与按钮状态。"""
        src, val = self._src_kwargs()
        if src == "esp32":
            self.lbl_src.config(text="IP:")
            self.btn_browse.pack_forget()
            if not self.ent_src.get().strip():
                self.ent_src.insert(0, DEFAULT_ESP32_IP)
        elif src == "usb":
            self.lbl_src.config(text="索引:")
            self.btn_browse.pack_forget()
        else:
            self.lbl_src.config(text="目录:")
            self.btn_browse.pack(side="right", padx=2)

    def _browse_folder(self):
        path = filedialog.askdirectory(title="选择标定图像目录")
        if path:
            self.ent_src.delete(0, "end")
            self.ent_src.insert(0, path)

    def toggle_preview(self):
        """启动/停止预览工作线程。"""
        if self._preview_run.is_set():
            self._preview_run.clear()
            self.btn_preview.config(text="开始预览")
            return
        # 校验棋盘参数 (预览时也需要用来检测角点)
        try:
            self._read_board_params()
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return
        # 校验图像源参数
        src, val = self._src_kwargs()
        if src == "esp32" and not val:
            messagebox.showerror("参数错误", "请填写 ESP32 的 IP 地址")
            return
        if src == "usb":
            try:
                int(val or 0)
            except ValueError:
                messagebox.showerror("参数错误", f"USB 索引必须是整数: '{val}'")
                return
        self._preview_run.set()
        self.btn_preview.config(text="停止预览")
        self._worker = threading.Thread(target=self._preview_loop,
                                        args=(src, val), daemon=True)
        self._worker.start()
        self.var_cap_status.set("预览运行中...")

    def _preview_loop(self, src, val):
        """工作线程: 按图像源循环抓帧 -> 角点检测 -> 画叠加 -> 投递到队列。

        注意: 本线程绝不直接操作 tkinter 控件, 一切 UI 更新经由队列
        转交主线程 (tkinter 非线程安全)。
        """
        cap = None
        pattern = (self._params["cols"], self._params["rows"])
        try:
            while self._preview_run.is_set():
                frame = None
                if src == "esp32":
                    try:
                        with urllib.request.urlopen(
                                f"http://{val}{ESP32_SNAPSHOT_API}", timeout=5) as r:
                            frame = cv2.imdecode(
                                np.frombuffer(r.read(), np.uint8), cv2.IMREAD_COLOR)
                    except Exception as e:
                        self.q.put(("cam_status", f"ESP32 连接失败: {e}"))
                        time.sleep(2)
                        continue
                    time.sleep(ESP32_FETCH_GAP)
                elif src == "usb":
                    if cap is None:                       # 首次进入才打开设备
                        cap = cv2.VideoCapture(int(val) if val else 0)
                    ok, frame = cap.read()
                    if not ok:
                        self.q.put(("cam_status", "USB 读取失败, 重试中..."))
                        time.sleep(USB_RETRY_GAP)
                        continue
                else:                                     # folder 模式: 显示目录第一张
                    if frame is None:
                        frame = self._first_folder_image(val)
                        if frame is None:
                            self.q.put(("cam_status", "目录中没有图像"))
                            time.sleep(2)
                            continue
                    self._preview_run.clear()             # 静态图, 显示一次即可

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners = detect_corners(gray, pattern)
                disp = frame.copy()
                if corners is not None:                   # 绿色角点连线叠加
                    cv2.drawChessboardCorners(disp, pattern, corners, True)
                    self.q.put(("frame", disp, frame, corners, True))
                else:                                     # 红框提示未检测到
                    h, w = frame.shape[:2]
                    cv2.rectangle(disp, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
                    self.q.put(("frame", disp, frame, None, False))
        finally:
            if cap is not None:
                cap.release()

    def _first_folder_image(self, folder):
        """目录模式下取第一张图作为静态预览。"""
        if not os.path.isdir(folder):
            return None
        for name in sorted(os.listdir(folder)):
            if name.rsplit(".", 1)[-1].lower() in IMAGE_EXT:
                return cv2.imread(os.path.join(folder, name))
        return None

    def _poll_queue(self):
        """主线程周期任务: 消费队列消息, 更新 UI (每 PREVIEW_POLL_MS)。

        性能要点: 预览帧在队列里可能积压 (工作线程 > UI 刷新率时),
        逐帧重绘在 WSLg 的 RDP 显示链路上开销很大, 会明显卡顿;
        因此旧帧全部丢弃, 每个周期只重绘最新一帧 (帧合并)。
        """
        newest_frame = None
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "frame":
                    newest_frame = msg            # 只留最新, 旧帧丢弃
                elif kind == "cam_status":
                    self.var_cam_status.set(msg[1])
                elif kind == "append":
                    self._append_image(msg[1], msg[2], msg[3])
                elif kind == "add_done":
                    _, n_ok, n_fail = msg
                    self.var_cap_status.set(
                        f"目录加载完成: {n_ok} 张有效, {n_fail} 张无角点")
                elif kind == "calib_done":
                    _, result = msg
                    self._show_result(result)
                elif kind == "log":
                    self.var_cap_status.set(msg[1])
        except queue.Empty:
            pass
        if newest_frame is not None:              # 每周期最多重绘最新一帧
            _, disp, orig, corners, ok = newest_frame
            self._show_image(disp)
            with self._latest_lock:               # 缓存供"抓取"使用
                self._latest = (disp, corners)
            self._latest_orig = orig
            self.var_cam_status.set(
                f"{disp.shape[1]}x{disp.shape[0]}  "
                + ("检测到角点" if ok else "未检测到角点"))
        self.root.after(PREVIEW_POLL_MS, self._poll_queue)

    def _show_image(self, bgr):
        """把 BGR ndarray 显示到预览画布 (等比缩放 -> PNG -> PhotoImage)。"""
        h, w = bgr.shape[:2]
        scale = min(PREVIEW_MAX_W / w, PREVIEW_MAX_H / h, 1.0)
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        png = cv2.imencode(".png", bgr)[1].tobytes()
        self._photo = tk.PhotoImage(data=png)             # 保留引用!
        self.lbl_preview.config(image=self._photo)

    # ------------------------------------------------------------------
    #                          采集与列表管理
    # ------------------------------------------------------------------
    def _read_board_params(self):
        """读取并校验棋盘格参数, 存入 self._params。"""
        try:
            cols = int(self.ent_cols.get())
            rows = int(self.ent_rows.get())
            square_mm = float(self.ent_square_mm.get())
        except ValueError:
            raise ValueError("棋盘参数必须是数字 (边长可为小数)")
        if not (3 <= cols <= 20 and 3 <= rows <= 20):
            raise ValueError("内角点行列数应在 3~20 之间")
        if not (0.5 <= square_mm <= 1000):
            raise ValueError("格子边长 (mm) 应在 0.5~1000 之间")
        self._params = {"cols": cols, "rows": rows, "square": square_mm / 1000.0}
        return self._params

    @staticmethod
    def _draw_board_png(path, cols, rows, px=60):
        """生成可打印/可投屏的棋盘格 PNG。

        cols/rows: 内角点数 (实际绘制 (cols+1)x(rows+1) 个格子);
        px: 每格边长 (像素)。白色外边距 2 格, 便于检测和固定。
        """
        margin = 2 * px
        n_cols, n_rows = cols + 1, rows + 1
        w, h = n_cols * px + 2 * margin, n_rows * px + 2 * margin
        img = np.full((h, w), 255, np.uint8)
        for r in range(n_rows):
            for c in range(n_cols):
                if (r + c) % 2 == 0:
                    y, x = margin + r * px, margin + c * px
                    img[y:y + px, x:x + px] = 0
        cv2.imwrite(path, img)

    def gen_board_image(self):
        """按当前配置的行列数生成棋盘格 PNG。

        用法提示: 保存后可直接在笔记本/显示器上全屏打开, 对屏幕拍摄标定
        (屏幕是平面, 等效标定板)。注意亮度调高、关闭护眼/夜间模式、避免反光。
        """
        try:
            params = self._read_board_params()
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile="chessboard.png",
            filetypes=[("PNG 图片", "*.png")])
        if not path:
            return
        self._draw_board_png(path, params["cols"], params["rows"])
        self.var_cap_status.set(
            f"棋盘格已生成 -> {path} (可全屏显示在屏幕上拍摄标定)")
        messagebox.showinfo(
            "生成成功",
            f"已保存: {path}\n\n"
            "用法: 在笔记本/显示器上全屏打开这张图,\n"
            "相机对着屏幕变换角度拍摄即可标定。\n"
            "注意: 亮度调高, 关闭夜间模式/护眼模式, 避免反光。")

    def _append_image(self, disp, corners, orig):
        """加入一张有效图像 (disp=带角点叠加的显示图, orig=原始帧)。"""
        self.images.append({"frame": disp, "orig": orig,
                            "corners": corners, "ok": True})
        self._refresh_list()

    def capture_one(self):
        """抓取当前预览帧: 有角点才入队 (无角点的图对标定毫无贡献)。"""
        with self._latest_lock:
            snap = self._latest
        if snap is None:
            messagebox.showinfo("提示", "暂无预览帧, 请先点击 [开始预览]")
            return
        disp, corners = snap
        if corners is None:
            self.var_cap_status.set("最近一张未检测到角点, 未加入 (换个姿态再试)")
            return
        # disp 叠加了绿色角点连线, 另存一份原始帧供去畸变预览使用
        orig = self._latest_orig
        self._append_image(disp, corners, orig)

    def load_folder_images(self):
        """批量加载目录图像 (后台线程逐张检测, 避免卡界面)。"""
        src, val = self._src_kwargs()
        folder = val if src == "folder" else self.ent_src.get().strip()
        if not os.path.isdir(folder):
            messagebox.showerror("错误", f"目录不存在: {folder}")
            return
        try:
            self._read_board_params()
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return
        pattern = (self._params["cols"], self._params["rows"])
        threading.Thread(target=self._load_folder_worker,
                         args=(folder, pattern), daemon=True).start()
        self.var_cap_status.set("正在加载目录图像...")

    def _load_folder_worker(self, folder, pattern):
        """工作线程: 逐张读取 + 检测, 每张独立投递, 主线程统一入列表。"""
        n_ok = n_fail = 0
        for name in sorted(os.listdir(folder)):
            if name.rsplit(".", 1)[-1].lower() not in IMAGE_EXT:
                continue
            img = cv2.imread(os.path.join(folder, name))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = detect_corners(gray, pattern)
            if corners is not None:
                disp = img.copy()
                cv2.drawChessboardCorners(disp, pattern, corners, True)
                self.q.put(("append", disp, corners, img))
                n_ok += 1
            else:
                n_fail += 1
        self.q.put(("add_done", n_ok, n_fail))

    def _refresh_list(self):
        """重建列表显示并更新计数 (数据源: self.images)。"""
        self.listbox.delete(0, "end")
        for i, it in enumerate(self.images):
            h, w = it["frame"].shape[:2]
            self.listbox.insert("end", f"{i + 1:03d}  {w}x{h}  "
                                f"{it['corners'].shape[0]} 角点  "
                                + ("OK" if it["ok"] else "--"))
        self.var_count.set(f"{len(self.images)} 张")

    def delete_selected(self):
        for i in sorted(self.listbox.curselection(), reverse=True):
            del self.images[i]
        self._refresh_list()

    def clear_all(self):
        if self.images and messagebox.askyesno("确认", "清空全部已采集图像?"):
            self.images.clear()
            self._refresh_list()

    def export_images(self):
        """把已采集的原始图像导出到文件夹 (用于离线复现/排查标定问题)。"""
        if not self.images:
            messagebox.showinfo("提示", "还没有已采集图像")
            return
        folder = filedialog.askdirectory(title="选择导出目录")
        if not folder:
            return
        n = 0
        for i, it in enumerate(self.images):
            path = os.path.join(folder, f"img_{i:03d}.jpg")
            if cv2.imwrite(path, it["orig"]):
                n += 1
        self.var_cap_status.set(f"已导出 {n} 张原始图像 -> {folder}")

    # ------------------------------------------------------------------
    #                            标定与结果
    # ------------------------------------------------------------------
    def start_calibration(self):
        """读取参数后, 在后台线程执行多轮平均标定。"""
        try:
            params = self._read_board_params()
            n_runs = max(1, int(self.ent_runs.get()))
            ratio = float(self.ent_ratio.get())
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return
        if not (0.3 <= ratio <= 1.0):
            messagebox.showerror("参数错误", "子集比例应在 0.3~1.0 之间")
            return
        ok_imgs = [it for it in self.images if it["ok"]]
        if len(ok_imgs) < MIN_OK_IMAGES:
            messagebox.showwarning(
                "图像不足",
                f"有效图像仅 {len(ok_imgs)} 张, 至少需要 {MIN_OK_IMAGES} 张。\n"
                "建议: 多姿态采集 15~25 张 (位置/角度/距离都要变化)。")
            return
        self.btn_cal.config(state="disabled")
        self.var_cap_status.set(f"标定中... ({n_runs} 轮 x {ratio:.0%} 子集)")
        ops = [_board_corners_3d(params["cols"], params["rows"],
                                       params["square"])] * len(ok_imgs)
        ips = [it["corners"] for it in ok_imgs]
        threading.Thread(target=self._calib_worker,
                         args=(ops, ips, n_runs, ratio, params),
                         daemon=True).start()

    def _calib_worker(self, ops, ips, n_runs, ratio, params):
        """工作线程: 多轮平均标定 (算法在 calibration.calibrate_multi)。"""
        h, w = [it for it in self.images if it["ok"]][0]["frame"].shape[:2]
        size = (w, h)
        try:
            result = calibrate_multi(ops, ips, size,
                                           n_runs=n_runs, subset_ratio=ratio)
        except Exception as e:
            self.q.put(("log", f"标定失败: {e}"))
            self.q.put(("calib_done", None))
            return
        result["size"] = size
        result["params"] = params
        result["n_images"] = len(ops)
        self.q.put(("calib_done", result))

    def _show_result(self, result):
        """在结果文本框展示多轮标定统计。"""
        self.btn_cal.config(state="normal")
        if result is None:
            self.var_cap_status.set("标定失败, 见上方提示")
            return
        self._result = result
        K, std = result["K_mean"], result["K_std"]
        dm, ds = result["dist_mean"], result["dist_std"]
        rms = np.array(result["rms_list"])
        p = result["params"]
        lines = [
            f"图像: {result['n_images']} 张 | {result['size'][0]}x{result['size'][1]}"
            f" | 棋盘 {p['cols']}x{p['rows']} | 格边长 {p['square'] * 1000:.1f} mm",
            f"多轮: {result['runs_ok']}/{result['runs_total']} 轮成功"
            f" | RMS {rms.mean():.3f} ± {rms.std():.3f} px",
            "-" * 52,
            f"fx  = {K[0, 0]:8.2f} ± {std[0]:6.2f}   (相对 {std[0] / K[0, 0] * 100:.2f}%)",
            f"fy  = {K[1, 1]:8.2f} ± {std[1]:6.2f}   (相对 {std[1] / K[1, 1] * 100:.2f}%)",
            f"cx  = {K[0, 2]:8.2f} ± {std[2]:6.2f}",
            f"cy  = {K[1, 2]:8.2f} ± {std[3]:6.2f}",
            "-" * 52,
            "畸变 (均值 ± 标准差):",
            f"  k1 = {dm[0]: .4f} ± {ds[0]:.4f}",
            f"  k2 = {dm[1]: .4f} ± {ds[1]:.4f}",
            f"  p1 = {dm[2]: .4f} ± {ds[2]:.4f}",
            f"  p2 = {dm[3]: .4f} ± {ds[3]:.4f}",
            f"  k3 = {dm[4]: .4f} ± {ds[4]:.4f}",
            "-" * 52,
        ]
        # 按数值判定哪类参数不够稳 (标准差阈值: 焦距看相对值, 其余看绝对值)
        warn = []
        if std[0] / K[0, 0] > 0.005 or std[1] / K[1, 1] > 0.005:
            warn.append("fx/fy (需要不同距离/倾斜)")
        if std[2] > 1.5 or std[3] > 1.5:
            warn.append("cx/cy (需要棋盘覆盖画面四角)")
        if ds[0] > 0.01:
            warn.append("k1 (需要画面四角+倾斜)")
        if ds[2] > 0.002 or ds[3] > 0.002:
            warn.append("p1/p2 (需要多角度倾斜)")
        if warn:
            lines.append("标准差偏大: " + "; ".join(warn))
            lines.append("建议按括号内提示补拍姿态多样化的图像后再标定。")
        else:
            lines.append("各参数标准差均在健康范围, 结果稳定, 可直接使用。")
        self._set_text("\n".join(lines))
        self.var_cap_status.set("标定完成! 可保存 YAML 或查看去畸变效果")

    def _set_text(self, content):
        self.txt.config(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", content)
        self.txt.config(state="disabled")

    def save_yaml(self):
        """把多轮平均结果保存为 YAML (FileStorage, ROS/C++ 可直接读取)。"""
        if self._result is None:
            messagebox.showinfo("提示", "请先完成一次标定")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".yaml", initialfile="calib_result.yaml",
            filetypes=[("YAML", "*.yaml *.yml")])
        if not path:
            return
        r = self._result
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        fs.write("image_width", r["size"][0])
        fs.write("image_height", r["size"][1])
        fs.write("camera_matrix", r["K_mean"])
        fs.write("dist_coeffs", r["dist_mean"].reshape(1, 5))
        # 多轮标准差 (诊断用: 判断各参数是否稳定)
        fs.write("camera_matrix_std_fx_fy_cx_cy", r["K_std"].reshape(1, 4))
        fs.write("dist_coeffs_std", r["dist_std"].reshape(1, 5))
        fs.write("rms_reproj_error_mean", float(np.mean(r["rms_list"])))
        fs.write("rms_reproj_error_std", float(np.std(r["rms_list"])))
        fs.write("n_images", r["n_images"])
        fs.write("n_runs", r["runs_ok"])
        fs.write("chessboard_cols", r["params"]["cols"])
        fs.write("chessboard_rows", r["params"]["rows"])
        fs.write("square_size_m", float(r["params"]["square"]))
        fs.release()
        self.var_cap_status.set(f"结果已保存 -> {path}")

    def show_undistorted(self):
        """用平均内参/畸变对最近一张有效图去畸变, 暂停预览以保持显示。"""
        if self._result is None:
            messagebox.showinfo("提示", "请先完成一次标定")
            return
        ok_imgs = [it for it in self.images if it["ok"]]
        if not ok_imgs:
            messagebox.showinfo("提示", "没有可用图像")
            return
        if self._preview_run.is_set():               # 暂停预览, 让去畸变图驻留
            self.toggle_preview()
        K, dist = self._result["K_mean"], self._result["dist_mean"]
        img = ok_imgs[-1]["orig"]                    # 用原始帧 (无角点叠加线)
        h, w = img.shape[:2]
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1.0)
        fixed = cv2.undistort(img, K, dist, None, newK)
        compare = np.hstack([img, fixed])            # 左原图 | 右去畸变
        self._show_image(compare)
        self.var_cam_status.set("去畸变对比 (左: 原图, 右: 校正后)")

    # ------------------------------------------------------------------
    #                              退出
    # ------------------------------------------------------------------
    def on_close(self):
        """退出前停线程、释放相机、保存参数。"""
        self._preview_run.clear()
        if self._worker is not None:
            self._worker.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
        self._save_settings()
        self.root.destroy()


def _pick_cjk_fonts():
    """探测系统中可用的中文字体族 (WSL/极简系统常缺中文字体, 导致界面全为方块)。

    返回 (常规字体族, 等宽中文字体族); 都找不到时返回 (None, None),
    此时回退 Tk 默认字体 (需用户自行安装字体, 见文件头说明)。
    """
    fams = set(tkfont.families())
    reg_cands = ["Microsoft YaHei", "微软雅黑", "Noto Sans CJK SC", "Noto Sans SC",
                 "WenQuanYi Zen Hei", "文泉驿正黑", "WenQuanYi Micro Hei",
                 "SimHei", "黑体", "AR PL UMing CN"]
    mono_cands = ["Noto Sans Mono CJK SC", "WenQuanYi Zen Hei Mono",
                  "Microsoft YaHei UI"]
    reg = next((c for c in reg_cands if c in fams), None)
    mono = next((c for c in mono_cands if c in fams), reg)
    return reg, mono


def _probe_vcxsrv(host="127.0.0.1", port=6000, timeout=0.3):
    """探测 Windows 侧是否运行着 VcXsrv (X server 监听 6000 端口)。

    背景: WSLg 的显示走 RDP 编码流水线, 频繁重绘会卡顿;
    VcXsrv 是本地 X11 直连, 明显更流畅。
    返回 True 表示有原生 X server 可用。
    """
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main():
    # 显示路径自动选择: VcXsrv 已启动 -> 用它 (127.0.0.1:0, 须 WSL2 mirrored 网络);
    # 否则回落 WSLg 的 :0。这样用户双击 vcxsrv.xlaunch 即获得流畅模式, 零配置。
    if _probe_vcxsrv():
        os.environ["DISPLAY"] = "127.0.0.1:0"
        print("[显示] 使用 VcXsrv (X11 直连, 流畅模式)")
    root = tk.Tk()
    CalibGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

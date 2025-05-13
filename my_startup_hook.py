# -*- coding: utf-8 -*-
# PyInstaller 启动运行时 hook：my_startup_hook.py

import sys
import os

# ──────────────────────────────────────────────────────────────────────────────
# 0) 在冻结（onefile）模式下，强制设置 tcl/tk 的库目录
# ──────────────────────────────────────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS  # PyInstaller 解压目录
    # 这两个目录必须跟你的 hook-tkinter.py 里收集的保持一致
    os.environ['TCL_LIBRARY'] = os.path.join(base_path, 'tcl', 'tcl8.6')
    os.environ['TK_LIBRARY'] = os.path.join(base_path, 'tcl', 'tk8.6')

# ──────────────────────────────────────────────────────────────────────────────
# 接下来再导入 tkinter，就会正确在上面两个目录里寻找 init.tcl
# ──────────────────────────────────────────────────────────────────────────────
import tkinter as tk
import threading
import traceback
import logging
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 1) 显示 Splash
# ──────────────────────────────────────────────────────────────────────────────
# 获取打包后资源目录中 splash.png 的路径。
# 在 PyInstaller onefile 模式下，资源被提取到临时文件夹，路径由 sys._MEIPASS 提供。
# 如果未打包 (开发模式运行)，则直接使用当前文件所在目录。
base_path = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
splash_path = os.path.join(base_path, 'splash.png')

# 创建 Tkinter 主窗口用于显示启动画面
splash_root = tk.Tk()
splash_root.overrideredirect(True)  # 去除窗口边框和标题栏，使窗口无边框（适合作为启动画面）
# 加载启动图片
splash_image = None
try:
    splash_image = tk.PhotoImage(file=splash_path)
except Exception as e:
    # 如果图片加载失败，不终止流程；可以在此处打印或记录错误，但在无控制台时仅跳过
    splash_image = None

# 如果成功加载了图片，创建标签显示图片；否则创建一个空标签占位
if splash_image:
    splash_label = tk.Label(splash_root, image=splash_image)
else:
    splash_label = tk.Label(splash_root, text="Loading...", font=("Arial", 18))
splash_label.pack()

# 将启动画面窗口居中显示在屏幕中央
screen_width = splash_root.winfo_screenwidth()
screen_height = splash_root.winfo_screenheight()
# 获取图片尺寸（如果未加载图片则使用标签尺寸）
img_width = splash_image.width() if splash_image else splash_label.winfo_reqwidth()
img_height = splash_image.height() if splash_image else splash_label.winfo_reqheight()
# 计算居中坐标
pos_x = (screen_width - img_width) // 2
pos_y = (screen_height - img_height) // 2
# 设置窗口初始大小和位置
splash_root.geometry(f"{img_width}x{img_height}+{pos_x}+{pos_y}")

# 显示窗口并刷新一帧，确保启动画面及时呈现
splash_root.update_idletasks()  # 刷新布局
splash_root.deiconify()         # 显示窗口（Tk窗口创建初始为隐藏状态，当使用 overrideredirect 时需要显式deiconify）
splash_root.update()            # 强制刷新界面，立即呈现启动画面

# -------------- Monkey-Patch Tk.__init__ 逻辑 --------------
# 保存 tkinter.Tk 原始的 __init__ 方法，以便后续调用
_original_tk_init = tk.Tk.__init__

def _new_tk_init(self, *args, **kwargs):
    """
    自定义 Tk.__init__ 方法包装，用于在主窗口初始化后关闭启动画面。
    """
    # 调用 Tk 原始初始化，使主窗口正常创建
    _original_tk_init(self, *args, **kwargs)
    # 主窗口已创建，立即销毁启动画面窗口
    try:
        if splash_root is not None:
            splash_root.destroy()
    except Exception as e:
        # 如已销毁或发生异常，忽略即可
        pass

# 将 Tk.__init__ monkey-patch 替换为自定义方法
tk.Tk.__init__ = _new_tk_init

# -------------- 全局异常捕获和日志记录 --------------
# 确定日志文件路径：优先写入程序所在目录（若为打包运行，则 sys.executable 为打包后的可执行文件路径）
if getattr(sys, 'frozen', False):
    # 冻结（打包）状态，使用可执行文件所在目录
    log_dir = os.path.dirname(sys.executable)
else:
    # 非冻结状态，使用当前工作目录
    log_dir = os.getcwd()
log_file = os.path.join(log_dir, "error.log")

# 定义全局异常处理函数
def log_uncaught_exception(exc_type, exc_value, exc_traceback):
    """
    捕获未处理异常并写入日志文件。
    """
    # 格式化异常信息
    error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "="*60 + "\n")
            f.write("Unhandled Exception:\n")
            f.write(error_message)
            f.write("\n")
    except Exception:
        pass  # 若日志写入失败则忽略（可能无权限等）
    # 调用默认的异常处理行为（打印到 stderr），若需要可注释掉下一行
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# 设置 sys.excepthook 为自定义的处理函数
sys.excepthook = log_uncaught_exception

# 针对线程异常（Python 3.8+提供 threading.excepthook）进行捕获记录
if hasattr(threading, 'excepthook'):
    _orig_thread_excepthook = threading.excepthook

    def log_thread_exception(args):
        """
        捕获线程中的未处理异常并写入日志文件。
        """
        # args.exc_type, args.exc_value, args.exc_traceback 分别为异常的类型、实例和追溯信息
        error_message = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n" + "="*60 + "\n")
                f.write(f"Unhandled Exception in thread '{args.thread.name}':\n")
                f.write(error_message)
                f.write("\n")
        except Exception:
            pass
        # 调用原本 threading.excepthook（如有需要，可保留默认行为）
        if _orig_thread_excepthook:
            _orig_thread_excepthook(args)

    # 设置线程异常处理hook
    threading.excepthook = log_thread_exception

# （钩子配置完毕，后续将自动继续运行主应用的 gui_app.py）

# my_startup_hook.py

import sys
import os
import threading
import logging
import traceback
from pathlib import Path

# ------------------------------------------------------------
# 1) Splash Screen 部分
# ------------------------------------------------------------
try:
    import tkinter as tk
    from PIL import Image, ImageTk
except ImportError:
    # 如果环境里没装 Pillow 或 tkinter，就跳过
    tk = None

def _show_splash():
    if not tk:
        return
    try:
        # 找到打包后临时目录里的 splash.png
        if getattr(sys, "frozen", False):
            img_path = os.path.join(sys._MEIPASS, "splash.png")
        else:
            # 开发态直接找当前目录
            img_path = os.path.join(os.path.dirname(__file__), "splash.png")

        # 创建一个无边框的窗口来显示图片
        root = tk.Tk()
        root.overrideredirect(True)

        # 打开并展示图片
        img = Image.open(img_path)
        w, h = img.size
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        x, y = (sw - w) // 2, (sh - h) // 2
        root.geometry(f"{w}x{h}+{x}+{y}")

        tk_img = ImageTk.PhotoImage(img)
        lbl = tk.Label(root, image=tk_img)
        lbl.image = tk_img
        lbl.pack()

        # 全局保存一下 root，以便后来销毁
        sys._splash_root = root
        root.mainloop()
    except Exception:
        # 若中途出错，静默跳过，不影响主程序
        pass

# 后台线程启动 splash
threading.Thread(target=_show_splash, daemon=True).start()

# 在用户第一个 Tk() 实例化的时候，把 splash 关掉
if tk:
    _orig_tk_init = tk.Tk.__init__

    def _patched_tk_init(self, *args, **kwargs):
        try:
            if hasattr(sys, "_splash_root"):
                sys._splash_root.destroy()
                del sys._splash_root
        except Exception:
            pass
        return _orig_tk_init(self, *args, **kwargs)

    tk.Tk.__init__ = _patched_tk_init


# ------------------------------------------------------------
# 2) 全局异常捕获部分（保持你原来的日志逻辑）
# ------------------------------------------------------------

# 日志目录
log_dir = Path(os.getenv("APPDATA", Path.home() / ".local")) / "CellGPS"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "error.log"

# 设置 logger
logger = logging.getLogger("CellGPSStartup")
logger.setLevel(logging.DEBUG)

# 文件记录 UTF-8
file_hdl = logging.FileHandler(log_file, encoding="utf-8")
fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_hdl.setFormatter(fmt)
logger.addHandler(file_hdl)

# 可选地也输出到 stderr
console_hdl = logging.StreamHandler(sys.stderr)
console_hdl.setFormatter(fmt)
logger.addHandler(console_hdl)

def _global_exc_hook(exc_type, exc_value, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.critical(f"未捕获的异常:\n{msg}")
    # 然后走原始的 excepthook（把错误打印到控制台）
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _global_exc_hook

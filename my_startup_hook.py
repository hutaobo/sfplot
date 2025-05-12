# my_startup_hook.py
import sys
import os
import threading
import logging
import traceback
from pathlib import Path

# -----------------------------------------------------------------------------
# 1) 在最早阶段显示 Splash
# -----------------------------------------------------------------------------
_splash_root = None

def _show_splash():
    global _splash_root
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError:
        return
    # 创建无边框窗口
    root = tk.Tk()
    root.overrideredirect(True)

    # 载入图片（onefile 模式下存在于 sys._MEIPASS）
    base = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
    img_path = os.path.join(base, 'splash.png')
    img = Image.open(img_path)
    photo = ImageTk.PhotoImage(img)

    # 把图片放到 Label，居中显示
    label = tk.Label(root, image=photo)
    label.image = photo

    w, h = img.width, img.height
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = (screen_w - w) // 2
    y = (screen_h - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.update()

    _splash_root = root

# 在单独线程里显示（避免阻塞 hook 自身加载）
threading.Thread(target=_show_splash, daemon=True).start()


# -----------------------------------------------------------------------------
# 2) 当主窗口真正创建后，销毁 Splash
#    我们 Monkey Patch Tk.__init__，在第一个窗口启动时自动关掉它
# -----------------------------------------------------------------------------
def _destroy_splash():
    global _splash_root
    if _splash_root:
        try:
            _splash_root.destroy()
        except Exception:
            pass
        finally:
            _splash_root = None

def _wrap_tk_init(orig_init):
    def new_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # 第一个 Tk 窗口一创建，就关掉 Splash
        _destroy_splash()
    return new_init

try:
    import tkinter
    tkinter.Tk.__init__ = _wrap_tk_init(tkinter.Tk.__init__)
except Exception:
    pass


# -----------------------------------------------------------------------------
# 3) 你的全局异常日志逻辑
# -----------------------------------------------------------------------------
# 日志文件夹
log_dir = Path(os.getenv("APPDATA", Path.home() / ".local")) / "CellGPS"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "error.log"

logger = logging.getLogger("CellGPSStartup")
logger.setLevel(logging.DEBUG)

# 文件 handler
file_handler = logging.FileHandler(log_file, encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 可选：也输出到 stderr
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def global_exception_handler(exctype, value, tb):
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    logger.critical(f"未捕获的异常:\n{error_msg}")
    # 然后交给默认处理
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = global_exception_handler

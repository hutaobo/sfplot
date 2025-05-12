# splash_hook.py

import os
import sys

# —— 1) 先设好 Tcl/Tk 库的位置 ——
if getattr(sys, "frozen", False):
    # 打包时，PyInstaller 会把 tcl/tk 库解压到 sys._MEIPASS 下的 tcl 子目录
    base = sys._MEIPASS
    os.environ.setdefault("TCL_LIBRARY", os.path.join(base, "tcl", "tcl8.6"))
    os.environ.setdefault("TK_LIBRARY",  os.path.join(base, "tcl", "tk8.6"))

# —— 2) 再导入 tkinter ——
import threading
import tkinter as tk
from PIL import Image, ImageTk

def _show_splash():
    # 创建一个无边框窗口
    root = tk.Tk()
    root.overrideredirect(True)

    # 加载图片（Onefile 模式下读自 _MEIPASS）
    if getattr(sys, "frozen", False):
        img_path = os.path.join(sys._MEIPASS, "splash.png")
    else:
        img_path = "splash.png"
    img = Image.open(img_path)
    w, h = img.size

    # 居中
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w) // 2
    y = (sh - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    # 显示
    tk_img = ImageTk.PhotoImage(img)
    lbl = tk.Label(root, image=tk_img)
    lbl.image = tk_img
    lbl.pack()

    # 保存引用，供 my_startup_hook.py 销毁
    sys._splash_root = root

    # 进入主循环（线程自动退出时窗口关闭）
    root.mainloop()

# 用后台线程启动，保证后续运行时 hook 和主脚本能并行加载
t = threading.Thread(target=_show_splash, daemon=True)
t.start()

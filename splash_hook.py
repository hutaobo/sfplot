# splash_hook.py
import sys, os, threading, time
# Tkinter 以及 PIL
import tkinter as tk
from PIL import Image, ImageTk

def _show_splash():
    root = tk.Tk()
    root.overrideredirect(True)
    # 加载打包后位于 _MEIPASS 的 splash.png
    base = getattr(sys, '_MEIPASS', os.path.abspath('.'))
    img_path = os.path.join(base, 'splash.png')
    img = Image.open(img_path)
    tk_img = ImageTk.PhotoImage(img)
    lbl = tk.Label(root, image=tk_img, borderwidth=0)
    lbl.pack()
    root.update_idletasks()
    # 居中
    w, h = root.winfo_width(), root.winfo_height()
    ws, hs = root.winfo_screenwidth(), root.winfo_screenheight()
    x, y = (ws - w)//2, (hs - h)//2
    root.geometry(f'{w}x{h}+{x}+{y}')
    # 3 秒后销毁 Splash
    root.after(3000, root.destroy)
    root.mainloop()

# 在最开始就启动 Splash 线程
threading.Thread(target=_show_splash, daemon=True).start()

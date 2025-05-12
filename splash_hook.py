# splash_hook.py
import sys, os, tkinter as tk
from PIL import Image, ImageTk

def show_splash():
    base = getattr(sys, '_MEIPASS', os.path.abspath('.'))
    img_path = os.path.join(base, 'splash.png')
    root = tk.Tk()
    root.overrideredirect(True)
    root.geometry('+%d+%d' % (
        (root.winfo_screenwidth() - 600)//2,
        (root.winfo_screenheight() - 400)//2))
    img = Image.open(img_path)
    tk_img = ImageTk.PhotoImage(img)
    label = tk.Label(root, image=tk_img)
    label.image = tk_img
    label.pack()
    root.update()

    # Store splash window so my_startup_hook can destroy it later
    sys._splash_root = root

show_splash()

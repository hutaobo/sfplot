# -*- coding: utf-8 -*-
# PyInstaller startup hook: my_startup_hook.py

import sys
import os
import pathlib

# ── 在冻结模式下指向打包后的 Tcl/Tk 脚本 ─────────────────────────────
if getattr(sys, 'frozen', False):
    base = pathlib.Path(sys._MEIPASS)
    os.environ['TCL_LIBRARY'] = str(base / 'lib' / 'tcl8.6')
    os.environ['TK_LIBRARY']  = str(base / 'lib' / 'tk8.6')

# 1) Now import tkinter (it will look for init.tcl in the above paths)
import tkinter as tk
import traceback
import threading
import logging
from pathlib import Path

# 2) Show Splash Screen using a hidden root and a Toplevel window
base_path = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
splash_path = os.path.join(base_path, 'splash.png')

# Create a temporary root window and hide it (this will be the parent for the splash)
tcl_interp = tk.Tcl()
tcl_interp.loadtk()  # 加载 Tk 子系统
splash_win = tk.Toplevel()
splash_win.overrideredirect(True)  # Remove window decorations (border, title bar)

# Load the splash image
splash_image = None
try:
    splash_image = tk.PhotoImage(file=splash_path)
except Exception:
    splash_image = None  # If loading image fails, we'll use text instead

# Create a label to hold the image (or text as fallback) in the splash window
if splash_image:
    splash_label = tk.Label(splash_win, image=splash_image)
else:
    splash_label = tk.Label(splash_win, text="Loading...", font=("Arial", 18))
splash_label.pack()

# Center the splash window on the screen
screen_width = splash_win.winfo_screenwidth()
screen_height = splash_win.winfo_screenheight()
img_width = splash_image.width() if splash_image else splash_label.winfo_reqwidth()
img_height = splash_image.height() if splash_image else splash_label.winfo_reqheight()
pos_x = (screen_width - img_width) // 2
pos_y = (screen_height - img_height) // 2
splash_win.geometry(f"{img_width}x{img_height}+{pos_x}+{pos_y}")

# Show the splash window now (since we used overrideredirect, explicitly deiconify it)
splash_win.update_idletasks()
splash_win.deiconify()
splash_win.update()

_original_tk_init = tk.Tk.__init__
def _new_tk_init(self, *args, **kwargs):
    _original_tk_init(self, *args, **kwargs)     # init the main Tk window
    tk._default_root = self                     # <<< make main window the default root
    # Now we can safely destroy the splash and temp_root
    try:
        if splash_win is not None:
            splash_win.destroy()
    except Exception:
        pass
    try:
        tcl_interp.destroy()
    except Exception:
        pass

tk.Tk.__init__ = _new_tk_init

# 4) (Optional) Global exception and threading exception hooks for logging
log_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.getcwd()
log_file = os.path.join(log_dir, "error.log")

def log_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Log any uncaught exceptions to a file."""
    error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "="*60 + "\nUnhandled Exception:\n" + error_message + "\n")
    except Exception:
        pass
    # Optionally, call the default handler (prints to stderr)
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = log_uncaught_exception

if hasattr(threading, "excepthook"):
    _orig_thread_excepthook = threading.excepthook
    def log_thread_exception(args):
        error_message = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n" + "="*60 + f"\nUnhandled Exception in thread '{args.thread.name}':\n" + error_message + "\n")
        except Exception:
            pass
        if _orig_thread_excepthook:
            _orig_thread_excepthook(args)
    threading.excepthook = log_thread_exception

# (Hook setup is done. The main application will start execution next.)

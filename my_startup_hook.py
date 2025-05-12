# my_startup_hook.py

import sys
import os
import logging
import traceback
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------
# Global exception handler hook for PyInstaller launcher
# ------------------------------------------------------------
# Configure log directory and file
log_dir = Path(os.getenv("APPDATA", Path.home() / ".local")) / "CellGPS"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "error.log"

# Set up logger
logger = logging.getLogger("CellGPSStartup")
logger.setLevel(logging.DEBUG)

# File handler writing UTF-8 encoded logs
file_handler = logging.FileHandler(log_file, encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Optional: also log to stderr for console visibility
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Global exception hook
def global_exception_handler(exctype, value, tb):
    # Format the full traceback
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    # Log as critical, with a proper newline
    logger.critical(f"未捕获的异常:\n{error_msg}")
    # Optionally call the default excepthook to print to console
    sys.__excepthook__(exctype, value, tb)

# Install the exception hook
sys.excepthook = global_exception_handler


# ------------------------------------------------------------
# Monkey-patch tkinter.Tk to destroy the splash screen
# ------------------------------------------------------------
try:
    import tkinter as _tk

    # Save original Tk class
    _OriginalTk = _tk.Tk

    def _SplashAwareTk(*args, **kwargs):
        """
        Before creating the real main window, destroy the splash if it exists.
        """
        # Destroy splash window created by splash_hook.py
        try:
            splash = getattr(sys, "_splash_root", None)
            if splash:
                splash.destroy()
                # Remove reference so we don't try again
                del sys._splash_root
        except Exception:
            pass

        # Now create the actual main Tk window
        return _OriginalTk(*args, **kwargs)

    # Override Tk in this runtime
    _tk.Tk = _SplashAwareTk

except ImportError:
    # If tkinter isn't available, skip this step
    pass

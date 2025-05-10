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

# Install the hook
sys.excepthook = global_exception_handler

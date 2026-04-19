# -*- coding: utf-8 -*-
"""Runtime hook for the packaged CellGPS GUI."""

from __future__ import annotations

import os
import pathlib
import sys
import threading
import traceback


def _configure_tcl_tk() -> None:
    """Point Tkinter at the bundled Tcl/Tk resources when frozen."""
    if not getattr(sys, "frozen", False):
        return

    base = pathlib.Path(sys._MEIPASS)
    candidates = (
        (base / "tcl" / "tcl8.6", base / "tcl" / "tk8.6"),
        (base / "lib" / "tcl8.6", base / "lib" / "tk8.6"),
    )
    for tcl_dir, tk_dir in candidates:
        if tcl_dir.is_dir() and tk_dir.is_dir():
            os.environ.setdefault("TCL_LIBRARY", str(tcl_dir))
            os.environ.setdefault("TK_LIBRARY", str(tk_dir))
            return


def _install_exception_logging() -> None:
    log_dir = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
    log_file = os.path.join(log_dir, "error.log")

    def log_uncaught_exception(exc_type, exc_value, exc_traceback):
        error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        try:
            with open(log_file, "a", encoding="utf-8") as handle:
                handle.write("\n" + "=" * 60 + "\nUnhandled Exception:\n" + error_message + "\n")
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = log_uncaught_exception

    if hasattr(threading, "excepthook"):
        original_thread_excepthook = threading.excepthook

        def log_thread_exception(args):
            error_message = "".join(
                traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
            )
            try:
                with open(log_file, "a", encoding="utf-8") as handle:
                    handle.write(
                        "\n"
                        + "=" * 60
                        + f"\nUnhandled Exception in thread '{args.thread.name}':\n"
                        + error_message
                        + "\n"
                    )
            except Exception:
                pass
            if original_thread_excepthook:
                original_thread_excepthook(args)

        threading.excepthook = log_thread_exception


_configure_tcl_tk()
_install_exception_logging()

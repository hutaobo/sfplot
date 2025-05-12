from __future__ import annotations
import queue
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import END, filedialog, messagebox, ttk

import pandas as pd
from PIL import Image, ImageTk

# SFPlot imports
from sfplot import (
    compute_cophenetic_distances_from_df,
    plot_cophenetic_heatmap,
    load_xenium_data,
    compute_cophenetic_distances_from_adata,
)


def resource_path(relative_path: str) -> str:
    """
    Helper to get absolute path, if needed (e.g., when bundling with PyInstaller).
    """
    try:
        base_path = sys._MEIPASS  # type: ignore
    except Exception:
        base_path = Path(__file__).parent
    return str(Path(base_path) / relative_path)


class MainApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CellGPS GUI")
        self.geometry("1000x750")

        # Shared queue for thread communication
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._STEPS = {
            "start": 5,
            "csv_read": 20,
            "calc_dist": 60,
            "plot": 90,
            "done": 100,
        }

        # Notebook (tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        # --- Tab 1: CSV Heatmap ---
        self.tab_csv = tk.Frame(self.notebook)
        self.notebook.add(self.tab_csv, text="CSV Heatmap")
        self._build_csv_tab()

        # --- Tab 2: Xenium Heatmap ---
        self.tab_xenium = tk.Frame(self.notebook)
        self.notebook.add(self.tab_xenium, text="Xenium Heatmap")
        self._build_xenium_tab()

        # Start polling queue
        self.after(100, self._poll_queue)
        self._log_csv("Program started successfully")

    def _build_csv_tab(self) -> None:
        # Top controls
        top = tk.Frame(self.tab_csv)
        top.pack(fill="x", padx=5, pady=5)
        self.load_btn = tk.Button(top, text="Select CSV File", command=self._ask_csv_file)
        self.load_btn.pack(side="left")
        self.draw_btn = tk.Button(top, text="Plot CSV Heatmap", state="disabled", command=self._start_csv_worker)
        self.draw_btn.pack(side="left", padx=5)
        # Zoom control
        self.scale_var = tk.DoubleVar(value=1.0)
        zoom_menu = ttk.OptionMenu(top, self.scale_var, 1.0, 0.25, 0.5, 1.0, 2.0, 4.0, command=self._on_scale_change_csv)
        tk.Label(top, text="Zoom:").pack(side="left", padx=(20,0))
        zoom_menu.pack(side="left")
        # Progress
        self._progress = ttk.Progressbar(top, mode="determinate", length=200, maximum=100)
        self._progress.pack(side="left", padx=10)
        self._progress_label = tk.Label(top, text="0%")
        self._progress_label.pack(side="left")

        # Display area
        self.display_frame = tk.LabelFrame(self.tab_csv, text="Display Area")
        self.display_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # Log area
        log_frame = tk.LabelFrame(self.tab_csv, text="Running Log")
        log_frame.pack(side="left", fill="y", padx=5, pady=5)
        self.log_text = tk.Text(log_frame, width=30, state="disabled")
        self.log_text.pack(fill="both", expand=True)

        # State
        self.csv_path: str | None = None
        self._orig_img: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._image_frame: tk.Frame | None = None

    def _build_xenium_tab(self) -> None:
        # Top controls
        top2 = tk.Frame(self.tab_xenium)
        top2.pack(fill="x", padx=5, pady=5)
        self.xenium_btn = tk.Button(top2, text="Select Xenium Dir", command=self._ask_xenium_dir)
        self.xenium_btn.pack(side="left")
        self.selcsv_btn = tk.Button(top2, text="Select Selection CSV", command=self._ask_selection_csv)
        self.selcsv_btn.pack(side="left", padx=5)
        self.plot_x_btn = tk.Button(top2, text="Plot Xenium Heatmap", state="disabled", command=self._start_xenium_worker)
        self.plot_x_btn.pack(side="left", padx=5)
        # Zoom control
        self.scale_var2 = tk.DoubleVar(value=1.0)
        zoom_menu2 = ttk.OptionMenu(top2, self.scale_var2, 1.0, 0.25, 0.5, 1.0, 2.0, 4.0, command=self._on_scale_change_x)
        tk.Label(top2, text="Zoom:").pack(side="left", padx=(20,0))
        zoom_menu2.pack(side="left")
        # Progress
        self._progress2 = ttk.Progressbar(top2, mode="determinate", length=200, maximum=100)
        self._progress2.pack(side="left", padx=10)
        self._prog_label2 = tk.Label(top2, text="0%")
        self._prog_label2.pack(side="left")

        # Display area
        self.display_frame2 = tk.LabelFrame(self.tab_xenium, text="Display Area")
        self.display_frame2.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # Log area
        log_frame2 = tk.LabelFrame(self.tab_xenium, text="Running Log")
        log_frame2.pack(side="left", fill="y", padx=5, pady=5)
        self.log_text2 = tk.Text(log_frame2, width=30, state="disabled")
        self.log_text2.pack(fill="both", expand=True)

        # State
        self.xenium_path: str | None = None
        self.selection_csv: str | None = None
        self._orig_img2: Image.Image | None = None
        self._photo2: ImageTk.PhotoImage | None = None
        self._image_frame2: tk.Frame | None = None

    # -------- CSV Tab Callbacks --------
    def _ask_csv_file(self) -> None:
        path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        self.csv_path = path
        self._log_csv(f"CSV file: {Path(path).name}")
        self.draw_btn.configure(state="normal")

    def _start_csv_worker(self) -> None:
        self.load_btn.configure(state="disabled")
        self.draw_btn.configure(state="disabled")
        threading.Thread(target=self._csv_worker, daemon=True).start()

    def _csv_worker(self) -> None:
        try:
            self._queue.put(("progress", self._STEPS["start"]))
            self._queue.put(("log", "Reading CSV file…"))
            df = pd.read_csv(self.csv_path)

            self._queue.put(("progress", self._STEPS["calc_dist"]))
            self._queue.put(("log", "Computing cophenetic distances…"))
            r, c = compute_cophenetic_distances_from_df(df)

            self._queue.put(("progress", self._STEPS["plot"]))
            self._queue.put(("log", "Plotting heatmap…"))
            img = plot_cophenetic_heatmap(r, matrix_name="row_coph", sample="", return_image=True, dpi=300)

            self._queue.put(("image", img))
            self._queue.put(("done", None))
        except Exception:
            tb = traceback.format_exc()
            self._queue.put(("error", tb))

    def _on_scale_change_csv(self, _=None) -> None:
        if self._orig_img:
            self._display_csv_image(self._orig_img)

    # -------- Xenium Tab Callbacks --------
    def _ask_xenium_dir(self) -> None:
        d = filedialog.askdirectory(title="Select Xenium Data Directory")
        if not d:
            return
        self.xenium_path = d
        self._log_x(f"Xenium dir: {d}")
        self._enable_plot_x()

    def _ask_selection_csv(self) -> None:
        f = filedialog.askopenfilename(title="Select Selection CSV", filetypes=[("CSV", "*.csv")])
        if not f:
            return
        self.selection_csv = f
        self._log_x(f"Selection CSV: {Path(f).name}")
        self._enable_plot_x()

    def _enable_plot_x(self) -> None:
        if self.xenium_path and self.selection_csv:
            self.plot_x_btn.configure(state="normal")

    def _start_xenium_worker(self) -> None:
        self.xenium_btn.configure(state="disabled")
        self.selcsv_btn.configure(state="disabled")

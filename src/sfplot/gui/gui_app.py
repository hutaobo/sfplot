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
        self._photo: ImageTk.PhotoImage | None = None
        self._scroll_canvas: tk.Canvas | None = None
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
        self._photo2: ImageTk.PhotoImage | None = None
        self._scroll_canvas2: tk.Canvas | None = None
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
            df = pd.read_csv(self.csv_path)  # assume first two cols as matrix

            self._queue.put(("progress", self._STEPS["calc_dist"]))
            self._queue.put(("log", "Computing cophenetic distances…"))
            # expects df with numeric data
            r, c = compute_cophenetic_distances_from_df(df)

            self._queue.put(("progress", self._STEPS["plot"]))
            self._queue.put(("log", "Plotting heatmap…"))
            img = plot_cophenetic_heatmap(r, matrix_name="row_coph", sample="", return_image=True, dpi=300)

            self._queue.put(("image", img))
            self._queue.put(("done", None))
        except Exception:
            tb = traceback.format_exc()
            self._queue.put(("error", tb))

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
        self.plot_x_btn.configure(state="disabled")
        threading.Thread(target=self._xenium_worker, daemon=True).start()

    def _xenium_worker(self) -> None:
        try:
            self._queue.put(("x_progress", self._STEPS["start"]))
            self._queue.put(("x_log", "Loading Xenium data…"))
            adata = load_xenium_data(self.xenium_path, normalize=False)

            self._queue.put(("x_progress", self._STEPS["csv_read"]))
            self._queue.put(("x_log", "Reading selection CSV…"))
            df = pd.read_csv(self.selection_csv, comment="#", header=0)
            cell_ids = df["Cell ID"].tolist()
            sub = adata[adata.obs["cell_id"].isin(cell_ids)].copy()

            self._queue.put(("x_progress", self._STEPS["calc_dist"]))
            self._queue.put(("x_log", "Computing cophenetic distances…"))
            r, c = compute_cophenetic_distances_from_adata(sub)

            self._queue.put(("x_progress", self._STEPS["plot"]))
            self._queue.put(("x_log", "Plotting heatmap…"))
            img = plot_cophenetic_heatmap(
                r, matrix_name="row_coph", sample="", return_image=True, dpi=300
            )

            self._queue.put(("x_image", img))
            self._queue.put(("done", None))
        except Exception:
            tb = traceback.format_exc()
            self._queue.put(("x_error", tb))

    # -------- Queue Polling & Embedding --------
    def _poll_queue(self) -> None:
        try:
            while True:
                tag, payload = self._queue.get_nowait()
                # CSV tab
                if tag == "log":
                    self._log_csv(payload)
                elif tag == "progress":
                    val = int(payload)
                    self._progress.configure(value=val)
                    self._progress_label.configure(text=f"{val}%")
                elif tag == "image":
                    self._embed_image(payload)
                elif tag == "error":
                    messagebox.showerror("Error", payload)
                elif tag == "done":
                    self.load_btn.configure(state="normal")
                    self.draw_btn.configure(state="normal")
                # Xenium tab
                elif tag == "x_log":
                    self._log_x(payload)
                elif tag == "x_progress":
                    v = int(payload)
                    self._progress2.configure(value=v)
                    self._prog_label2.configure(text=f"{v}%")
                elif tag == "x_image":
                    self._embed_image2(payload)
                elif tag == "x_error":
                    messagebox.showerror("Error", payload)
                elif tag == "done":
                    self.xenium_btn.configure(state="normal")
                    self.selcsv_btn.configure(state="normal")
                    self.plot_x_btn.configure(state="normal")
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    def _log_csv(self, msg: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(END, msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _log_x(self, msg: str) -> None:
        self.log_text2.configure(state="normal")
        self.log_text2.insert(END, msg + "\n")
        self.log_text2.see("end")
        self.log_text2.configure(state="disabled")

    def _embed_image(self, pil_img: Image.Image) -> None:
        if self._image_frame:
            self._image_frame.destroy()
        self._image_frame = tk.Frame(self.display_frame)
        self._image_frame.pack(fill="both", expand=True)
        canvas = tk.Canvas(self._image_frame)
        hbar = ttk.Scrollbar(self._image_frame, orient="horizontal", command=canvas.xview)
        vbar = ttk.Scrollbar(self._image_frame, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")
        hbar.pack(side="bottom", fill="x")
        self._photo = ImageTk.PhotoImage(pil_img)
        canvas.create_image(0, 0, image=self._photo, anchor="nw")
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

    def _embed_image2(self, pil_img: Image.Image) -> None:
        if self._image_frame2:
            self._image_frame2.destroy()
        self._image_frame2 = tk.Frame(self.display_frame2)
        self._image_frame2.pack(fill="both", expand=True)
        canvas = tk.Canvas(self._image_frame2)
        hbar = ttk.Scrollbar(self._image_frame2, orient="horizontal", command=canvas.xview)
        vbar = ttk.Scrollbar(self._image_frame2, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")
        hbar.pack(side="bottom", fill="x")
        self._photo2 = ImageTk.PhotoImage(pil_img)
        canvas.create_image(0, 0, image=self._photo2, anchor="nw")
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))


def main() -> None:
    app = MainApp()
    app.mainloop()


if __name__ == "__main__":
    main()

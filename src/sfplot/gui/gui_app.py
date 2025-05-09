from __future__ import annotations
import queue
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import END, filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

from sfplot import compute_cophenetic_distances_from_df, plot_cophenetic_heatmap

# Log file
_EXEC_DIR = Path(sys.argv[0]).resolve().parent
_LOG_FILE = _EXEC_DIR / "cellgps_error.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def _write_error(msg: str) -> None:
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n---\n")

class MainApp(tk.Tk):
    _STEPS = {"start": 0, "csv_read": 10, "calc_dist": 50, "plot": 90, "done": 100}

    def __init__(self) -> None:
        super().__init__()
        self.title("SFPlot Cophenetic Heatmap")
        self.geometry("900x700")
        self.report_callback_exception = self._handle_exception
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()

        # Top toolbar
        top = tk.Frame(self)
        top.pack(fill="x", padx=5, pady=5)

        # Load button
        self.load_btn = tk.Button(top, text="Select CSV", command=self._ask_file)
        self.load_btn.pack(side="left")

        # Column selection comboboxes
        self.x_col_cb = ttk.Combobox(top, state="disabled")
        self.x_col_cb.pack(side="left", padx=5)
        self.y_col_cb = ttk.Combobox(top, state="disabled")
        self.y_col_cb.pack(side="left", padx=5)
        self.cell_col_cb = ttk.Combobox(top, state="disabled")
        self.cell_col_cb.pack(side="left", padx=5)

        # Plot button
        self.draw_btn = tk.Button(top, text="Plot Heatmap", state="disabled", command=self._start_worker)
        self.draw_btn.pack(side="left", padx=5)

        # Progress bar
        self._progress = ttk.Progressbar(top, mode="determinate", length=200, maximum=100)
        self._progress.pack(side="left", padx=10)
        self._progress_label = tk.Label(top, text="0%")
        self._progress_label.pack(side="left")

        # Log area
        log_frame = tk.LabelFrame(self, text="Log")
        log_frame.pack(side="left", fill="y", padx=5, pady=5)
        self.log_text = tk.Text(log_frame, width=30, state="disabled")
        self.log_text.pack(fill="both", expand=True)

        # Plot area
        self.display_frame = tk.LabelFrame(self, text="Plot Area")
        self.display_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        self._figure_canvas = None
        self.df: pd.DataFrame | None = None

        self.after(100, self._poll_queue)
        self._log("Application started successfully")

    def _log(self, msg: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(END, msg + "\n")
        self.log_text.see(END)
        self.log_text.configure(state="disabled")
        print(msg)

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        err = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        _write_error(err)
        self._log(f"Error: {exc_value}")
        messagebox.showerror("Application Error", f"An error occurred:\n{exc_value}\n\nDetails logged to: {_LOG_FILE}")

    def _ask_file(self) -> None:
        self._log(">> Opening file…")
        path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV", "*.csv")])
        if not path:
            self._log(">> Cancelled.")
            return
        self._log(f"Read CSV: {Path(path).name}")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Read Error", f"Failed to read file: {e}")
            return
        self.df = df
        cols = list(df.columns)
        # Update combobox options
        for cb in (self.x_col_cb, self.y_col_cb, self.cell_col_cb):
            cb.configure(values=cols, state="readonly")
            cb.set("")
        self.draw_btn.configure(state="normal")
        self._log("Columns loaded. Please select x, y and cell type columns.")

    def _start_worker(self) -> None:
        x_col = self.x_col_cb.get()
        y_col = self.y_col_cb.get()
        cell_col = self.cell_col_cb.get()
        if not all((x_col, y_col, cell_col)):
            messagebox.showwarning("Selection Missing", "Please select columns for x, y, and cell type.")
            return
        self.load_btn.configure(state="disabled")
        self.draw_btn.configure(state="disabled")
        threading.Thread(target=self._worker, args=(self.df, x_col, y_col, cell_col), daemon=True).start()

    def _worker(self, df: pd.DataFrame, x_col: str, y_col: str, celltype_col: str) -> None:
        try:
            self._queue.put(("progress", self._STEPS["start"]))
            self._queue.put(("log", f"Computing distances: x={x_col}, y={y_col}, cell={celltype_col}"))
            row_df, _ = compute_cophenetic_distances_from_df(
                df, x_col=x_col, y_col=y_col, celltype_col=celltype_col, output_dir=None
            )
            self._queue.put(("progress", self._STEPS["calc_dist"]))

            self._queue.put(("log", "Generating heatmap…"))
            cluster_grid = plot_cophenetic_heatmap(
                row_df,
                matrix_name="row_coph",
                sample="",
                return_figure=True,
            )

            self._queue.put(("progress", self._STEPS["plot"]))
            self._queue.put(("log", "Heatmap generation complete"))
            self._queue.put(("figure", cluster_grid))
            self._queue.put(("progress", self._STEPS["done"]))

        except Exception as e:
            err_msg = f"Background error: {e}\n{traceback.format_exc()}"
            _write_error(err_msg)
            self._queue.put(("error", str(e)))
        finally:
            self._queue.put(("done", None))

    def _poll_queue(self) -> None:
        try:
            while True:
                tag, payload = self._queue.get_nowait()
                if tag == "log":
                    self._log(payload)
                elif tag == "progress":
                    val = int(payload)
                    self._progress.configure(value=val)
                    self._progress_label.configure(text=f"{val}%")
                elif tag == "figure":
                    self._embed_figure(payload)
                elif tag == "error":
                    messagebox.showerror("Error", payload)
                elif tag == "done":
                    self.load_btn.configure(state="normal")
                    self.draw_btn.configure(state="normal")
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    def _embed_figure(self, cluster_grid):
        if self._figure_canvas is not None:
            self._figure_canvas.get_tk_widget().destroy()
        self._figure_canvas = FigureCanvasTkAgg(cluster_grid.fig, master=self.display_frame)
        self._figure_canvas.draw()
        self._figure_canvas.get_tk_widget().pack(fill="both", expand=True)


def main() -> None:
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    main()

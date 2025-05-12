# src/sfplot/gui/gui_app.py

"""SFPlot Cophenetic Heatmap GUI — supports high-resolution images and mouse wheel/scrollbars"""

from __future__ import annotations
import queue
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import END, filedialog, messagebox, ttk

import matplotlib

# use the TkAgg backend
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

from sfplot import compute_cophenetic_distances_from_df, plot_cophenetic_heatmap  # noqa: E402

# log file
_EXEC_DIR = Path(sys.argv[0]).resolve().parent
_LOG_FILE = _EXEC_DIR / "cellgps_error.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def _write_error(msg: str) -> None:
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n---\n")


class MainApp(tk.Tk):
    """Main window: background thread + scrollable Canvas + mouse wheel support"""
    _STEPS = {"start": 0, "csv_read": 10, "calc_dist": 50, "plot": 90, "done": 100}

    def __init__(self) -> None:
        super().__init__()
        self.title("CellGPS")
        self.geometry("900x700")
        self.report_callback_exception = self._handle_exception
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()

        # top toolbar
        top = tk.Frame(self)
        top.pack(fill="x", padx=5, pady=5)

        # Select CSV button
        self.load_btn = tk.Button(top, text="Select CSV", command=self._ask_file)
        self.load_btn.pack(side="left")

        # X: label and combobox
        self.x_label = tk.Label(top, text="X:")
        self.x_label.pack(side="left", padx=(10, 0))
        self.x_col_cb = ttk.Combobox(top, state="disabled")
        self.x_col_cb.pack(side="left", padx=5)

        # Y: label and combobox
        self.y_label = tk.Label(top, text="Y:")
        self.y_label.pack(side="left", padx=(10, 0))
        self.y_col_cb = ttk.Combobox(top, state="disabled")
        self.y_col_cb.pack(side="left", padx=5)

        # Cell Type: label and combobox
        self.cell_label = tk.Label(top, text="Cell Type:")
        self.cell_label.pack(side="left", padx=(10, 0))
        self.cell_col_cb = ttk.Combobox(top, state="disabled")
        self.cell_col_cb.pack(side="left", padx=5)

        # Plot Heatmap button
        self.draw_btn = tk.Button(top, text="Plot Heatmap", state="disabled", command=self._start_worker)
        self.draw_btn.pack(side="left", padx=5)

        # Progress bar
        self._progress = ttk.Progressbar(top, mode="determinate", length=200, maximum=100)
        self._progress.pack(side="left", padx=10)
        self._progress_label = tk.Label(top, text="0%")
        self._progress_label.pack(side="left")

        # Running Log
        log_frame = tk.LabelFrame(self, text="Running Log")
        log_frame.pack(side="left", fill="y", padx=5, pady=5)
        self.log_text = tk.Text(log_frame, width=30, state="disabled")
        self.log_text.pack(fill="both", expand=True)

        # Display Area
        self.display_frame = tk.LabelFrame(self, text="Display Area")
        self.display_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # image references
        self._photo = None  # type: ImageTk.PhotoImage | None
        self._scroll_canvas = None  # type: tk.Canvas | None
        self._image_frame = None  # type: tk.Frame | None
        self.df: pd.DataFrame | None = None

        # start polling
        self.after(100, self._poll_queue)
        self._log("Program started successfully")

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
        messagebox.showerror("Error", f"An error occurred:\n{exc_value}\n\nLogged to: {_LOG_FILE}")

    def _ask_file(self) -> None:
        self._log(">> Opening file…")
        path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV", "*.csv")])
        if not path:
            self._log(">> Cancelled.")
            return
        self._log(f"Reading CSV: {Path(path).name}")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Read Error", f"Failed to read file: {e}")
            return
        self.df = df
        cols = list(df.columns)
        for cb in (self.x_col_cb, self.y_col_cb, self.cell_col_cb):
            cb.configure(values=cols, state="readonly")
            cb.set("")
        # defaults
        if "xc" in cols:
            self.x_col_cb.set("xc")
        elif "x" in cols:
            self.x_col_cb.set("x")
        if "yc" in cols:
            self.y_col_cb.set("yc")
        elif "y" in cols:
            self.y_col_cb.set("y")
        if "target" in cols:
            self.cell_col_cb.set("target")
        elif "celltype" in cols:
            self.cell_col_cb.set("celltype")
        self.draw_btn.configure(state="normal")
        self._log("Columns loaded. Please select X, Y, and Cell Type.")

    def _start_worker(self) -> None:
        x_col = self.x_col_cb.get()
        y_col = self.y_col_cb.get()
        cell_col = self.cell_col_cb.get()
        if not all((x_col, y_col, cell_col)):
            messagebox.showwarning("Selection Missing", "Please select X, Y, and Cell Type columns.")
            return
        self.load_btn.configure(state="disabled")
        self.draw_btn.configure(state="disabled")
        threading.Thread(target=self._worker, args=(self.df, x_col, y_col, cell_col), daemon=True).start()

    def _worker(self, df: pd.DataFrame, x_col: str, y_col: str, celltype_col: str) -> None:
        try:
            self._queue.put(("progress", self._STEPS["start"]))
            self._queue.put(("log", f"Computing distance matrix: x={x_col}, y={y_col}, cell={celltype_col}"))
            row_df, _ = compute_cophenetic_distances_from_df(
                df, x_col=x_col, y_col=y_col, celltype_col=celltype_col, output_dir=None
            )
            self._queue.put(("progress", self._STEPS["calc_dist"]))

            self._queue.put(("log", "Generating high-res heatmap…"))
            image = plot_cophenetic_heatmap(
                row_df,
                matrix_name="row_coph",
                sample="",
                return_image=True,
                dpi=300,
            )

            self._queue.put(("progress", self._STEPS["plot"]))
            self._queue.put(("log", "Heatmap generation complete"))
            self._queue.put(("image", image))
            self._queue.put(("progress", self._STEPS["done"]))

        except Exception as e:
            err_msg = f"Worker error: {e}\n{traceback.format_exc()}"
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
                elif tag == "image":
                    self._embed_image(payload)
                elif tag == "error":
                    messagebox.showerror("Error", payload)
                elif tag == "done":
                    self.load_btn.configure(state="normal")
                    self.draw_btn.configure(state="normal")
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    def _embed_image(self, pil_img: Image.Image) -> None:
        """Display PIL image"""
        if self._image_frame is not None:
            self._image_frame.destroy()
            self._image_frame = None

        self._image_frame = tk.Frame(self.display_frame)
        self._image_frame.pack(fill="both", expand=True)

        self._scroll_canvas = tk.Canvas(self._image_frame)
        vscroll = ttk.Scrollbar(self._image_frame, orient="vertical", command=self._scroll_canvas.yview)
        hscroll = ttk.Scrollbar(self._image_frame, orient="horizontal", command=self._scroll_canvas.xview)

        self._scroll_canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)

        vscroll.pack(side="right", fill="y")
        hscroll.pack(side="bottom", fill="x")
        self._scroll_canvas.pack(side="left", fill="both", expand=True)

        self._photo = ImageTk.PhotoImage(pil_img)
        inner_frame = tk.Frame(self._scroll_canvas)
        self._scroll_canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        img_lbl = tk.Label(inner_frame, image=self._photo)
        img_lbl.pack()

        def _update_scrollregion(event):
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))
        inner_frame.bind("<Configure>", _update_scrollregion)

        def _on_mousewheel(event):
            self._scroll_canvas.yview_scroll(-1 * (event.delta // 120), "units")
        def _on_shift_mousewheel(event):
            self._scroll_canvas.xview_scroll(-1 * (event.delta // 120), "units")

        self._scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self._scroll_canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)
        self._scroll_canvas.bind_all("<Button-4>", lambda e: self._scroll_canvas.yview_scroll(-1, "units"))
        self._scroll_canvas.bind_all("<Button-5>", lambda e: self._scroll_canvas.yview_scroll(1, "units"))

        self._scroll_canvas.update_idletasks()
        self._scroll_canvas.yview_moveto(0)
        self._scroll_canvas.xview_moveto(0)

        self._log("Image loaded — use mouse wheel or scrollbars to navigate")

def main() -> None:
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    main()

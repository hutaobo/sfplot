from __future__ import annotations

import queue
import threading
import traceback
from importlib import import_module
from pathlib import Path
from typing import Any, Callable
import tkinter as tk
from tkinter import END, filedialog, messagebox, ttk

import pandas as pd
from PIL import Image, ImageTk

SearcherApi = tuple[
    Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
    Callable[[Any], tuple[pd.DataFrame, pd.DataFrame]],
    Callable[..., Image.Image],
]


def main() -> None:
    app = MainApp()
    app.mainloop()


class MainApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CellGPS GUI")
        self.geometry("1000x750")

        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._searcher_api: SearcherApi | None = None
        self._xenium_loader: Callable[[str, bool], Any] | None = None
        self._steps = {
            "start": 5,
            "csv_read": 20,
            "calc_dist": 60,
            "plot": 90,
            "done": 100,
        }

        self.adata_cache: Any | None = None
        self.csv_path: str | None = None
        self.xenium_path: str | None = None
        self.selection_csv: str | None = None
        self._orig_img: Image.Image | None = None
        self._orig_img2: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._photo2: ImageTk.PhotoImage | None = None
        self._image_frame: tk.Frame | None = None
        self._image_frame2: tk.Frame | None = None

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.tab_csv = tk.Frame(self.notebook)
        self.notebook.add(self.tab_csv, text="CSV Heatmap")
        self._build_csv_tab()

        self.tab_xenium = tk.Frame(self.notebook)
        self.notebook.add(self.tab_xenium, text="Xenium Heatmap")
        self._build_xenium_tab()

        self.after(100, self._poll_queue)
        self._log_csv("Program started successfully")

    def _get_searcher_api(self) -> SearcherApi:
        if self._searcher_api is None:
            module = import_module("sfplot.Searcher_Findee_Score")
            self._searcher_api = (
                module.compute_cophenetic_distances_from_df,
                module.compute_cophenetic_distances_from_adata,
                module.plot_cophenetic_heatmap,
            )
        return self._searcher_api

    def _get_xenium_loader(self) -> Callable[[str, bool], Any]:
        if self._xenium_loader is None:
            module = import_module("sfplot.data_processing")
            self._xenium_loader = module.load_xenium_data
        return self._xenium_loader

    def _build_csv_tab(self) -> None:
        top = tk.Frame(self.tab_csv)
        top.pack(fill="x", padx=5, pady=5)
        self.load_btn = tk.Button(top, text="Select CSV File", command=self._ask_csv_file)
        self.load_btn.pack(side="left")
        self.draw_btn = tk.Button(
            top,
            text="Plot CSV Heatmap",
            state="disabled",
            command=self._start_csv_worker,
        )
        self.draw_btn.pack(side="left", padx=5)

        self.scale_var = tk.DoubleVar(value=1.0)
        tk.Label(top, text="Zoom:").pack(side="left", padx=(20, 0))
        ttk.OptionMenu(
            top,
            self.scale_var,
            1.0,
            0.25,
            0.5,
            1.0,
            2.0,
            4.0,
            command=self._on_scale_change_csv,
        ).pack(side="left")

        self._progress = ttk.Progressbar(top, mode="determinate", length=200, maximum=100)
        self._progress.pack(side="left", padx=10)
        self._progress_label = tk.Label(top, text="0%")
        self._progress_label.pack(side="left")

        self.display_frame = tk.LabelFrame(self.tab_csv, text="Display Area")
        self.display_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        log_frame = tk.LabelFrame(self.tab_csv, text="Running Log")
        log_frame.pack(side="left", fill="y", padx=5, pady=5)
        self.log_text = tk.Text(log_frame, width=30, state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def _build_xenium_tab(self) -> None:
        top = tk.Frame(self.tab_xenium)
        top.pack(fill="x", padx=5, pady=5)

        self.xenium_btn = tk.Button(top, text="Select Xenium Dir", command=self._ask_xenium_dir)
        self.xenium_btn.pack(side="left")

        self.load_xenium_btn = tk.Button(
            top,
            text="Load Xenium Data",
            state="disabled",
            command=self._load_xenium_data,
        )
        self.load_xenium_btn.pack(side="left", padx=5)

        self.selcsv_btn = tk.Button(
            top,
            text="Select Selection CSV",
            state="disabled",
            command=self._ask_selection_csv,
        )
        self.selcsv_btn.pack(side="left", padx=5)

        self.plot_x_btn = tk.Button(
            top,
            text="Plot Xenium Heatmap",
            state="disabled",
            command=self._start_xenium_plot,
        )
        self.plot_x_btn.pack(side="left", padx=5)

        self.scale_var2 = tk.DoubleVar(value=1.0)
        tk.Label(top, text="Zoom:").pack(side="left", padx=(20, 0))
        ttk.OptionMenu(
            top,
            self.scale_var2,
            1.0,
            0.25,
            0.5,
            1.0,
            2.0,
            4.0,
            command=self._on_scale_change_x,
        ).pack(side="left")

        self._progress2 = ttk.Progressbar(top, mode="determinate", length=200, maximum=100)
        self._progress2.pack(side="left", padx=10)
        self._prog_label2 = tk.Label(top, text="0%")
        self._prog_label2.pack(side="left")

        self.display_frame2 = tk.LabelFrame(self.tab_xenium, text="Display Area")
        self.display_frame2.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        log_frame = tk.LabelFrame(self.tab_xenium, text="Running Log")
        log_frame.pack(side="left", fill="y", padx=5, pady=5)
        self.log_text2 = tk.Text(log_frame, width=30, state="disabled")
        self.log_text2.pack(fill="both", expand=True)

    def _ask_csv_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
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
            compute_from_df, _, plot_heatmap = self._get_searcher_api()
            self._queue.put(("csv_progress", self._steps["start"]))
            self._queue.put(("csv_log", "Reading CSV file..."))
            df = pd.read_csv(self.csv_path)
            self._queue.put(("csv_progress", self._steps["calc_dist"]))
            self._queue.put(("csv_log", "Computing distances..."))
            row_cophenetic, _ = compute_from_df(df)
            self._queue.put(("csv_progress", self._steps["plot"]))
            self._queue.put(("csv_log", "Plotting heatmap..."))
            img = plot_heatmap(
                row_cophenetic,
                matrix_name="row_coph",
                sample="",
                return_image=True,
                dpi=300,
            )
            self._queue.put(("csv_image", img))
            self._queue.put(("csv_done", None))
        except Exception:
            self._queue.put(("csv_error", traceback.format_exc()))

    def _on_scale_change_csv(self, _=None) -> None:
        if self._orig_img:
            self._display_csv_image(self._orig_img)

    def _ask_xenium_dir(self) -> None:
        path = filedialog.askdirectory(title="Select Xenium Data Directory")
        if not path:
            return
        self.xenium_path = path
        self._log_x(f"Xenium dir selected: {path}")
        self.load_xenium_btn.configure(state="normal")

    def _load_xenium_data(self) -> None:
        self.xenium_btn.configure(state="disabled")
        self.load_xenium_btn.configure(state="disabled")
        threading.Thread(target=self._xenium_load_worker, daemon=True).start()

    def _xenium_load_worker(self) -> None:
        try:
            load_xenium_data = self._get_xenium_loader()
            self._queue.put(("x_log", "Loading Xenium data..."))
            self.adata_cache = load_xenium_data(self.xenium_path, normalize=False)
            self._queue.put(("x_log", "Xenium data loaded."))
            self._queue.put(("x_load_done", None))
        except Exception:
            self._queue.put(("x_load_error", traceback.format_exc()))

    def _ask_selection_csv(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        self.selection_csv = path
        self._log_x(f"Selection CSV: {Path(path).name}")
        self.plot_x_btn.configure(state="normal")

    def _start_xenium_plot(self) -> None:
        self.selcsv_btn.configure(state="disabled")
        self.plot_x_btn.configure(state="disabled")
        threading.Thread(target=self._xenium_plot_worker, daemon=True).start()

    def _xenium_plot_worker(self) -> None:
        try:
            _, compute_from_adata, plot_heatmap = self._get_searcher_api()
            self._queue.put(("x_progress", self._steps["start"]))
            self._queue.put(("x_log", "Reading selection CSV..."))
            df = pd.read_csv(self.selection_csv, comment="#")
            cell_ids = df["Cell ID"].tolist()
            self._queue.put(("x_progress", self._steps["calc_dist"]))
            self._queue.put(("x_log", "Computing distances..."))
            sub = self.adata_cache[self.adata_cache.obs["cell_id"].isin(cell_ids)].copy()
            row_cophenetic, _ = compute_from_adata(sub)
            self._queue.put(("x_progress", self._steps["plot"]))
            self._queue.put(("x_log", "Plotting heatmap..."))
            img = plot_heatmap(
                row_cophenetic,
                matrix_name="row_coph",
                sample="",
                return_image=True,
                dpi=300,
            )
            self._queue.put(("x_image", img))
            self._queue.put(("x_plot_done", None))
        except Exception:
            self._queue.put(("x_plot_error", traceback.format_exc()))

    def _on_scale_change_x(self, _=None) -> None:
        if self._orig_img2:
            self._display_x_image(self._orig_img2)

    def _poll_queue(self) -> None:
        try:
            while True:
                tag, payload = self._queue.get_nowait()
                if tag == "csv_log":
                    self._log_csv(str(payload))
                elif tag == "csv_progress":
                    value = int(payload)
                    self._progress.configure(value=value)
                    self._progress_label.configure(text=f"{value}%")
                elif tag == "csv_image":
                    self._display_csv_image(payload)
                elif tag == "csv_done":
                    self.load_btn.configure(state="normal")
                    self.draw_btn.configure(state="normal")
                    self._progress.configure(value=self._steps["done"])
                    self._progress_label.configure(text=f"{self._steps['done']}%")
                elif tag == "csv_error":
                    self.load_btn.configure(state="normal")
                    self.draw_btn.configure(state="normal")
                    messagebox.showerror("Error", payload)
                elif tag == "x_log":
                    self._log_x(str(payload))
                elif tag == "x_progress":
                    value = int(payload)
                    self._progress2.configure(value=value)
                    self._prog_label2.configure(text=f"{value}%")
                elif tag == "x_image":
                    self._display_x_image(payload)
                elif tag == "x_load_done":
                    self.xenium_btn.configure(state="normal")
                    self.load_xenium_btn.configure(state="normal")
                    self.selcsv_btn.configure(state="normal")
                elif tag == "x_plot_done":
                    self.selcsv_btn.configure(state="normal")
                    self.plot_x_btn.configure(state="normal")
                    self.xenium_btn.configure(state="normal")
                    self.load_xenium_btn.configure(state="normal")
                    self._progress2.configure(value=self._steps["done"])
                    self._prog_label2.configure(text=f"{self._steps['done']}%")
                elif tag == "x_load_error":
                    self.xenium_btn.configure(state="normal")
                    self.load_xenium_btn.configure(state="normal")
                    messagebox.showerror("Error", payload)
                elif tag == "x_plot_error":
                    self.selcsv_btn.configure(state="normal")
                    self.plot_x_btn.configure(state="normal")
                    self.xenium_btn.configure(state="normal")
                    self.load_xenium_btn.configure(state="normal")
                    messagebox.showerror("Error", payload)
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

    def _display_csv_image(self, pil_img: Image.Image) -> None:
        self._orig_img = pil_img
        self.display_frame.update_idletasks()
        max_w = self.display_frame.winfo_width() - 20
        max_h = self.display_frame.winfo_height() - 20
        img = pil_img.copy()
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        scale = self.scale_var.get()
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
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
        self._photo = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, image=self._photo, anchor="nw")
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))
        canvas.bind_all("<Button-4>", lambda _event: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda _event: canvas.yview_scroll(1, "units"))

    def _display_x_image(self, pil_img: Image.Image) -> None:
        self._orig_img2 = pil_img
        self.display_frame2.update_idletasks()
        max_w = self.display_frame2.winfo_width() - 20
        max_h = self.display_frame2.winfo_height() - 20
        img = pil_img.copy()
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        scale = self.scale_var2.get()
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
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
        self._photo2 = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, image=self._photo2, anchor="nw")
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))
        canvas.bind_all("<Button-4>", lambda _event: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda _event: canvas.yview_scroll(1, "units"))


if __name__ == "__main__":
    main()

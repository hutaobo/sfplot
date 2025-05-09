# src/sfplot/gui/gui_app.py

"""SFPlot Cophenetic Heatmap GUI — 带滚动条的 Figure 嵌入

- 后台线程计算 cophenetic distances。
- 调用 plot_cophenetic_heatmap 绘图，直接在当前 Figure 上嵌入。
- 在可滚动 Canvas 中展示，横向 & 纵向滚动。
"""

from __future__ import annotations
import queue
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import END, filedialog, messagebox, ttk
import matplotlib
# 使用 TkAgg 后端
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
import pandas as pd  # noqa: E402

from sfplot import compute_cophenetic_distances_from_df, plot_cophenetic_heatmap  # noqa: E402

# 日志文件
_EXEC_DIR = Path(sys.argv[0]).resolve().parent
_LOG_FILE = _EXEC_DIR / "cellgps_error.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def _write_error(msg: str) -> None:
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n---\n")

class MainApp(tk.Tk):
    """主窗口：支持横向 & 纵向滚动的图像显示"""
    _STEPS = {"start": 0, "csv_read": 10, "calc_dist": 50, "plot": 90, "done": 100}

    def __init__(self) -> None:
        super().__init__()
        self.title("SFPlot Cophenetic Heatmap")
        self.geometry("900x700")
        self.report_callback_exception = self._handle_exception
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()

        # 顶部工具栏：按钮 & 进度
        top_frame = tk.Frame(self)
        top_frame.pack(fill="x", padx=5, pady=5)
        self.load_btn = tk.Button(top_frame, text="选择 CSV 并绘图", command=self._ask_file)
        self.load_btn.pack(side="left")
        self._progress = ttk.Progressbar(top_frame, mode="determinate", length=200, maximum=100)
        self._progress.pack(side="left", padx=10)
        self._progress_label = tk.Label(top_frame, text="0%")
        self._progress_label.pack(side="left")

        # 日志区
        log_frame = tk.LabelFrame(self, text="运行日志")
        log_frame.pack(side="left", fill="y", padx=5, pady=5)
        self.log_text = tk.Text(log_frame, width=30, state="disabled")
        self.log_text.pack(fill="both", expand=True)

        # 可滚动绘图区
        display_frame = tk.LabelFrame(self, text="绘图区域")
        display_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        # grid 布局，用于滚动条
        display_frame.rowconfigure(0, weight=1)
        display_frame.columnconfigure(0, weight=1)
        # Canvas + 滚动条
        self._scroll_canvas = tk.Canvas(display_frame)
        self._scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self._hbar = ttk.Scrollbar(display_frame, orient="horizontal", command=self._scroll_canvas.xview)
        self._hbar.grid(row=1, column=0, sticky="ew")
        self._vbar = ttk.Scrollbar(display_frame, orient="vertical", command=self._scroll_canvas.yview)
        self._vbar.grid(row=0, column=1, sticky="ns")
        self._scroll_canvas.configure(xscrollcommand=self._hbar.set, yscrollcommand=self._vbar.set)

        # 用于嵌入的 FigureCanvasTkAgg
        self._canvas: FigureCanvasTkAgg | None = None

        self.after(100, self._poll_queue)
        self._log("程序启动成功")

    def _log(self, msg: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(END, msg + "\n")
        self.log_text.see(END)
        self.log_text.configure(state="disabled")
        print(msg)

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        err = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        _write_error(err)
        self._log(f"错误: {exc_value}")
        messagebox.showerror("程序错误", f"发生错误:\n{exc_value}\n\n已记录到日志: {_LOG_FILE}")

    def _ask_file(self) -> None:
        self._log(">> 打开文件…")
        path = filedialog.askopenfilename(title="选择 CSV 文件", filetypes=[("CSV", "*.csv")])
        if not path:
            self._log(">> 取消。")
            return
        self.load_btn.configure(state="disabled")
        threading.Thread(target=self._worker, args=(Path(path),), daemon=True).start()

    def _worker(self, csv_path: Path) -> None:
        try:
            self._queue.put(("progress", self._STEPS["start"]))
            self._queue.put(("log", f"读取 CSV: {csv_path.name}"))
            df = pd.read_csv(csv_path)
            self._queue.put(("progress", self._STEPS["csv_read"]))

            self._queue.put(("log", "计算 cophenetic distances…"))
            row_df, _ = compute_cophenetic_distances_from_df(
                df, x_col="xc", y_col="yc", celltype_col="target", output_dir=None
            )
            self._queue.put(("progress", self._STEPS["calc_dist"]))

            self._queue.put(("log", "调用 plot_cophenetic_heatmap 绘制…"))
            plot_cophenetic_heatmap(
                row_df,
                output_filename=None,
                matrix_name="row_coph",
                sample="Tonsil",
            )
            fig = plt.gcf()
            self._queue.put(("progress", self._STEPS["plot"]))
            self._queue.put(("log", "图像生成完成，准备嵌入…"))

            self._queue.put(("figure", fig))
            self._queue.put(("progress", self._STEPS["done"]))
            self._queue.put(("log", "完成。"))
        except Exception as e:
            err_msg = f"后台错误: {e}\n{traceback.format_exc()}"
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
                    messagebox.showerror("错误", payload)
                elif tag == "done":
                    self.load_btn.configure(state="normal")
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

        def _embed_figure(self, fig: plt.Figure) -> None:
        # 清理旧画布
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
        # 创建新的 FigureCanvasTkAgg
        self._canvas = FigureCanvasTkAgg(fig, master=self._scroll_canvas)
        self._canvas.draw()
        widget = self._canvas.get_tk_widget()
        # 将 widget 嵌入 scroll_canvas
        self._scroll_canvas.create_window(0, 0, anchor="nw", window=widget)
        # 更新 scrollregion
        self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))

        # 保持 canvas_container 更新
        widget.bind('<Configure>', lambda e: self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all")))

    def _on_mousewheel(self, event):
        # 垂直滚动
        self._scroll_canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def _on_shift_mousewheel(self, event):
        # 水平滚动
        self._scroll_canvas.xview_scroll(-1 * int(event.delta / 120), "units")

# 绑定全局鼠标滚轮事件，到滚动画布
    def bind_scroll(self):
        # Windows 和 macOS 鼠标滚轮
        self._scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._scroll_canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)
        # Linux 鼠标滚轮
        self._scroll_canvas.bind_all("<Button-4>", lambda e: self._scroll_canvas.yview_scroll(-1, "units"))
        self._scroll_canvas.bind_all("<Button-5>", lambda e: self._scroll_canvas.yview_scroll(1, "units"))

# 在初始化后调用绑定
    def _post_init_bind(self):
        self.bind_scroll()

# 程序入口

def main() -> None:
    app = MainApp()
    app._post_init_bind()
    app.mainloop()

if __name__ == "__main__":
    main()() -> None:
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    main()

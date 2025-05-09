# src/sfplot/gui/gui_app.py

"""SFPlot Cophenetic Heatmap GUI — 直接使用 plot_cophenetic_heatmap 绘制并展示 Figure

- 后台线程计算 cophenetic distances。
- 调用 plot_cophenetic_heatmap 绘图，直接从当前 Figure 获取并嵌入 GUI。
- 提供可滚动画布，保持原图大小。
"""

from __future__ import annotations
import queue
import sys
import threading
import traceback
from pathlib import Path
from tkinter import END, filedialog, messagebox, ttk
import tkinter as tk

import matplotlib
# 使用交互式 TkAgg 后端
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
import pandas as pd  # noqa: E402

from sfplot import compute_cophenetic_distances_from_df, plot_cophenetic_heatmap  # noqa: E402

# -------------------------------------------------------------
# 日志文件
# -------------------------------------------------------------
_EXEC_DIR = Path(sys.argv[0]).resolve().parent
_LOG_FILE = _EXEC_DIR / "cellgps_error.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def _write_error(msg: str) -> None:
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n---\n")

# -------------------------------------------------------------
# GUI 主窗口
# -------------------------------------------------------------
class MainApp(tk.Tk):
    """主窗口：后台线程 + Matplotlib Figure 嵌入 + 可滚动展示"""
    _STEPS = {"start": 0, "csv_read": 10, "calc_dist": 50, "plot": 90, "done": 100}

    def __init__(self) -> None:
        super().__init__()
        self.title("SFPlot Cophenetic Heatmap")
        self.geometry("900x700")
        self.report_callback_exception = self._handle_exception
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()

        # 顶部控件
        frame = tk.Frame(self)
        frame.pack(fill="x", padx=5, pady=5)
        self.load_btn = tk.Button(frame, text="选择 CSV 并绘图", command=self._ask_file)
        self.load_btn.pack(side="left")
        self._progress = ttk.Progressbar(frame, mode="determinate", length=200, maximum=100)
        self._progress.pack(side="left", padx=10)
        self._progress_label = tk.Label(frame, text="0%")
        self._progress_label.pack(side="left")

        # 日志区
        log_frame = tk.LabelFrame(self, text="运行日志")
        log_frame.pack(side="left", fill="y", padx=5, pady=5)
        self.log_text = tk.Text(log_frame, width=30, state="disabled")
        self.log_text.pack(fill="both", expand=True)

        # 可滚动绘图区
        display_frame = tk.LabelFrame(self, text="绘图区域")
        display_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        self.canvas_container = display_frame
        self._canvas: FigureCanvasTkAgg | None = None
        self._photo = None  # 占位

        self.after(100, self._poll_queue)
        self._log("程序启动成功")

    def _log(self, msg: str) -> None:
        self.log_text.config(state="normal")
        self.log_text.insert(END, msg + "\n")
        self.log_text.see(END)
        self.log_text.config(state="disabled")
        print(msg)

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        err = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        _write_error(err)
        self._log(f"错误: {exc_value}")
        messagebox.showerror(
            "程序错误",
            f"发生错误:\n{exc_value}\n\n已记录到日志: {_LOG_FILE}"
        )

    def _ask_file(self) -> None:
        self._log(">> 打开文件…")
        path = filedialog.askopenfilename(title="选择 CSV 文件", filetypes=[("CSV", "*.csv")])
        if not path:
            self._log(">> 已取消。")
            return
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

            # 绘图
            self._queue.put(("log", "调用 plot_cophenetic_heatmap 绘制…"))
            plot_cophenetic_heatmap(
                row_df,
                output_filename=None,
                matrix_name="row_coph",
                sample="Tonsil",
            )
            # 从当前 Figure 获取
            fig = plt.gcf()
            self._queue.put(("progress", self._STEPS["plot"]))
            self._queue.put(("log", "图像生成完成，准备嵌入…"))

            # 嵌入 Figure
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
                    self._progress.config(value=val)
                    self._progress_label.config(text=f"{val}%")
                elif tag == "figure":
                    self._embed_figure(payload)
                elif tag == "error":
                    messagebox.showerror("错误", f"{payload}")
                elif tag == "done":
                    self.load_btn.config(state="normal")
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    def _embed_figure(self, fig: plt.Figure) -> None:
        # 清理旧画布
        if self._canvas:
            self._canvas.get_tk_widget().destroy()
        # 嵌入新的 Figure
        self._canvas = FigureCanvasTkAgg(fig, master=self.canvas_container)
        self._canvas.draw()
        widget = self._canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)

def main() -> None:
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    main()

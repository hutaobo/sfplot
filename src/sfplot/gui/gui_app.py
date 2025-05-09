# src/sfplot/gui/gui_app.py

"""SFPlot Cophenetic Heatmap GUI — 支持 PDF 嵌入和鼠标滚轮/滚动条"""

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

# PDF 转图片
from pdf2image import convert_from_path  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

from sfplot import compute_cophenetic_distances_from_df, plot_cophenetic_heatmap  # noqa: E402

# 日志文件
_EXEC_DIR = Path(sys.argv[0]).resolve().parent
_LOG_FILE = _EXEC_DIR / "cellgps_error.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def _write_error(msg: str) -> None:
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n---\n")

class MainApp(tk.Tk):
    """主窗口：后台线程 + 可滚动 Canvas + 鼠标滚轮"""
    _STEPS = {"start": 0, "csv_read": 10, "calc_dist": 50, "plot": 90, "done": 100}

    def __init__(self) -> None:
        super().__init__()
        self.title("SFPlot Cophenetic Heatmap")
        self.geometry("900x700")
        self.report_callback_exception = self._handle_exception
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()

        # 顶部工具栏
        top = tk.Frame(self)
        top.pack(fill="x", padx=5, pady=5)
        self.load_btn = tk.Button(top, text="选择 CSV 并绘图", command=self._ask_file)
        self.load_btn.pack(side="left")
        self._progress = ttk.Progressbar(top, mode="determinate", length=200, maximum=100)
        self._progress.pack(side="left", padx=10)
        self._progress_label = tk.Label(top, text="0%")
        self._progress_label.pack(side="left")

        # 日志区
        log_frame = tk.LabelFrame(self, text="运行日志")
        log_frame.pack(side="left", fill="y", padx=5, pady=5)
        self.log_text = tk.Text(log_frame, width=30, state="disabled")
        self.log_text.pack(fill="both", expand=True)

        # 可滚动绘图区
        display_frame = tk.LabelFrame(self, text="绘图区域")
        display_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        display_frame.rowconfigure(0, weight=1)
        display_frame.columnconfigure(0, weight=1)
        self._scroll_canvas = tk.Canvas(display_frame)
        self._scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self._hbar = ttk.Scrollbar(display_frame, orient="horizontal", command=self._scroll_canvas.xview)
        self._hbar.grid(row=1, column=0, sticky="ew")
        self._vbar = ttk.Scrollbar(display_frame, orient="vertical", command=self._scroll_canvas.yview)
        self._vbar.grid(row=0, column=1, sticky="ns")
        self._scroll_canvas.configure(xscrollcommand=self._hbar.set, yscrollcommand=self._vbar.set)

        # 当前图像引用
        self._photo = None  # type: ImageTk.PhotoImage

        # 绑定滚动
        self.bind_scroll()

        # 启动队列
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
            self._log(">> 已取消。")
            return
        self.load_btn.configure(state="disabled")
        threading.Thread(target=self._worker, args=(Path(path),), daemon=True).start()

    def _worker(self, csv_path: Path) -> None:
        try:
            self._queue.put(("progress", self._STEPS["start"]))
            self._queue.put(("log", f"读取 CSV: {csv_path.name}"))
            df = pd.read_csv(csv_path)
            self._queue.put(("progress", self._STEPS["csv_read"]))

            self._queue.put(("log", "调用 plot_cophenetic_heatmap 绘图并生成 PDF…"))
            row_df, _ = compute_cophenetic_distances_from_df(
                df, x_col="xc", y_col="yc", celltype_col="target", output_dir=None
            )
            pdf_path = _EXEC_DIR / "StructureMap.pdf"
            plot_cophenetic_heatmap(
                row_df,
                output_filename=str(pdf_path),
                matrix_name="row_coph",
                sample="",
            )
            self._queue.put(("progress", self._STEPS["plot"]))
            self._queue.put(("log", f"PDF 已保存: {pdf_path}"))

            self._queue.put(("log", "PDF 转图片…"))
            pages = convert_from_path(str(pdf_path), dpi=200, first_page=1, last_page=1)
            img = pages[0]
            self._queue.put(("progress", self._STEPS["done"]))
            self._queue.put(("log", "完成，准备展示…"))
            self._queue.put(("image", img))
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
                elif tag == "image":
                    self._embed_image(payload)
                elif tag == "error":
                    messagebox.showerror("错误", payload)
                elif tag == "done":
                    self.load_btn.configure(state="normal")
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    def _embed_image(self, pil_img: Image.Image) -> None:
        # 清理 Canvas
        self._scroll_canvas.delete("all")
        # 转 PhotoImage
        self._photo = ImageTk.PhotoImage(pil_img)
        # 放置图像
        self._scroll_canvas.create_image(0, 0, image=self._photo, anchor="nw")
        # 更新 scrollregion
        self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))

    def _on_mousewheel(self, event):
        self._scroll_canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def _on_shift_mousewheel(self, event):
        self._scroll_canvas.xview_scroll(-1 * int(event.delta / 120), "units")

    def bind_scroll(self) -> None:
        self._scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._scroll_canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self._scroll_canvas.bind_all("<Button-4>", lambda e: self._scroll_canvas.yview_scroll(-1, "units"))
        self._scroll_canvas.bind_all("<Button-5>", lambda e: self._scroll_canvas.yview_scroll(1, "units"))


def main() -> None:
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    main()

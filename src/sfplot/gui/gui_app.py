
# src/sfplot/gui/gui_app.py

"""SFPlot Cophenetic Heatmap GUI

使用 Tkinter + Matplotlib 提供交互式热图绘制界面。
在打包 (PyInstaller) 或源码环境下运行均可。
"""

from __future__ import annotations

import os
import sys
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import matplotlib

# **关键修改 1**
# 使用交互式 TkAgg 后端，而不是非 GUI 的 "Agg"
matplotlib.use("TkAgg")  # 必须位于 pyplot 导入之前

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
import pandas as pd  # noqa: E402

# 业务逻辑函数（项目内模块）
from sfplot import (  # noqa: E402
    compute_cophenetic_distances_from_df,
    plot_cophenetic_heatmap,
)

# -------------------------------------------------------------
# 日志文件设置
# -------------------------------------------------------------

_EXEC_DIR = Path(sys.argv[0]).resolve().parent
_LOG_FILE = _EXEC_DIR / "cellgps_error.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def _write_error(msg: str) -> None:
    """将异常信息写入日志文件"""
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n---\n")

# -------------------------------------------------------------
# GUI 主窗口
# -------------------------------------------------------------

class MainApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SFPlot Cophenetic Heatmap")
        self.geometry("900x700")

        # 捕获未处理异常，避免窗口闪退
        self.report_callback_exception = self._handle_exception

        # ---------------- 控件布局 ----------------
        # 按钮区
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", padx=5, pady=5)
        self.load_btn = tk.Button(btn_frame, text="选择 CSV 并绘图", command=self._load_and_plot)
        self.load_btn.pack(side="left")

        # 日志区
        log_frame = tk.LabelFrame(self, text="运行日志")
        log_frame.pack(side="left", fill="y", padx=5, pady=5)
        self.log_text = tk.Text(log_frame, width=30, state="disabled")
        self.log_text.pack(fill="both", expand=True)

        # 绘图区
        canvas_frame = tk.LabelFrame(self, text="绘图区域")
        canvas_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        self.canvas_container = canvas_frame
        self._canvas: FigureCanvasTkAgg | None = None

        self._log("程序启动成功")

    # ---------------- 工具方法 ----------------
    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        self._log(f"错误: {exc_value}")
        _write_error(error_msg)
        messagebox.showerror("程序错误", f"发生错误:\n{exc_value}\n\n详细信息已记录到日志: {_LOG_FILE}")

    def _log(self, msg: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")
        print(msg)

    # ---------------- 主流程 ----------------
    def _load_and_plot(self) -> None:
        self._log(">> 打开文件对话框…")
        path = filedialog.askopenfilename(title="选择 CSV 文件", filetypes=[("CSV 文件", "*.csv")])
        if not path:
            self._log(">> 未选择任何文件，取消。")
            return

        try:
            self._log(f">> 读取 CSV：{Path(path).name} …")
            df = pd.read_csv(path)
            self._log(">> 数据读取成功。")
        except Exception as e:
            messagebox.showerror("错误", f"无法读取文件：\n{e}")
            self._log(f">> 读取失败：{e}")
            return

        _orig_savefig, _orig_close = plt.savefig, plt.close
        plt.savefig = lambda *a, **k: None
        plt.close = lambda *a, **k: None

        try:
            self._log(">> 计算 cophenetic distances …")
            compute_cophenetic_distances_from_df(
                df,
                x_col="xc",
                y_col="yc",
                celltype_col="target",
                output_dir=None,
            )

            fig = plt.gcf()
            axes_cnt = len(fig.axes)
            images_cnt = len(fig.axes[0].images) if axes_cnt else 0
            self._log(f">> 调试: Axes={axes_cnt}, images_in_axes0={images_cnt}")

            if not (axes_cnt and images_cnt):
                raise RuntimeError("未生成热图图像，可能绘图函数未正确执行")

            self._log(">> 绘制 heatmap …")
            mat = fig.axes[0].images[0].get_array()
            plot_cophenetic_heatmap(matrix=mat, matrix_name="row_coph", sample="Tonsil")

        except Exception as e:
            messagebox.showerror("错误", f"绘图过程中出错：\n{e}")
            self._log(f">> 绘图失败：{e}")
            return
        finally:
            plt.savefig, plt.close = _orig_savefig, _orig_close

        self._log(">> 嵌入结果到界面 …")
        self._embed_figure(plt.gcf())
        self._log(">> 完成。")

    # ---------------- 绘图嵌入 ----------------
    def _embed_figure(self, fig: "plt.Figure") -> None:
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()

        self._canvas = FigureCanvasTkAgg(fig, master=self.canvas_container)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

# -------------------------------------------------------------
# 程序入口
# -------------------------------------------------------------

def main() -> None:
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error_msg = f"程序启动失败: {exc}\n{traceback.format_exc()}"
        _write_error(error_msg)
        print(error_msg)
        messagebox.showerror("启动错误", f"程序启动失败:\n{exc}\n\n详细信息已写入日志: {_LOG_FILE}")
        input("按回车键退出…")

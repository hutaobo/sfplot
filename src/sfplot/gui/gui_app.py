# src/sfplot/gui/gui_app.py

import os
import sys
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib

# 为了和 TkAgg 配合，先设置后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sfplot import (
    compute_cophenetic_distances_from_df,
    plot_cophenetic_heatmap,
)

# 设置日志目录
log_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
log_file = os.path.join(log_dir, 'cellgps_error.log')

# 确保日志目录存在
os.makedirs(os.path.dirname(log_file), exist_ok=True)


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SFPlot Cophenetic Heatmap")
        self.geometry("900x700")

        # 添加全局异常处理
        self.report_callback_exception = self.handle_exception

        # ——— 按钮区 —————————————————————
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", padx=5, pady=5)
        self.load_btn = tk.Button(
            btn_frame, text="选择 CSV 并绘图", command=self.load_and_plot
        )
        self.load_btn.pack(side="left")

        # ——— 日志区 —————————————————————
        log_frame = tk.LabelFrame(self, text="运行日志")
        log_frame.pack(side="left", fill="y", padx=5, pady=5)
        self.log_text = tk.Text(log_frame, width=30, state="disabled")
        self.log_text.pack(fill="both", expand=True)

        # ——— 画布区 —————————————————————
        canvas_frame = tk.LabelFrame(self, text="绘图区域")
        canvas_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        # 用于放置 FigureCanvasTkAgg
        self.canvas_container = canvas_frame

        # 当前画布引用
        self.canvas = None

        # 记录初始化完成
        self.log("程序启动成功")

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """处理未捕获的异常"""
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        self.log(f"错误: {error_msg}")

        # 写入日志文件
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{error_msg}\n---\n")

        # 显示错误对话框
        messagebox.showerror("程序错误", f"发生错误:\n{exc_value}\n\n详细信息已记录到日志")

    def log(self, msg: str):
        """在日志区追加一行文本"""
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

        # 同时输出到控制台
        print(msg)

    def load_and_plot(self):
        """读取 CSV 并调用 sfplot 画图，最后嵌入到界面"""
        # 1) 选文件
        self.log(">> 打开文件对话框…")
        path = filedialog.askopenfilename(
            title="选择 CSV 文件",
            filetypes=[("CSV 文件", "*.csv")],
        )
        if not path:
            self.log(">> 未选择任何文件，取消。")
            return

        # 2) 读数据
        try:
            self.log(f">> 读取 CSV：{os.path.basename(path)} …")
            df = pd.read_csv(path)
            self.log(">> 数据读取成功。")
        except Exception as e:
            messagebox.showerror("错误", f"无法读取文件：\n{e}")
            self.log(f">> 读取失败：{e}")
            return

        # 3) 屏蔽 plt.savefig/plt.close（内部调用时不实际写磁盘或关窗口）
        _orig_savefig = plt.savefig
        _orig_close = plt.close
        plt.savefig = lambda *args, **kwargs: None
        plt.close = lambda *args, **kwargs: None

        # 4) 计算并绘图
        try:
            self.log(">> 计算 cophenetic distances …")
            compute_cophenetic_distances_from_df(
                df,
                x_col="xc",
                y_col="yc",
                celltype_col="target",
                output_dir=None,
            )
            self.log(">> 绘制 heatmap …")
            # 从当前 Axes 取图像矩阵
            mat = plt.gcf().axes[0].images[0].get_array()
            plot_cophenetic_heatmap(
                matrix=mat, matrix_name="row_coph", sample="Tonsil"
            )
        except Exception as e:
            messagebox.showerror("错误", f"绘图过程中出错：\n{e}")
            self.log(f">> 绘图失败：{e}")
            return
        finally:
            # 恢复原函数
            fig = plt.gcf()
            plt.savefig = _orig_savefig
            plt.close = _orig_close

        # 5) 嵌入或更新画布
        self.log(">> 嵌入结果到界面 …")
        self._embed_figure(fig)
        self.log(">> 完成。")

    def _embed_figure(self, fig: plt.Figure):
        """将 Matplotlib Figure 嵌入到 canvas_container"""
        # 如果已存在旧画布，先销毁
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_container)
        self.canvas.draw()
        w = self.canvas.get_tk_widget()
        w.pack(fill="both", expand=True)


def main():
    app = MainApp()
    app.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"程序启动失败: {str(e)}\n{traceback.format_exc()}"
        # 写入错误日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{error_msg}\n---\n")
        # 显示错误
        print(error_msg)
        # 弹出错误对话框
        import tkinter.messagebox as mb

        mb.showerror("启动错误", f"程序启动失败:\n{str(e)}\n\n详细信息已写入日志: {log_file}")
        # 等待用户确认，防止控制台闪退
        input("按回车键退出...")

# src/sfplot/gui/gui_app.py

import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sfplot import (
    compute_cophenetic_distances_from_df,
    plot_cophenetic_heatmap,
)


def main():
    root = tk.Tk()
    root.title("SFPlot Cophenetic Heatmap")
    root.geometry("800x600")

    def load_and_plot():
        path = filedialog.askopenfilename(
            title="选择CSV文件",
            filetypes=[("CSV 文件", "*.csv")]
        )  # 打开文件对话框:contentReference[oaicite:7]{index=7}
        if not path:
            return
        try:
            df = pd.read_csv(path)  # 读取 CSV:contentReference[oaicite:8]{index=8}
        except Exception as e:
            messagebox.showerror("错误", f"无法读取文件：\n{e}")
            return

        # 计算 Cophenetic 距离矩阵
        row_coph, col_coph = compute_cophenetic_distances_from_df(
            df,
            x_col="xc",
            y_col="yc",
            celltype_col="target",
            output_dir=None,
        )

        # 生成热图 Figure
        fig = plot_cophenetic_heatmap(
            matrix=row_coph,
            matrix_name="row_coph",
            sample="Tonsil",
        )

        # 如果之前已经有图表，则清除
        for widget in root.pack_slaves():
            if isinstance(widget, tk.Canvas):
                widget.destroy()

        # 将 Matplotlib Figure 嵌入到 Tkinter 窗口中:contentReference[oaicite:9]{index=9}
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    btn = tk.Button(root, text="选择 CSV 并绘图", command=load_and_plot)
    btn.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()

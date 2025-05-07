# src/sfplot/gui/gui_app.py

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from sfplot import (
    compute_cophenetic_distances_from_df,
    plot_cophenetic_heatmap,
)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_and_plot(root):
    # 1) 打开文件对话框读取 CSV
    path = filedialog.askopenfilename(
        title="选择CSV文件",
        filetypes=[("CSV 文件", "*.csv")]
    )  # :contentReference[oaicite:4]{index=4}
    if not path:
        return
    try:
        df = pd.read_csv(path)  # :contentReference[oaicite:5]{index=5}
    except Exception as e:
        messagebox.showerror("错误", f"无法读取文件：\n{e}")
        return

    # 2) Monkey-patch savefig 与 close
    _orig_savefig = plt.savefig
    _orig_close   = plt.close
    plt.savefig = lambda *args, **kwargs: None
    plt.close   = lambda *args, **kwargs: None  # :contentReference[oaicite:6]{index=6}

    # 3) 调用包内绘图函数（内部的 savefig/close 均被抑制）
    compute_cophenetic_distances_from_df(
        df, x_col="xc", y_col="yc", celltype_col="target", output_dir=None
    )
    plot_cophenetic_heatmap(
        matrix=plt.gcf().axes[0].images[0].get_array(),  # 例：若返回图像保存在当前 Axes
        matrix_name="row_coph",
        sample="Tonsil",
    )

    # 4) 获取当前 Figure 并还原原函数
    fig = plt.gcf()
    plt.savefig = _orig_savefig
    plt.close   = _orig_close  # :contentReference[oaicite:7]{index=7}

    # 5) 清除旧画布并嵌入新图
    for widget in root.pack_slaves():
        if isinstance(widget, tk.Canvas):
            widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # :contentReference[oaicite:8]{index=8}

def main():
    root = tk.Tk()
    root.title("SFPlot Cophenetic Heatmap")
    root.geometry("800x600")

    btn = tk.Button(root, text="选择 CSV 并绘图",
                    command=lambda: load_and_plot(root))
    btn.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()

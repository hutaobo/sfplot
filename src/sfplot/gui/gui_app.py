# src/sfplot/gui/gui_app.py

import tkinter as tk
from sfplot.plotting import plot_xyz  # 或你包里其它绘图函数

def main():
    # 1. 创建窗口
    root = tk.Tk()
    root.title("SFPlot 绘图工具")
    root.geometry("600x400")

    # 2. 按钮：点击生成图表
    btn = tk.Button(
        root,
        text="生成图表",
        font=("Arial", 14),
        width=20,
        height=2,
        command=plot_xyz  # 直接调用你的绘图函数
    )
    btn.pack(pady=50)

    # 3. 启动事件循环
    root.mainloop()


# 允许直接 python -m 调用
if __name__ == "__main__":
    main()

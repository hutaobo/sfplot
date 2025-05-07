# src/sfplot/gui/gui_app.py
import tkinter as tk
from sfplot.plotting import plot_xyz  # 导入业务逻辑

def main():
    root = tk.Tk()
    root.title("SFPlot 绘图工具")
    btn = tk.Button(root, text="生成图表", command=plot_xyz)
    btn.pack(padx=20, pady=20)
    root.mainloop()

if __name__ == "__main__":
    main()

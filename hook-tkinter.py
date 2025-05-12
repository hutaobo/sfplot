# hook-tkinter.py
import sys
import os
from PyInstaller.utils.hooks import collect_dynamic_libs

# 收集 tkinter 库本身的 DLL（tk86t.dll、tcl86t.dll 等）
binaries = collect_dynamic_libs('tkinter')

# 把 Conda 环境下的 tcl8.6、tk8.6 目录原封不动打包到 EXE 里
datas = []
prefix = sys.prefix
tcl_root = os.path.join(prefix, 'Library', 'tcl')
for sub in ('tcl8.6', 'tk8.6'):
    src = os.path.join(tcl_root, sub)
    if os.path.isdir(src):
        # 打包后解压到 “tcl/tcl8.6” 和 “tcl/tk8.6”
        datas.append((src, os.path.join('tcl', sub)))

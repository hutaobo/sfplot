# hook-tkinter.py

import sys
import os
from PyInstaller.utils.hooks import collect_dynamic_libs

# 1) 收集 tkinter 的 DLL（tk86t.dll、tcl86t.dll 等）
binaries = collect_dynamic_libs('tkinter')

# 2) 收集 tcl 和 tk 资源目录
datas = []
# Windows 下 conda 环境的 Tcl/Tk 路径，一般在 sys.exec_prefix/tcl/...
tcl_root = os.path.join(sys.exec_prefix, 'tcl', 'tcl8.6')
tk_root  = os.path.join(sys.exec_prefix, 'tcl', 'tk8.6')

if os.path.isdir(tcl_root):
    datas.append((tcl_root, os.path.join('tcl', 'tcl8.6')))
if os.path.isdir(tk_root):
    datas.append((tk_root,  os.path.join('tcl', 'tk8.6')))

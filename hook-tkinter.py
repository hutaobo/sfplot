# hook-tkinter.py

import sys
import os
from PyInstaller.utils.hooks import collect_dynamic_libs

# 1) 收集 tkinter 本身的动态库（tk86t.dll、tcl86t.dll 等）
binaries = collect_dynamic_libs('tkinter')

# 2) 把 Conda 环境下的 tcl8.6、tk8.6 脚本目录完整打进去，
#    并且放到程序运行时解压目录下的 lib/ 下，这样 Tkinter 就能找到 init.tcl 了。
datas = []
prefix = sys.prefix
# Conda 环境的 tcl 脚本一般在 <env>/Library/tcl
tcl_root = os.path.join(prefix, 'Library', 'tcl')
for sub in ('tcl8.6', 'tk8.6'):
    src = os.path.join(tcl_root, sub)
    if os.path.isdir(src):
        # 解压后会在 sys._MEIPASS/lib/tcl8.6 和 lib/tk8.6
        datas.append((src, os.path.join('lib', sub)))

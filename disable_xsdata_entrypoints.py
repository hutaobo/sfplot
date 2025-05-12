# disable_xsdata_entrypoints.py

# 在程序运行时，重写 xsdata.utils.hooks.load_entry_points
# 让它直接返回空 iterable，避免去扫描 importlib.metadata entry_points
try:
    import xsdata.utils.hooks as _xs_hooks
    _xs_hooks.load_entry_points = lambda *args, **kwargs: ()
except ImportError:
    # 如果 xsdata 根本没装，也不用管
    pass

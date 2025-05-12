# disable_xsdata_entrypoints.py

import importlib.metadata

# 先保存原始函数引用
_original_entry_points = importlib.metadata.entry_points
_original_distribution = importlib.metadata.distribution

def _fake_entry_points(*args, **kwargs):
    """
    当用户请求任何 entry_points() 时，
    如果 filter 里包含 xsdata，就直接返回空列表
    否则调用原函数。
    """
    eps = _original_entry_points(*args, **kwargs)
    # importlib.metadata.entry_points() 在 Py ≥ 3.10 返回 EntryPoints 对象，
    # 在更老版本返回 dict-of-lists；统一处理：
    try:
        # 如果是 3.10+，过滤掉 xsdata 相关
        filtered = [ep for ep in eps if not ep.module.startswith("xsdata")]
        return type(eps)(filtered)
    except Exception:
        # 老旧格式，下钻到 group
        new = {}
        for grp, lst in eps.items():
            new[grp] = [ep for ep in lst if not ep.module.startswith("xsdata")]
        return new

def _fake_distribution(name, *args, **kwargs):
    """
    当调用 distribution('xsdata-pydantic-basemodel') 时，
    直接模拟一个“空”的包，避免找不到 metadata 报错。
    其它包照常调用。
    """
    if name.startswith("xsdata"):
        # 创建一个假的 Distribution 对象，只要实现 entry_points 方法
        class FakeDist:
            def entry_points(self_inner):
                return []
        return FakeDist()
    return _original_distribution(name, *args, **kwargs)

# 最后把 importlib.metadata 里相关函数替换掉
importlib.metadata.entry_points = _fake_entry_points
importlib.metadata.distribution = _fake_distribution

# （可选）也保留你原来的 xsdata hook，以防有人直接 import xsdata.utils.hooks
import xsdata.utils.hooks
def _no_entrypoints(*args, **kwargs):
    return []
xsdata.utils.hooks.load_entry_points = _no_entrypoints

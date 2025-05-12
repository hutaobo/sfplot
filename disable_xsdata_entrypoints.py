# disable_xsdata_entrypoints.py

import sys

# 1) 先给 xsdata 自身的 hook 直接打补丁，保证万无一失
try:
    import xsdata.utils.hooks
    xsdata.utils.hooks.load_entry_points = lambda *args, **kwargs: []
except ImportError:
    pass

# 2) 支持 Py3.8+ 自带的 importlib.metadata，也兼容旧版的 importlib_metadata
try:
    import importlib.metadata as _md
except ImportError:
    import importlib_metadata as _md

# 保存原始引用
_original_entry_points = _md.entry_points
_original_distribution = _md.distribution

def _fake_entry_points(**kwargs):
    """
    拦截任何 importlib.metadata.entry_points(...) 调用，
    过滤掉所有 module/value 里以 xsdata 打头的 entry‐points。
    """
    eps = _original_entry_points(**kwargs)

    # 新版返回 EntryPoints 对象
    try:
        # EntryPoints 是一个可迭代的集合
        filtered = [ep for ep in eps if not ep.value.split(":")[0].startswith("xsdata")]
        return type(eps)(filtered)
    except Exception:
        # 旧版返回 dict-of-lists
        new = {}
        for group, lst in eps.items():
            new[group] = [ep for ep in lst if not ep.value.split(":")[0].startswith("xsdata")]
        return new

def _fake_distribution(name, *args, **kwargs):
    """
    拦截 importlib.metadata.distribution("xsdata…")，
    返回一个最小化的 Dummy 对象，只实现 entry_points()。
    """
    if name.startswith("xsdata"):
        class _DummyDist:
            def entry_points(self):
                return []
        return _DummyDist()
    return _original_distribution(name, *args, **kwargs)

# 最后替换掉 importlib.metadata / importlib_metadata 里的两个函数
_md.entry_points   = _fake_entry_points
_md.distribution   = _fake_distribution

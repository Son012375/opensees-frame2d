"""OpenSeesPy 호환 모듈.

openseespy.opensees (Python 3.8 전용)와
opensees.openseespy.Model (Python 3.12+) 중 사용 가능한 것을 자동 선택.

Usage:
    from core.ops_compat import ops
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    ...
"""
from __future__ import annotations

import ctypes
import os
import sys

# ── 1차: openseespy.opensees (레거시, Python 3.8 전용) ──
try:
    import openseespy.opensees as ops
except (RuntimeError, ImportError):
    ops = None

# ── 2차: opensees 패키지 (Python 3.12+) ──
if ops is None:
    try:
        # DLL 사전 로드 (Windows)
        if sys.platform == "win32":
            _site = os.path.normpath(os.path.join(
                os.path.dirname(sys.executable), "..", "Lib", "site-packages"
            ))
            _bin = os.path.join(_site, "opensees", "bin")
            _lib = os.path.normpath(os.path.join(
                os.path.dirname(sys.executable), "..", "Library", "bin"
            ))
            for _d in [_bin, _lib]:
                if os.path.isdir(_d):
                    os.add_dll_directory(_d)
            _dll = os.path.join(_bin, "OpenSeesRT.dll")
            if os.path.exists(_dll):
                ctypes.CDLL(_dll)

        from opensees.openseespy import Model as _Model

        class _OpsCompat:
            """opensees.openseespy.Model을 함수형 API로 감싸는 래퍼.

            ops.wipe(), ops.model(...), ops.node(...) 등을 지원한다.
            """

            def __init__(self):
                self._model: _Model | None = None

            def wipe(self):
                self._model = None

            def model(self, *args, **kwargs):
                # model('basic', '-ndm', 3, '-ndf', 6)
                ndm = 2
                ndf = 3
                for i, a in enumerate(args):
                    if a == '-ndm' and i + 1 < len(args):
                        ndm = int(args[i + 1])
                    if a == '-ndf' and i + 1 < len(args):
                        ndf = int(args[i + 1])
                ndm = kwargs.get('ndm', ndm)
                ndf = kwargs.get('ndf', ndf)
                self._model = _Model(ndm=ndm, ndf=ndf)

            def __getattr__(self, name):
                if name.startswith('_'):
                    raise AttributeError(name)
                if self._model is None:
                    raise RuntimeError("ops.model()을 먼저 호출하세요.")
                return getattr(self._model, name)

        ops = _OpsCompat()

    except Exception as e:
        raise ImportError(
            f"OpenSeesPy를 사용할 수 없습니다. "
            f"openseespy 또는 opensees 패키지를 설치하세요: {e}"
        )

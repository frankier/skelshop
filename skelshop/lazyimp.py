from typing import Any

LAZY_MODULES = {
    # May need GPU
    "pyopenpose": "openpose.pyopenpose",
    "dlib": "dlib",
    "face_recognition": "face_recognition",
    "dlib_models": "face_recognition.api",
    "torch": "torch",
    # Slow imports
    "pandas": "pandas",
    "seaborn": "seaborn",
    "pyplot": "matplotlib.pyplot",
}


def __getattr__(name: str) -> Any:
    import importlib

    if name in LAZY_MODULES:
        full_module_name = LAZY_MODULES[name]
        mod = importlib.import_module(full_module_name)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

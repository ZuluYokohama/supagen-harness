"""Supagen — super agent harness (Prime + multiplane)."""

__version__ = "0.3.1"

# Always put monorepo prime/scripts + harness on sys.path (buddy-safe).
try:
    from .paths import ensure_sys_path

    ensure_sys_path()
except Exception:
    pass

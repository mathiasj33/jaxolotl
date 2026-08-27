"""Import MuJoCo MJX while suppressing its optional-backend startup messages."""

import contextlib
import io
import logging


class _DeviceNoticeFilter(logging.Filter):
    """Drop MJX's repeated device-resolution notice."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("Using JAX default device")


# MJX uses the root logger; root filters survive Hydra handler reconfiguration.
logging.getLogger().addFilter(_DeviceNoticeFilter())

# Warp availability notices are printed while MJX is imported.
with contextlib.redirect_stdout(io.StringIO()):
    from mujoco import mjx  # noqa: F401

__all__ = ["mjx"]

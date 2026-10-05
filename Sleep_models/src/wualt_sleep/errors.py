"""Exception hierarchy.

Everything the pipeline raises deliberately descends from `WualtSleepError`, so a
caller can distinguish "our contract was violated" from "numpy blew up".
"""

from __future__ import annotations

__all__ = [
    "WualtSleepError",
    "SchemaError",
    "FeatureDictionaryError",
    "ConfigError",
    "StateMachineError",
    "LeakageError",
    "NotFittedError",
]


class WualtSleepError(Exception):
    """Base for all pipeline errors."""


class SchemaError(WualtSleepError):
    """A Row/Epoch/event failed its contract (wrong width, bad dtype, NaN timestamp)."""


class FeatureDictionaryError(WualtSleepError):
    """Feature vector does not match `config/features.yaml` (order, count, or names)."""


class ConfigError(WualtSleepError):
    """Config file is malformed, or a value is outside its documented range."""


class StateMachineError(WualtSleepError):
    """An illegal state transition was attempted (§4)."""


class LeakageError(WualtSleepError):
    """A subject appeared in both train and test.

    This is fatal, never a warning. §8 is explicit that random row splits leak subject
    physiology and produce fake accuracy — and sliding-window augmentation makes the
    failure mode trivially easy to hit.
    """


class NotFittedError(WualtSleepError):
    """predict() called before fit()/load()."""

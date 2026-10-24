"""Backend measurements with explicit units and measurement context.

Bounds below are transport sanity limits, not clinical normal ranges.
"""
from typing import Literal
from pydantic import AwareDatetime, Field, model_validator
from .schemas import StrictModel

Quality = Literal['usable', 'limited', 'unreliable', 'unknown']


class HRVReference(StrictModel):
    metric: Literal['rmssd', 'sdnn']
    value_ms: float = Field(ge=0, le=2000)
    window_seconds: float = Field(gt=0, le=86400)
    at: AwareDatetime
    comparable_context: bool | None = Field(default=None, strict=True)


class HRVMeasurement(StrictModel):
    metric: Literal['rmssd', 'sdnn']
    value_ms: float = Field(ge=0, le=2000)
    window_start: AwareDatetime
    window_end: AwareDatetime
    signal_quality: Quality = 'unknown'
    artifact_corrected: bool | None = Field(default=None, strict=True)
    reference: HRVReference | None = None

    @model_validator(mode='after')
    def window(self):
        if not 0 < (self.window_end-self.window_start).total_seconds() <= 86400:
            raise ValueError('HRV window must be positive and at most 24 hours')
        if self.reference and self.reference.at > self.window_start:
            raise ValueError('HRV reference must be from before the measurement window')
        return self


class OtherMeasurement(StrictModel):
    # Each name defines its unit; units cannot be guessed from an arbitrary value.
    name: Literal['spo2_percent', 'respiratory_rate_per_minute', 'skin_temperature_c']
    value: float
    at: AwareDatetime
    signal_quality: Quality = 'unknown'

    @model_validator(mode='after')
    def value_range(self):
        bounds = {'spo2_percent': (0, 100), 'respiratory_rate_per_minute': (0, 150),
                  'skin_temperature_c': (-20, 60)}
        low, high = bounds[self.name]
        if not low <= self.value <= high:
            raise ValueError('Measurement is outside the accepted transport range')
        return self

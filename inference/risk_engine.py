"""
Risk scoring for fAIre alerts.

This module keeps safety logic separate from the model. The CNN answers
"what is in this frame?" while the risk engine answers "how urgent is this
alert once sensor conditions are included?"
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

try:
    from inference.flashover_predictor import FlashoverSignals, compute_flashover_index
except ImportError:  # allows direct local execution during quick experiments
    from flashover_predictor import FlashoverSignals, compute_flashover_index


@dataclass
class SensorReading:
    temperature_c: Optional[float] = None
    smoke_raw: Optional[float] = None
    co_raw: Optional[float] = None
    voc_raw: Optional[float] = None
    oxygen_pct: Optional[float] = None
    heat_flux_kw_m2: Optional[float] = None
    temp_rise_c_per_min: Optional[float] = None
    distance_cm: Optional[float] = None


@dataclass
class FireAlert:
    label: str
    confidence: float
    risk_score: float
    priority: str
    message: str
    sensors: Dict[str, Any]
    flashover_index: float
    flashover_level: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_temperature(temp_c: Optional[float]) -> float:
    """Convert temperature to a 0-1 heat-severity estimate for robot proximity."""
    if temp_c is None:
        return 0.0
    return clamp((temp_c - 25.0) / (80.0 - 25.0))


def normalize_smoke(smoke_raw: Optional[float]) -> float:
    """Normalize analog smoke reading. Arduino analogRead range is usually 0-1023."""
    if smoke_raw is None:
        return 0.0
    return clamp(smoke_raw / 1023.0)


def normalize_co(co_raw: Optional[float]) -> float:
    """Normalize analog CO reading if a CO sensor is attached."""
    if co_raw is None:
        return 0.0
    return clamp(co_raw / 1023.0)


def sensor_to_flashover_signals(sensors: SensorReading) -> FlashoverSignals:
    return FlashoverSignals(
        temperature_c=sensors.temperature_c,
        smoke_raw=sensors.smoke_raw,
        co_raw=sensors.co_raw,
        voc_raw=sensors.voc_raw,
        oxygen_pct=sensors.oxygen_pct,
        heat_flux_kw_m2=sensors.heat_flux_kw_m2,
        temp_rise_c_per_min=sensors.temp_rise_c_per_min,
    )


def compute_risk_score(
    confidence: float,
    sensors: Optional[SensorReading] = None,
    vision_weight: float = 0.60,
    sensor_weight: float = 0.25,
    flashover_weight: float = 0.15,
) -> float:
    """
    Combine model confidence, simple sensor severity, and flashover warning.

    For search-and-rescue, the vision model drives the alert, but high smoke,
    temperature, CO, or flashover conditions should raise the priority.
    """
    confidence = clamp(confidence)
    sensors = sensors or SensorReading()

    sensor_severity = max(
        normalize_temperature(sensors.temperature_c),
        normalize_smoke(sensors.smoke_raw),
        normalize_co(sensors.co_raw),
    )
    flashover = compute_flashover_index(sensor_to_flashover_signals(sensors))
    score = (
        vision_weight * confidence
        + sensor_weight * sensor_severity
        + flashover_weight * flashover.index
    )
    return round(clamp(score), 3)


def priority_from_score(score: float) -> str:
    if score >= 0.80:
        return "critical"
    if score >= 0.60:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def build_alert(label: str, confidence: float, sensors: Optional[SensorReading] = None) -> FireAlert:
    sensors = sensors or SensorReading()
    flashover = compute_flashover_index(sensor_to_flashover_signals(sensors))
    risk_score = compute_risk_score(confidence, sensors)
    priority = priority_from_score(max(risk_score, flashover.index))

    if flashover.level in {"critical", "high"}:
        message = f"{priority.upper()} alert: possible {label} detected with {flashover.level} flashover warning."
    elif priority in {"critical", "high"}:
        message = f"{priority.upper()} search priority: possible {label} detected."
    else:
        message = f"Low-confidence {label} signal; continue scanning."

    return FireAlert(
        label=label,
        confidence=round(float(confidence), 3),
        risk_score=risk_score,
        priority=priority,
        message=message,
        sensors=sensors.__dict__,
        flashover_index=flashover.index,
        flashover_level=flashover.level,
    )

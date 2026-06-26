"""
Risk scoring for fAIre alerts.

This module keeps the safety logic separate from the model. That makes the system
cleaner: the CNN answers "what is in this frame?" while the risk engine answers
"how urgent is this alert?"
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class SensorReading:
    temperature_c: Optional[float] = None
    smoke_raw: Optional[float] = None
    co_raw: Optional[float] = None
    distance_cm: Optional[float] = None


@dataclass
class FireAlert:
    label: str
    confidence: float
    risk_score: float
    priority: str
    message: str
    sensors: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_temperature(temp_c: Optional[float]) -> float:
    """Convert temperature to 0-1 severity. Conservative placeholder scale."""
    if temp_c is None:
        return 0.0
    # 25C normal-ish, 80C severe for electronics/robot proximity.
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


def compute_risk_score(
    confidence: float,
    sensors: Optional[SensorReading] = None,
    vision_weight: float = 0.70,
    sensor_weight: float = 0.30,
) -> float:
    """
    Combine model confidence with sensor severity.

    For search-and-rescue, the vision model drives the alert, but high smoke,
    temperature, or CO should raise the priority of the search area.
    """
    confidence = clamp(confidence)
    sensors = sensors or SensorReading()

    sensor_severity = max(
        normalize_temperature(sensors.temperature_c),
        normalize_smoke(sensors.smoke_raw),
        normalize_co(sensors.co_raw),
    )
    score = vision_weight * confidence + sensor_weight * sensor_severity
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
    risk_score = compute_risk_score(confidence, sensors)
    priority = priority_from_score(risk_score)

    if priority in {"critical", "high"}:
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
    )

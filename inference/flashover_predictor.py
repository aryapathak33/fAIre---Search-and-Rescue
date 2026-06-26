"""
Flashover risk estimation for fAIre.

This module is intentionally written as a transparent heuristic rather than a
black-box safety claim. Flashover is driven mostly by heat release, upper-layer
gas temperature, heat flux, ventilation, and available fuel. A small hobby robot
usually will not have enough sensors to prove flashover is about to happen, but
it can still compute a useful warning index from the signals it does have.

Project-defined index:
    0.00 - 0.39  low
    0.40 - 0.64  guarded
    0.65 - 0.84  high
    0.85 - 1.00  critical

Important: this is not a certified firefighting instrument. It is a prototype
feature for exploring how temperature, smoke, CO, oxygen depletion, VOC/gas
readings, heat flux, and rapid temperature rise could be combined.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FlashoverSignals:
    """Environmental signals that can contribute to a flashover warning."""

    temperature_c: Optional[float] = None
    smoke_raw: Optional[float] = None          # typical analog range: 0-1023
    co_raw: Optional[float] = None             # typical analog range: 0-1023
    voc_raw: Optional[float] = None            # typical analog range: 0-1023
    oxygen_pct: Optional[float] = None         # clean air is around 20.9%
    heat_flux_kw_m2: Optional[float] = None    # lab-grade sensor if available
    temp_rise_c_per_min: Optional[float] = None


@dataclass
class FlashoverEstimate:
    index: float
    level: str
    message: str
    drivers: List[str]
    signals: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_between(value: Optional[float], low: float, high: float) -> float:
    if value is None:
        return 0.0
    if high == low:
        return 0.0
    return clamp((float(value) - low) / (high - low))


def normalize_analog(value: Optional[float], max_value: float = 1023.0) -> float:
    if value is None:
        return 0.0
    return clamp(float(value) / max_value)


def oxygen_depletion_score(oxygen_pct: Optional[float]) -> float:
    """Score oxygen depletion as a ventilation/combustion warning signal."""
    if oxygen_pct is None:
        return 0.0
    # Clean dry air is about 20.9% oxygen. A lower value can signal active
    # combustion, poor ventilation, and dangerous atmosphere. This is a proxy,
    # not a standalone flashover criterion.
    return clamp((20.9 - float(oxygen_pct)) / (20.9 - 12.0))


def flashover_level(index: float) -> str:
    if index >= 0.85:
        return "critical"
    if index >= 0.65:
        return "high"
    if index >= 0.40:
        return "guarded"
    return "low"


def compute_flashover_index(signals: FlashoverSignals) -> FlashoverEstimate:
    """
    Compute a 0.0-1.0 project-defined flashover warning index.

    The formula uses the strongest thermal signal, the strongest chemistry/gas
    signal, and the rate of temperature rise:

        index = 0.55 * thermal + 0.30 * gas_chemistry + 0.15 * trend

    Typical flashover research often references upper-layer gas temperature near
    600 C and/or heat flux near 20 kW/m^2. If those readings are available and
    are crossed, this function forces the index into the critical range.
    """
    temp_score = normalize_between(signals.temperature_c, 80.0, 600.0)
    heat_flux_score = normalize_between(signals.heat_flux_kw_m2, 2.0, 20.0)
    thermal_score = max(temp_score, heat_flux_score)

    smoke_score = normalize_analog(signals.smoke_raw)
    co_score = normalize_analog(signals.co_raw)
    voc_score = normalize_analog(signals.voc_raw)
    oxygen_score = oxygen_depletion_score(signals.oxygen_pct)
    gas_score = max(smoke_score, co_score, voc_score, oxygen_score)

    trend_score = normalize_between(signals.temp_rise_c_per_min, 5.0, 60.0)

    index = 0.55 * thermal_score + 0.30 * gas_score + 0.15 * trend_score

    # Escalation gates based on common lab flashover criteria.
    if signals.temperature_c is not None and signals.temperature_c >= 600.0:
        index = max(index, 0.95)
    if signals.heat_flux_kw_m2 is not None and signals.heat_flux_kw_m2 >= 20.0:
        index = max(index, 0.95)
    if signals.temperature_c is not None and signals.temperature_c >= 400.0:
        index = max(index, 0.70)

    index = round(clamp(index), 3)
    level = flashover_level(index)

    drivers: List[str] = []
    if thermal_score >= 0.60:
        drivers.append("high thermal load")
    if heat_flux_score >= 0.60:
        drivers.append("high heat flux")
    if smoke_score >= 0.60:
        drivers.append("dense smoke / particulates")
    if co_score >= 0.60:
        drivers.append("elevated CO proxy")
    if voc_score >= 0.60:
        drivers.append("combustible gas / VOC proxy")
    if oxygen_score >= 0.60:
        drivers.append("oxygen depletion")
    if trend_score >= 0.60:
        drivers.append("rapid temperature rise")
    if not drivers:
        drivers.append("no dominant flashover driver")

    if level == "critical":
        message = "Critical flashover warning: conditions are near a dangerous rapid-growth range."
    elif level == "high":
        message = "High flashover warning: thermal or gas conditions deserve immediate attention."
    elif level == "guarded":
        message = "Guarded flashover warning: continue monitoring trend and sensor changes."
    else:
        message = "Low flashover warning: no strong rapid-growth signal from current readings."

    return FlashoverEstimate(
        index=index,
        level=level,
        message=message,
        drivers=drivers,
        signals=signals.__dict__,
    )


def explain_fire_chemistry() -> Dict[str, str]:
    """
    Return a simple map of fire-scene chemistry signals the robot can study.

    This is for documentation/UI use. A basic robot may only sense some of these;
    more precise chemical identification requires dedicated calibrated sensors.
    """
    return {
        "O2": "Oxygen supports combustion; falling oxygen can signal a dangerous enclosed atmosphere.",
        "CO": "Carbon monoxide is produced by incomplete combustion and is highly toxic.",
        "CO2": "Carbon dioxide is produced by combustion and can indicate fire growth and oxygen displacement.",
        "VOC / hydrocarbons": "Pyrolysis can release combustible vapors before full ignition.",
        "smoke particulates": "Dense smoke is a practical proxy for pyrolysis, visibility loss, and fire growth.",
        "temperature / heat flux": "Flashover risk is mainly thermal; heat and heat flux dominate the warning index.",
    }

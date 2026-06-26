from inference.risk_engine import SensorReading, build_alert, compute_risk_score, priority_from_score


def test_risk_score_increases_with_confidence():
    low = compute_risk_score(0.2, SensorReading())
    high = compute_risk_score(0.9, SensorReading())
    assert high > low


def test_risk_score_increases_with_smoke():
    clean = compute_risk_score(0.5, SensorReading(smoke_raw=0))
    smoky = compute_risk_score(0.5, SensorReading(smoke_raw=900))
    assert smoky > clean


def test_priority_levels():
    assert priority_from_score(0.85) == "critical"
    assert priority_from_score(0.65) == "high"
    assert priority_from_score(0.45) == "medium"
    assert priority_from_score(0.20) == "low"


def test_alert_shape():
    alert = build_alert("distress", 0.91, SensorReading(temperature_c=55, smoke_raw=500))
    payload = alert.to_dict()
    assert payload["label"] == "distress"
    assert payload["confidence"] == 0.91
    assert "risk_score" in payload
    assert payload["priority"] in {"low", "medium", "high", "critical"}

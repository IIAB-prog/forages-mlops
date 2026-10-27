from src.pipeline import _candidate_passes


def test_candidate_rejected_below_minimum_ap():
    candidate = {"performance_gate_passed": False, "metrics": {"average_precision": 0.40, "roc_auc": 0.70}}
    passed, reasons = _candidate_passes(candidate, None)
    assert not passed
    assert reasons


def test_candidate_rejected_for_regression_against_active_model():
    candidate = {"performance_gate_passed": True, "metrics": {"average_precision": 0.54, "roc_auc": 0.70}}
    baseline = {"metrics": {"average_precision": 0.60, "roc_auc": 0.75}}
    passed, reasons = _candidate_passes(candidate, baseline)
    assert not passed
    assert any("baisse" in reason for reason in reasons)


def test_candidate_passes_quality_gates():
    candidate = {"performance_gate_passed": True, "metrics": {"average_precision": 0.61, "roc_auc": 0.76}}
    baseline = {"metrics": {"average_precision": 0.60, "roc_auc": 0.75}}
    passed, reasons = _candidate_passes(candidate, baseline)
    assert passed
    assert reasons == []

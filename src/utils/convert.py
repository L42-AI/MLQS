"""Conversion utilities — shared helpers used across modules."""

def resample_rule_to_frequency_hz(resample_rule: str) -> float:
    if resample_rule.endswith("ms"):
        return 1000.0 / float(resample_rule.replace("ms", ""))
    if resample_rule.endswith("s"):
        return 1.0 / float(resample_rule.replace("s", ""))
    return 10.0

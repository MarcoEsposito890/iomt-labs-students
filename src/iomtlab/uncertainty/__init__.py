"""Calibration, prediction sets and routing on the sets (Lab 7).

  calibration_split    three disjoint groups of subjects: train,
                       calibration, test.
  ece                  expected calibration error, bins weighted by count.
  fit_temperature      the temperature T; apply_temperature the
                       probabilities of logits / T.
  conformal_quantile, conformal_sets, nonconformity
                       split conformal prediction: score, quantile, set.
  coverage, coverage_by_class, coverage_by_subject, singleton_rate
                       what the sets achieve on test data.
  routing_table, energy_constants, energy_per_decision, link_load
                       the node decides on singletons and forwards the rest:
                       accuracy, energy and channel (uncertainty.routing).
  figures              the figures of Lab 7 (uncertainty.figures, not
                       re-exported here).
"""
from iomtlab.uncertainty.calibration import (   # noqa: F401
    calibration_split, ece, fit_temperature, apply_temperature,
)
from iomtlab.uncertainty.conformal import (     # noqa: F401
    conformal_sets, conformal_quantile, coverage, coverage_by_class,
    singleton_rate,
)
from iomtlab.uncertainty.routing import (       # noqa: F401
    nonconformity, coverage_by_subject, routing_table, energy_constants,
    energy_per_decision, link_load,
)

__all__ = [
    "calibration_split", "ece", "fit_temperature", "apply_temperature",
    "conformal_sets", "conformal_quantile", "coverage", "coverage_by_class",
    "singleton_rate", "nonconformity", "coverage_by_subject", "routing_table",
    "energy_constants", "energy_per_decision", "link_load",
]

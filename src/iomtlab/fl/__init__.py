"""Federated learning and averaging without a server (Lab 8).

  aggregate   fedavg (mean weighted by the number of samples per client),
              clone_with, quantize_delta, dequantize_delta.
  partition   subject_clients (one client per subject) and
              dirichlet_partition (artificial non-IID split).
  rounds      update_bytes, round_cost (bytes, frames, radio and training
              energy of one round), run_fedavg, run_centralised, run_dsgd,
              bytes_to_reach.
  consensus   ring_graph, grid_graph, metropolis_weights, uniform_weights,
              second_eigenvalue, run_consensus, disagreement, iterations_to.
  figures     plot_client_mix, plot_round_cost, plot_consensus
              (iomtlab.fl.figures, not re-exported here).
"""
from iomtlab.fl.aggregate import (          # noqa: F401
    fedavg, clone_with, quantize_delta, dequantize_delta,
)
from iomtlab.fl.partition import (          # noqa: F401
    subject_clients, dirichlet_partition,
)
from iomtlab.fl.rounds import (             # noqa: F401
    update_bytes, round_cost, run_fedavg, run_centralised, run_dsgd, bytes_to_reach,
)
from iomtlab.fl.consensus import (          # noqa: F401
    ring_graph, grid_graph, metropolis_weights, uniform_weights, second_eigenvalue,
    run_consensus, disagreement, iterations_to,
)

__all__ = [
    "fedavg", "clone_with", "quantize_delta", "dequantize_delta",
    "subject_clients", "dirichlet_partition",
    "update_bytes", "round_cost", "run_fedavg", "run_centralised", "run_dsgd", "bytes_to_reach",
    "ring_graph", "grid_graph", "metropolis_weights", "uniform_weights", "second_eigenvalue",
    "run_consensus", "disagreement", "iterations_to",
]

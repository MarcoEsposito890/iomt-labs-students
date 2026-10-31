"""Running a model at the edge: measure it, cut it, send it over the link.

  measure    size_on_disk, time_per_window, nonzero_params: bytes and seconds
             measured on the real model (Labs 4, 5).
  split      split_model, cut_points, cut_cost: where a network can be cut
             and what each boundary costs, with h radio hops (Lab 6).
  cuts       cut_profile, cut_delay, run_split, train_exit, exit_curve: the
             cuts of a model with stages(), their delay, the real split and
             an exit at the cut (Lab 6); split_figures draws them. Imported as
             iomtlab.edge.cuts, not re-exported here.
  regimes    rule, envelope, regime_map, Link, LINKS, link_cost, head_resources,
             place, choose: where each cut wins along k = rho h, with links,
             constraints and the cheapest feasible placement (Lab 6, tesine).
             Imported as iomtlab.edge.regimes, not re-exported here.
  fusion     softmax, fuse_mean, fuse_confidence, fuse, score_bits,
             consensus_cost, gateway_cost: deciding with more than one node
             (Lab 8). Imported as iomtlab.edge.fusion.
  wire       pack_int8, unpack_int8: int8 serialisation with scale and zero
             point (Lab 6).
  compress   int8 weights, pruning, distillation and the four tests on the
             node (Lab 5); compress_figures draws them. Imported as
             iomtlab.edge.compress, not re-exported here.
"""
from iomtlab.edge.measure import (          # noqa: F401
    size_on_disk, time_per_window, nonzero_params,
)
from iomtlab.edge.split import (            # noqa: F401
    split_model, cut_points, cut_cost,
)
from iomtlab.edge.wire import (             # noqa: F401
    pack_int8, unpack_int8,
)

__all__ = [
    "size_on_disk", "time_per_window", "nonzero_params",
    "split_model", "cut_points", "cut_cost",
    "pack_int8", "unpack_int8",
]

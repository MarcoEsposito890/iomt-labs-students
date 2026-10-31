"""Signal side of the pipeline: loading, timing, resampling, windows, spectra, rotations.

Labs 0 and 2 use loading, timing, resample and windowing; from Lab 3 on,
build_windows (dataset) repeats the window construction of Lab 2. spectral and
rotations serve the theory figures of block 1.1 and are covered by tests.
"""
from iomtlab.signal.loading import (
    load_execution, list_executions, load_subjects_info, load_executions_info,
)
from iomtlab.signal.timing import (TimingReport, align_to_common_grid, delivery_regularity,
                                   device_offset, label_agreement, offset_after, shift_clock,
                                   timing_report)
from iomtlab.signal.resample import resample_to_hz, lowpass_ma, on_grid
from iomtlab.signal.windowing import (make_windows, normalize_full_scale, encode_labels,
                                      label_window)
from iomtlab.signal.dataset import build_windows, build_paired_windows
from iomtlab.signal.spectral import (
    WINDOWS, WindowProperties, window_properties, rfft_magnitude, fft_cost,
)
from iomtlab.signal.rotations import (
    GRAVITY, rotation_matrix, rotate, tilt_angles, complementary_filter,
)

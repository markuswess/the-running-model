"""Conversions between the SI units used by the model and the runner units in the UI.

The model is evaluated in SI units (Hz, m/s, s).  Inputs and plots use the
units runners think in: bpm, min/km, km and min.
"""

M_PER_KM = 1000.0
S_PER_MIN = 60.0


def bpm_to_hz(bpm):
    """Heart rate: bpm -> Hz."""
    return bpm / 60.0


def hz_to_bpm(hz):
    """Heart rate: Hz -> bpm."""
    return hz * 60.0


def min_to_s(minutes):
    """Time: min -> s."""
    return minutes * S_PER_MIN


def s_to_min(seconds):
    """Time: s -> min."""
    return seconds / S_PER_MIN


def pace_to_v(pace):
    """Pace in min/km -> velocity in m/s."""
    return M_PER_KM / (S_PER_MIN * pace)


def v_to_pace(v):
    """Velocity in m/s -> pace in min/km."""
    return M_PER_KM / (S_PER_MIN * v)


def pace_to_kmh(pace):
    """Pace in min/km -> speed in km/h."""
    return 60.0 / pace


def k_linear_to_si(k):
    """Linear slope: bpm per km/h -> Hz per m/s."""
    return k * 3.6 / 60.0


def k_sat_to_si(k):
    """Saturating rate: per km/h -> s/m."""
    return k * 3.6


def beta_to_si(beta):
    """Drift rate: bpm/min -> Hz/s."""
    return beta / 3600.0

import numpy as np
from ngapp.app import App
from ngapp.components import *

from .richtext import RichText
from .units import (
    beta_to_si,
    bpm_to_hz,
    hz_to_bpm,
    k_linear_to_si,
    k_sat_to_si,
    min_to_s,
    pace_to_v,
    s_to_min,
)


def _number_input(label, value, handler):
    return NumberInput(
        ui_label=label,
        ui_model_value=value,
        ui_debounce=300,
    ).on_update_model_value(handler)


def _fmt_pace(pace):
    """Format a pace in min/km as mm:ss/km."""
    minutes = int(pace)
    seconds = int(round((pace - minutes) * 60.0))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}/km"


def _nice_ticks(lo, hi, target=7):
    """Ticks with a 1/2/2.5/5 * 10^k step covering [lo, hi]."""
    span = hi - lo
    if span <= 0:
        return np.array([lo])
    raw = span / max(target - 1, 1)
    magnitude = 10.0 ** np.floor(np.log10(raw))
    for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiplier * magnitude
        if step >= raw:
            break
    start = np.ceil(lo / step) * step
    return np.arange(start, hi + 0.5 * step, step)


def _add_distance_axis(fig, v, T_end):
    """Add a top x-axis with the distance (km) covered at constant speed.

    ``v`` is in m/s and ``T_end`` in min; the primary x-axis is time in min.
    """
    total_km = v * min_to_s(T_end) / 1000.0
    if total_km <= 0:
        return
    dist = _nice_ticks(0.0, total_km)
    # distance [km] = v [m/s] * t [s] / 1000, with t in min
    tickvals = dist / (v * 60.0 / 1000.0)
    fig.update_layout(
        xaxis2={
            "overlaying": "x",
            "side": "top",
            "title": "distance (km)",
            "tickvals": tickvals,
            "ticktext": [f"{d:g}" for d in dist],
            "range": [0.0, T_end],
            "showgrid": False,
            "zeroline": False,
        }
    )


def _add_speed_axis(fig, pace_lo, pace_hi, nr_ticks=6):
    """Add a top x-axis with the speed (km/h) for a pace (min/km) x-axis."""
    tickvals = np.linspace(pace_lo, pace_hi, nr_ticks)
    fig.update_layout(
        xaxis2={
            "overlaying": "x",
            "side": "top",
            "title": "speed (km/h)",
            "tickvals": tickvals,
            "ticktext": [f"{60.0 / p:.1f}" for p in tickvals],
            "range": [pace_lo, pace_hi],
            "showgrid": False,
            "zeroline": False,
        }
    )


_INTRO = (
    r"We model the heart-rate response $H(t)$ with the linear "
    r"first-order ODE "
    r"$$\partial_t H(t) = \tau^{-1}\left(H_\infty(t, v, \dots) - H(t)\right),$$"
    r"where $H_\infty$ is the asymptotic heart rate for a given velocity "
    r"$v$ and may itself depend on time and other parameters; $\tau$ is the "
    r"time constant of the response. The model is evaluated in SI units "
    r"(Hz, m/s, s), while the inputs and plots use the units runners think "
    r"in: heart rate in bpm, pace in min/km, time in min and distance in "
    r"km. The overall assumption for now is that the runner is capable of "
    r"running the prescribed paces without slowing down due to fatigue."
)

_FIRST_MODEL = (
    r"We start with a simplified case: we assume that $H_\infty$ is "
    r"independent of time, the runner begins at heart rate $H_0$ and runs "
    r"at a constant pace, so $H_\infty$ is constant. The ODE then has the "
    r"analytic solution "
    r"$$H(t) = H_\infty - (H_\infty - H_0)\,e^{-t/\tau},$$"
    r"i.e. the heart rate approaches $H_\infty$ exponentially with time "
    r"constant $\tau$."
)

_SECOND_MODEL = (
    r"At a constant pace $p$ (min/km) the asymptotic heart rate is no "
    r"longer a free parameter but follows from the pace. Writing "
    r"$v = 60/p$ km/h for the corresponding speed, we compare two simple "
    r"choices for $H_\infty$: a linear relation "
    r"$$H_\infty = H_0 + k_\mathrm{linear}\,v,$$"
    r"and a saturating variant "
    r"$$H_\infty = H_0 + (H_\mathrm{max} - H_0)"
    r"\left(1 - e^{-k_\mathrm{sat}\,v}\right),$$"
    r"where $k_\mathrm{linear}$ is in bpm per km/h, $k_\mathrm{sat}$ in "
    r"per km/h and $H_\mathrm{max}$ is the maximal heart rate. The upper "
    r"plot shows the resulting $H(t)$ for both models, the lower plot "
    r"shows $H_\infty$ against pace. Since pace is inversely proportional "
    r"to speed, the linear model appears curved against pace."
)

_DRIFT_MODEL = (
    r"Over a long run the heart rate tends to creep upwards even at a "
    r"constant pace, a phenomenon known as cardiac drift. We model it by "
    r"letting the asymptotic heart rate grow linearly in time on top of "
    r"the linear speed model: "
    r"$$H_\infty^{\mathrm{drift}}(t) = H_\infty + \beta t, "
    r"\qquad H_\infty = H_0 + k_\mathrm{linear}\,v,$$"
    r"where $\beta$ is the drift rate in bpm/min and $v = 60/p$ km/h is "
    r"the speed for pace $p$. Because the asymptote now moves, the ODE "
    r"$$\partial_t H(t) = \tau^{-1}"
    r"\left(H_\infty^{\mathrm{drift}}(t) - H(t)\right)$$"
    r"is integrated numerically with a fourth-order Runge-Kutta scheme. "
    r"The solid curve is $H(t)$, the dashed curve is the drifting "
    r"asymptote $H_\infty^{\mathrm{drift}}(t)$."
)


def _rk4_drift(H_0, H_inf, beta, tau, T_end):
    """Integrate dH/dt = (H_inf + beta*t - H)/tau with classical RK4.

    All arguments are in SI units (Hz, Hz/s, s).
    """
    nr_points = int(np.clip(np.ceil(20.0 * T_end / tau), 1000, 20000))
    t = np.linspace(0.0, T_end, nr_points + 1)
    dt = t[1] - t[0]
    H = np.empty_like(t)
    H[0] = H_0

    def rhs(tt, h):
        return (H_inf + beta * tt - h) / tau

    for i in range(nr_points):
        ti = t[i]
        hi = H[i]
        k1 = rhs(ti, hi)
        k2 = rhs(ti + 0.5 * dt, hi + 0.5 * dt * k1)
        k3 = rhs(ti + 0.5 * dt, hi + 0.5 * dt * k2)
        k4 = rhs(ti + dt, hi + dt * k3)
        H[i + 1] = hi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return t, H


class TheRunningModel(App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = Heading(
            "Modelling Heart Rate Response", ui_style="margin-bottom: 8px"
        )

        # first model: H_inf is a free parameter
        self.plot = PlotlyComponent(id="plot").on_mounted(self.draw)
        self.H_inf = _number_input("H_inf (bpm)", 150, self.draw)
        self.H_0 = _number_input("H_0 (bpm)", 50, self.draw)
        self.tau = _number_input("tau (min)", 0.5, self.draw)
        self.T_end = _number_input("T_end (min)", 10, self.draw)

        # second model: H_inf is determined by the constant pace
        self.plot2 = PlotlyComponent(id="plot2").on_mounted(self.draw_hinf)
        self.plot3 = PlotlyComponent(id="plot3").on_mounted(self.draw_hinf_pace)
        self.pace = _number_input("pace (min/km)", 4, self._update_second_model)
        self.k_linear = _number_input(
            "k_linear (bpm per km/h)", 6, self._update_second_model
        )
        self.k_sat = _number_input(
            "k_sat (per km/h)", 0.08, self._update_second_model
        )
        self.H_0_2 = _number_input("H_0 (bpm)", 50, self._update_second_model)
        self.H_max = _number_input("H_max (bpm)", 190, self._update_second_model)
        self.tau_2 = _number_input("tau (min)", 0.5, self._update_second_model)
        self.T_end_2 = _number_input("T_end (min)", 10, self._update_second_model)

        # third model: cardiac drift, H_inf grows linearly in time
        self.plot4 = PlotlyComponent(id="plot4").on_mounted(self.draw_drift)
        self.pace_3 = _number_input("pace (min/km)", 4, self.draw_drift)
        self.k_linear_3 = _number_input(
            "k_linear (bpm per km/h)", 6, self.draw_drift
        )
        self.H_0_3 = _number_input("H_0 (bpm)", 50, self.draw_drift)
        self.beta = _number_input("beta (bpm/min)", 0.1, self.draw_drift)
        self.tau_3 = _number_input("tau (min)", 0.5, self.draw_drift)
        self.T_end_3 = _number_input("T_end (min)", 60, self.draw_drift)

        row_style = (
            "width: 100%; max-width: 1100px; gap: 16px;"
            " align-items: flex-start"
        )
        text_style = "width: 100%; max-width: 1100px; text-align: left"
        heading_style = (
            "width: 100%; max-width: 1100px; text-align: left;"
            " margin-top: 40px; margin-bottom: 4px"
        )
        input_col_style = "flex: 0 0 210px"
        plot_col_style = "flex: 1 1 auto; min-width: 0"

        self.component = Centered(
            self.title,
            RichText(_INTRO, ui_style=text_style),
            Heading("A first model", level=3, ui_style=heading_style),
            RichText(_FIRST_MODEL, ui_style=text_style),
            Row(
                Col(
                    self.H_inf,
                    self.H_0,
                    self.tau,
                    self.T_end,
                    ui_style=input_col_style,
                ),
                Col(self.plot, ui_style=plot_col_style),
                ui_style=row_style,
            ),
            Heading(
                "H_inf depends on the pace",
                level=3,
                ui_style=heading_style,
            ),
            RichText(_SECOND_MODEL, ui_style=text_style),
            Row(
                Col(
                    self.pace,
                    self.k_linear,
                    self.k_sat,
                    self.H_0_2,
                    self.H_max,
                    self.tau_2,
                    self.T_end_2,
                    ui_style=input_col_style,
                ),
                Col(self.plot2, self.plot3, ui_style=plot_col_style),
                ui_style=row_style,
            ),
            Heading("Cardiac drift", level=3, ui_style=heading_style),
            RichText(_DRIFT_MODEL, ui_style=text_style),
            Row(
                Col(
                    self.pace_3,
                    self.k_linear_3,
                    self.H_0_3,
                    self.beta,
                    self.tau_3,
                    self.T_end_3,
                    ui_style=input_col_style,
                ),
                Col(self.plot4, ui_style=plot_col_style),
                ui_style=row_style,
            ),
        )

    def draw(self):
        import plotly.graph_objects as go

        H_inf = self.H_inf.ui_model_value
        H_0 = self.H_0.ui_model_value
        tau = self.tau.ui_model_value
        T_end = self.T_end.ui_model_value
        if None in (H_inf, H_0, tau, T_end) or tau <= 0 or T_end <= 0:
            # keep showing the last valid curve while the user is editing
            return

        H_inf_si = bpm_to_hz(float(H_inf))
        H_0_si = bpm_to_hz(float(H_0))
        tau_s = min_to_s(float(tau))
        T_end_s = min_to_s(float(T_end))
        t = np.linspace(0.0, T_end_s, 1000)
        H = H_inf_si - (H_inf_si - H_0_si) * np.exp(-t / tau_s)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s_to_min(t), y=hz_to_bpm(H), mode="lines"))
        fig.update_layout(
            title=(
                f"Heart rate H_inf={hz_to_bpm(H_inf_si):.0f} bpm, "
                f"H_0={hz_to_bpm(H_0_si):.0f} bpm, tau={s_to_min(tau_s):g} min"
            ),
            xaxis_title="t (min)",
            yaxis_title="H (bpm)",
            xaxis={"range": [0.0, float(T_end)]},
        )
        self.plot.draw(fig)

    def draw_hinf(self):
        """Heart-rate response with H_inf = H_0 + k_linear * v."""
        import plotly.graph_objects as go

        pace = self.pace.ui_model_value
        k_linear = self.k_linear.ui_model_value
        k_sat = self.k_sat.ui_model_value
        H_0 = self.H_0_2.ui_model_value
        tau = self.tau_2.ui_model_value
        T_end = self.T_end_2.ui_model_value
        H_max = self.H_max.ui_model_value
        if (
            None in (pace, k_linear, k_sat, H_0, tau, T_end, H_max)
            or pace <= 0
            or tau <= 0
            or T_end <= 0
        ):
            # keep showing the last valid curve while the user is editing
            return

        pace = float(pace)
        H_0_si = bpm_to_hz(float(H_0))
        H_max_si = bpm_to_hz(float(H_max))
        tau_s = min_to_s(float(tau))
        T_end_min = float(T_end)
        v = pace_to_v(pace)
        k_lin_si = k_linear_to_si(float(k_linear))
        k_sat_si = k_sat_to_si(float(k_sat))

        H_inf_si = H_0_si + k_lin_si * v
        H_inf_sat_si = H_0_si + (H_max_si - H_0_si) * (
            1.0 - np.exp(-k_sat_si * v)
        )
        t = np.linspace(0.0, min_to_s(T_end_min), 1000)
        y_lin = H_inf_si - (H_inf_si - H_0_si) * np.exp(-t / tau_s)
        y_sat = H_inf_sat_si - (H_inf_sat_si - H_0_si) * np.exp(-t / tau_s)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=s_to_min(t), y=hz_to_bpm(y_lin), mode="lines", name="linear")
        )
        fig.add_trace(
            go.Scatter(
                x=s_to_min(t), y=hz_to_bpm(y_sat), mode="lines", name="saturating"
            )
        )
        fig.update_layout(
            title=(
                f"Heart rate H_inf={hz_to_bpm(H_inf_si):.0f} bpm, "
                f"pace={_fmt_pace(pace)}, H_0={float(H_0):g} bpm, "
                f"tau={float(tau):g} min"
            ),
            xaxis_title="t (min)",
            yaxis_title="H (bpm)",
            xaxis={"range": [0.0, T_end_min]},
        )
        _add_distance_axis(fig, v, T_end_min)
        self.plot2.draw(fig)

    def _update_second_model(self):
        self.draw_hinf()
        self.draw_hinf_pace()

    def draw_hinf_pace(self):
        """H_inf as a function of pace for both models."""
        import plotly.graph_objects as go

        pace = self.pace.ui_model_value
        k_linear = self.k_linear.ui_model_value
        k_sat = self.k_sat.ui_model_value
        H_0 = self.H_0_2.ui_model_value
        H_max = self.H_max.ui_model_value
        if (
            None in (pace, k_linear, k_sat, H_0, H_max)
            or pace <= 0
        ):
            # keep showing the last valid curve while the user is editing
            return

        pace = float(pace)
        H_0_si = bpm_to_hz(float(H_0))
        H_max_si = bpm_to_hz(float(H_max))
        k_lin_si = k_linear_to_si(float(k_linear))
        k_sat_si = k_sat_to_si(float(k_sat))

        pace_lo, pace_hi = 0.5 * pace, 2.0 * pace
        pp = np.linspace(pace_lo, pace_hi, 300)
        vv = pace_to_v(pp)
        H_lin = hz_to_bpm(H_0_si + k_lin_si * vv)
        H_sat = hz_to_bpm(
            H_0_si + (H_max_si - H_0_si) * (1.0 - np.exp(-k_sat_si * vv))
        )
        hover = (
            "pace %{x:.2f} min/km (%{customdata:.1f} km/h)"
            "<br>H_inf %{y:.0f} bpm<extra></extra>"
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pp,
                y=H_lin,
                mode="lines",
                name="linear",
                customdata=60.0 / pp,
                hovertemplate=hover,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pp,
                y=H_sat,
                mode="lines",
                name="saturating",
                customdata=60.0 / pp,
                hovertemplate=hover,
            )
        )
        fig.update_layout(
            title="H_inf depending on pace",
            xaxis_title="pace (min/km)",
            yaxis_title="H_inf (bpm)",
            xaxis={"range": [pace_lo, pace_hi]},
        )
        _add_speed_axis(fig, pace_lo, pace_hi)
        self.plot3.draw(fig)

    def draw_drift(self):
        """Heart-rate response with a linearly drifting H_inf (cardiac drift)."""
        import plotly.graph_objects as go

        pace = self.pace_3.ui_model_value
        k_linear = self.k_linear_3.ui_model_value
        H_0 = self.H_0_3.ui_model_value
        beta = self.beta.ui_model_value
        tau = self.tau_3.ui_model_value
        T_end = self.T_end_3.ui_model_value
        if (
            None in (pace, k_linear, H_0, beta, tau, T_end)
            or pace <= 0
            or tau <= 0
            or T_end <= 0
        ):
            # keep showing the last valid curve while the user is editing
            return

        pace = float(pace)
        H_0_si = bpm_to_hz(float(H_0))
        beta_si = beta_to_si(float(beta))
        tau_s = min_to_s(float(tau))
        T_end_min = float(T_end)
        v = pace_to_v(pace)
        H_inf_si = H_0_si + k_linear_to_si(float(k_linear)) * v

        t, H = _rk4_drift(H_0_si, H_inf_si, beta_si, tau_s, min_to_s(T_end_min))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=s_to_min(t), y=hz_to_bpm(H), mode="lines", name="H(t)")
        )
        fig.add_trace(
            go.Scatter(
                x=s_to_min(t),
                y=hz_to_bpm(H_inf_si + beta_si * t),
                mode="lines",
                name="H_inf_drift(t)",
                line={"dash": "dash"},
            )
        )
        fig.update_layout(
            title=(
                f"Cardiac drift H_inf={hz_to_bpm(H_inf_si):.0f} bpm, "
                f"beta={float(beta):g} bpm/min, pace={_fmt_pace(pace)}, "
                f"H_0={float(H_0):g} bpm, tau={float(tau):g} min"
            ),
            xaxis_title="t (min)",
            yaxis_title="H (bpm)",
            xaxis={"range": [0.0, T_end_min]},
        )
        _add_distance_axis(fig, v, T_end_min)
        self.plot4.draw(fig)

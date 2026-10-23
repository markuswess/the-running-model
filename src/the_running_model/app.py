import numpy as np
from ngapp.app import App
from ngapp.components import *

from .richtext import RichText


def _number_input(label, value, handler):
    return NumberInput(
        ui_label=label,
        ui_model_value=value,
        ui_debounce=300,
    ).on_update_model_value(handler)


_INTRO = (
    r"We model the heart-rate response $H(t)$ (bpm) with the linear "
    r"first-order ODE "
    r"$$\partial_t H(t) = \tau^{-1}\left(H_\infty(t, v, \dots) - H(t)\right),$$"
    r"where $H_\infty$ (bpm) is the asymptotic heart rate for a given "
    r"velocity $v$ (m/s) and may itself depend on time and other "
    r"parameters; $\tau$ (s) is the time constant of the response."
)

_FIRST_MODEL = (
    r"We start with a simplified case: the runner begins at heart rate "
    r"$H_0$ and runs at a constant pace, so $H_\infty$ is constant. "
    r"The ODE then has the analytic solution "
    r"$$H(t) = H_\infty - (H_\infty - H_0)\,e^{-t/\tau},$$"
    r"i.e. the heart rate approaches $H_\infty$ exponentially with time "
    r"constant $\tau$."
)

_SECOND_MODEL = (
    r"At a constant velocity $v$ the asymptotic heart rate is no longer "
    r"a free parameter but follows from the pace. We compare two simple "
    r"choices for $H_\infty(v)$: a linear relation "
    r"$$H_\infty(v) = H_0 + k_\mathrm{linear}\,v,$$"
    r"and a saturating variant "
    r"$$H_\infty(v) = H_0 + (H_\mathrm{max} - H_0)"
    r"\left(1 - e^{-k_\mathrm{sat}\,v}\right),$$"
    r"where $H_\mathrm{max}$ is the maximal heart rate and "
    r"$k_\mathrm{sat}$ controls how quickly the curve flattens out. "
    r"The upper plot shows the resulting $H(t)$ for both models, the "
    r"lower plot shows $H_\infty(v)$."
)


class TheRunningModel(App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = Heading(
            "Modelling Heart Rate Response", ui_style="margin-bottom: 8px"
        )

        # first model: H_inf is a free parameter
        self.plot = PlotlyComponent(id="plot").on_mounted(self.draw)
        self.H_inf = _number_input("H_inf", 150, self.draw)
        self.H_0 = _number_input("H_0", 50, self.draw)
        self.tau = _number_input("tau", 5, self.draw)
        self.T_end = _number_input("T_end", 100, self.draw)

        # second model: H_inf is determined by the constant velocity v
        self.plot2 = PlotlyComponent(id="plot2").on_mounted(self.draw_hinf)
        self.plot3 = PlotlyComponent(id="plot3").on_mounted(self.draw_hinf_v)
        self.v = _number_input("v", 15, self._update_second_model)
        self.k_linear = _number_input("k_linear", 6, self._update_second_model)
        self.k_sat = _number_input("k_sat", 0.08, self._update_second_model)
        self.H_0_2 = _number_input("H_0", 50, self._update_second_model)
        self.H_max = _number_input("H_max", 190, self._update_second_model)
        self.tau_2 = _number_input("tau", 5, self._update_second_model)
        self.T_end_2 = _number_input("T_end", 100, self._update_second_model)

        row_style = (
            "width: 100%; max-width: 1100px; gap: 16px;"
            " align-items: flex-start"
        )
        text_style = "width: 100%; max-width: 1100px; text-align: left"
        heading_style = (
            "width: 100%; max-width: 1100px; text-align: left;"
            " margin-top: 40px; margin-bottom: 4px"
        )
        input_col_style = "flex: 0 0 180px"
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
                "H_inf depends on the velocity",
                level=3,
                ui_style=heading_style,
            ),
            RichText(_SECOND_MODEL, ui_style=text_style),
            Row(
                Col(
                    self.v,
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

        H_inf = float(H_inf)
        H_0 = float(H_0)
        tau = float(tau)
        T_end = float(T_end)
        nr_points = 1000
        t = np.linspace(0, T_end, int(nr_points))
        y = H_inf - (H_inf - H_0) * np.exp(-t / tau)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines"))
        fig.update_layout(
            title=f"Heart Rate curve H_inf={H_inf}, H_0={H_0}, tau={tau}",
            xaxis_title="t",
            yaxis_title="H",
        )
        self.plot.draw(fig)

    def draw_hinf(self):
        """Heart-rate response with H_inf = H_0 + k_linear * v."""
        import plotly.graph_objects as go

        v = self.v.ui_model_value
        k_linear = self.k_linear.ui_model_value
        k_sat = self.k_sat.ui_model_value
        H_0 = self.H_0_2.ui_model_value
        tau = self.tau_2.ui_model_value
        T_end = self.T_end_2.ui_model_value
        H_max = self.H_max.ui_model_value
        if None in (v, k_linear, H_0, tau, T_end, k_sat, H_max) or tau <= 0 or T_end <= 0:
            # keep showing the last valid curve while the user is editing
            return

        H_inf = H_0 + k_linear * v
        H_inf_sat = H_0 + (H_max - H_0) * (1 - np.exp(-k_sat * v))
        nr_points = 1000
        t = np.linspace(0, T_end, int(nr_points))
        y_lin = H_inf - (H_inf - H_0) * np.exp(-t / tau)
        y_sat = H_inf_sat - (H_inf_sat - H_0) * np.exp(-t / tau)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y_lin, mode="lines", name="linear"))
        fig.add_trace(go.Scatter(x=t, y=y_sat, mode="lines", name="saturating"))
        fig.update_layout(
            title=(
                f"Heart Rate curve H_inf(v)={H_inf:.1f}, v={v},"
                f" H_0={H_0}, tau={tau}"
            ),
            xaxis_title="t",
            yaxis_title="H",
        )
        self.plot2.draw(fig)

    def _update_second_model(self):
        self.draw_hinf()
        self.draw_hinf_v()

    def draw_hinf_v(self):
        """H_inf as a function of the velocity for both models."""
        import plotly.graph_objects as go

        v = self.v.ui_model_value
        k_linear = self.k_linear.ui_model_value
        k_sat = self.k_sat.ui_model_value
        H_0 = self.H_0_2.ui_model_value
        H_max = self.H_max.ui_model_value
        if None in (v, k_linear, k_sat, H_0, H_max):
            # keep showing the last valid curve while the user is editing
            return

        v = float(v)
        k_linear = float(k_linear)
        k_sat = float(k_sat)
        H_0 = float(H_0)
        H_max = float(H_max)
        v_max = max(2.0 * v, 1.0)
        vv = np.linspace(0, v_max, 200)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=vv,
                y=H_0 + k_linear * vv,
                mode="lines",
                name="linear: H_0 + k_linear v",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=vv,
                y=H_0 + (H_max - H_0) * (1 - np.exp(-k_sat * vv)),
                mode="lines",
                name="saturating: H_0 + (H_max-H_0)(1-e^(-k_sat v))",
            )
        )
        fig.update_layout(
            title="H_inf depending on v",
            xaxis_title="v",
            yaxis_title="H_inf",
        )
        self.plot3.draw(fig)

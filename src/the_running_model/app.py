import numpy as np
from ngapp.app import App
from ngapp.components import *


class TheRunningModel(App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = Heading("Modelling HR response")
        self.plot = PlotlyComponent(id="plot").on_mounted(self.draw)
        self.HR_inf = NumberInput(
            ui_label="HR_inf",
            ui_model_value=150,
            ui_debounce=300,
        ).on_update_model_value(self.draw)
        self.HR_0 = NumberInput(
            ui_label="HR_0",
            ui_model_value=60,
            ui_debounce=300,
        ).on_update_model_value(self.draw)
        self.tau = NumberInput(
            ui_label="tau",
            ui_model_value=5,
            ui_debounce=300,
        ).on_update_model_value(self.draw)
        self.T_end = NumberInput(
            ui_label="T_end",
            ui_model_value=100,
            ui_debounce=300,
        ).on_update_model_value(self.draw)
        self.component = Centered(
            self.title,
            Row(Heading("A first model", level=3)),
            Row(
                Col(
                    self.HR_inf,
                    self.HR_0,
                    self.tau,
                    self.T_end,
                    ui_style="flex: 0 0 180px",
                ),
                Col(self.plot, ui_style="flex: 1 1 auto; min-width: 0"),
                ui_style=(
                    "width: 100%; max-width: 1100px; gap: 16px;"
                    " align-items: flex-start"
                ),
            ),
            Row("Assuming constant pace, at which the runner runs with HR_inf"),
        )

    def draw(self):
        import plotly.graph_objects as go

        HR_inf = self.HR_inf.ui_model_value
        HR_0 = self.HR_0.ui_model_value
        tau = self.tau.ui_model_value
        T_end = self.T_end.ui_model_value
        if None in (HR_inf, HR_0, tau, T_end) or tau <= 0 or T_end <= 0:
            # keep showing the last valid curve while the user is editing
            return

        HR_inf = float(HR_inf)
        HR_0 = float(HR_0)
        tau = float(tau)
        T_end = float(T_end)
        nr_points = 1000
        t = np.linspace(0, T_end, int(nr_points))
        y = HR_inf - (HR_inf - HR_0) * np.exp(-t / tau)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines"))
        fig.update_layout(
            title=f"Heart Rate curve HR_inf={HR_inf}, HR_0={HR_0}, tau={tau}",
            xaxis_title="t",
            yaxis_title="HR",
        )
        self.plot.draw(fig)

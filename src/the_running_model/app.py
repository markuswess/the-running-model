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
        ).on_update_model_value(self.draw)
        self.HR_0 = NumberInput(
            ui_label="HR_0",
            ui_model_value=60,
        ).on_update_model_value(self.draw)
        self.tau = NumberInput(
            ui_label="tau",
            ui_model_value=5,
        ).on_update_model_value(self.draw)
        self.T_end = NumberInput(
            ui_label="T_end",
            ui_model_value=100,
        ).on_update_model_value(self.draw)
        self.component = Centered(
            self.title,
            Row(Heading("A first model", level=3)),
            Row(
                Col(self.HR_inf, self.HR_0, self.tau, self.T_end),
                Col(self.plot)
                ),
            Row("Assuming constant pace, at which the runner runs with HR_inf"),
        )

    def draw(self):
        import plotly.graph_objects as go

        HR_inf = float(self.HR_inf.ui_model_value)
        HR_0 = float(self.HR_0.ui_model_value)
        tau = float(self.tau.ui_model_value)
        T_end = float(self.T_end.ui_model_value)
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

"""Rich text with LaTeX math, rendered in the browser via KaTeX.

Use ``$...$`` for inline math and ``$$...$$`` for display math, e.g.::

    RichText(r"the time constant is $\tau$")

KaTeX (CSS + JS + auto-render) is loaded once from a CDN and shared by all
instances, following the same pattern as ngapp's ``PlotlyComponent``.
"""

from ngapp.components import Component

_KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
]


class RichText(Component):
    """A div that renders its text with inline and display LaTeX math."""

    _state = "none"  # "none" | "loading" | "ready"
    _waiting: list = []

    def __init__(self, text: str, **kwargs):
        kwargs.setdefault("ui_style", "max-width: 1100px; text-align: center")
        super().__init__("div", **kwargs)
        self._text = text
        self._onload_proxy = None
        self.on_mounted(self._setup)

    def _setup(self):
        self.call_js(self._setup_js)

    def _setup_js(self, js):
        el = self._js_component
        el.textContent = self._text
        cls = RichText
        cls._waiting.append(self)
        if cls._state == "ready":
            self._typeset()
            return
        if cls._state == "loading":
            return
        cls._state = "loading"
        link = js.document.createElement("link")
        link.rel = "stylesheet"
        link.href = f"{_KATEX}/katex.min.css"
        js.document.head.appendChild(link)
        self._load_script(js, f"{_KATEX}/katex.min.js", self._on_katex_loaded)

    def _load_script(self, js, src, onload):
        import webgpu.platform as pl

        self._onload_proxy = pl.create_proxy(onload)
        script = js.document.createElement("script")
        script.src = src
        script.onload = self._onload_proxy
        js.document.head.appendChild(script)

    def _on_katex_loaded(self, *_args):
        self.call_js(self._load_autorender)

    def _load_autorender(self, js):
        self._load_script(
            js, f"{_KATEX}/contrib/auto-render.min.js", self._on_ready
        )

    def _on_ready(self, *_args):
        cls = RichText
        cls._state = "ready"
        waiting, cls._waiting = cls._waiting, []
        for comp in waiting:
            comp._typeset()

    def _typeset(self):
        self.call_js(self._typeset_js)

    def _typeset_js(self, js):
        js.window.renderMathInElement(
            self._js_component,
            {"delimiters": _DELIMITERS, "throwOnError": False},
        )

from ngapp import AppConfig

from . import TheRunningModel, __version__

_DESCRIPTION = "Interactive models of endurance/running performance"

config = AppConfig(
    name="the running model",
    version=__version__,
    python_class=TheRunningModel,
    frontend_pip_dependencies=["plotly>=7.1.0"],
    description=_DESCRIPTION,
)

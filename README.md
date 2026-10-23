# the-running-model

Interactive models of endurance/running performance, built as an
[ngapp](https://pypi.org/project/ngapp/) app with Plotly.

## Run

```bash
uv run the-running-model            # local dev server
uv run python -m the_running_model  # same thing via the module
```

Build the static site (as the GitHub Pages workflow does):

```bash
uv run python -m ngapp.cli.build_website -a the_running_model.appconfig
```

## Layout

- `src/the_running_model/app.py` — the app (`TheRunningModel`)
- `src/the_running_model/appconfig.py` — app metadata used by ngapp
- `.github/workflows/deploy.yml` — builds and deploys the static app to GitHub Pages

## The model so far

Heart rate rises exponentially from `HR_0` towards `HR_inf` with time constant `tau`
(`HR(t) = HR_inf - (HR_inf - HR_0) * exp(-t / tau)`), assuming a constant pace at
which the runner would eventually settle at `HR_inf`.

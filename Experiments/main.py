import sys

import typer
from experiments_package.experiments import (
    Autoregressive,
    MultiOutputWithoutWeather,
    MultiStep,
    MultiStepWithoutWeather,
    SingleOutputWithoutWeather,
    SingleStepMultiOutput,
    SingleStepSingleOutput,
)
from experiments_package.general import combine_results

OUTPUT_PATH = "./outputs"

app = typer.Typer()
experiments = [
    SingleStepSingleOutput("SingleOutput", OUTPUT_PATH),
    SingleStepMultiOutput("MultiOutput", OUTPUT_PATH),
    MultiStep("MultiStep", OUTPUT_PATH),
    Autoregressive("Autoregressive", OUTPUT_PATH),
    MultiOutputWithoutWeather("MultiOutput-NoWeather", OUTPUT_PATH),
    MultiStepWithoutWeather("MultiStep-NoWeather", OUTPUT_PATH),
    SingleOutputWithoutWeather("SingleOutput-NoWeather", OUTPUT_PATH),
]


def get_experiment(exp: str):
    if exp not in [ex.name for ex in experiments]:
        typer.echo(
            f"Allowed Values for experiment are: {str([ex.name for ex in experiments])}"
        )
        sys.exit(-1)

    return next((ex for ex in experiments if ex.name == exp))


@app.command("get-experiments")
def show_experiments():
    """Show the registered Experiments"""
    typer.echo("Following experiments are registered:")
    for exp in experiments:
        typer.echo(f"   - {exp.name}")


@app.command("get-models")
def show_models(exp: str):
    """Show the models available in a certain experiment"""
    experiment = get_experiment(exp)
    models = experiment.get_models().keys()
    typer.echo(f"Following Models are registered in Experiment <<{experiment.name}>> :")
    for model in models:
        typer.echo(f"   - {model}")


@app.command("run")
def run_model(exp: str, model: str):
    """Run a perticular model of an experiment"""
    experiment = get_experiment(exp)
    models = experiment.get_models().keys()
    if not model in models:
        typer.echo(f"Model not existent, allowed Values: {str(models)}")

    experiment.run_model(model)


@app.command("experiment")
def run_experiment(
    exp: str,
    run_all: bool = False,
):
    """Run an experiment with all registered models."""
    if run_all:
        typer.echo("Run all experiments")
        for experiment in experiments:
            experiment.run()
    else:
        experiment = get_experiment(exp)
        typer.echo(f"Run {experiment.name}")
        experiment.run()


@app.command("combine-results")
def combine():
    combine_results(OUTPUT_PATH)


if __name__ == "__main__":
    app()

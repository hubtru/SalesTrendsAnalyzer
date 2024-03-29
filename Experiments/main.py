import sys

import typer

from experiments_package.experiments import (
    MultiOutputWithoutWeather,
    MultiStep,
    MultiStepWithoutWeather,
    SingleOutputWithoutWeather,
    SingleStepMultiOutput,
    SingleStepSingleOutput,
    SingleOutputNoIds,
    SingleOutput7Days
)
from experiments_package.general import combine_results, Experiment

OUTPUT_PATH = "./outputs"

app = typer.Typer()
experiments: [Experiment] = [
    SingleStepSingleOutput("SingleOutput", OUTPUT_PATH),
    SingleStepMultiOutput("MultiOutput", OUTPUT_PATH),
    MultiStep("MultiStep", OUTPUT_PATH),
    # Autoregressive("Autoregressive", OUTPUT_PATH),
    MultiOutputWithoutWeather("MultiOutput-NoWeather", OUTPUT_PATH),
    MultiStepWithoutWeather("MultiStep-NoWeather", OUTPUT_PATH),
    SingleOutputWithoutWeather("SingleOutput-NoWeather", OUTPUT_PATH),
    SingleOutputNoIds("SingleOutput-NoIds", OUTPUT_PATH),
    SingleOutput7Days("SingleOutput-7Days", OUTPUT_PATH),
]


def get_experiment(exp: str) -> Experiment:
    if exp not in [ex.name for ex in experiments]:
        typer.echo(
            f"Allowed Values for experiment are: {str([ex.name for ex in experiments])}"
        )
        sys.exit(0)

    return next((ex for ex in experiments if ex.name == exp))


def check_model_existence(experiment: Experiment, model: str) -> bool:
    models = experiment.get_models()
    if model not in models.keys():
        typer.echo(f"Model not existent, allowed Values: {str(models)}")
        sys.exit(-1)
    else:
        return True


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
def run_model(exp: str, model: str = None):
    """Run an experiment with all registered models.
     -- model: only choose one model to run
    """

    experiment = get_experiment(exp)
    if model is None:
        typer.echo(f"Run {experiment.name}")
        experiment.run()
    else:
        check_model_existence(experiment, model)
        experiment.run_model(model)


@app.command("run-all")
def run_experiment(
):
    """Run all experiments."""
    typer.echo("Run all experiments")
    for experiment in experiments:
        experiment.run()


@app.command("combine-results")
def combine():
    """Combine all available results into one output table."""
    combine_results(OUTPUT_PATH)


@app.command("output-graph")
def output_graph(exp: str, model: str):
    """Draw the picture of the output graph of a model and experiment"""
    experiment = get_experiment(exp)
    check_model_existence(experiment, model)
    experiment.run_model(model, output_all_label_images=True)


@app.command("output-graph-all")
def output_graph():
    """Draw the picture of the output graph of all models in all experiments"""
    typer.echo("Run all experiments")
    for experiment in experiments:
        for model in experiment.get_models().keys():
            experiment.run_model(model, output_all_label_images=True)


if __name__ == "__main__":
    app()

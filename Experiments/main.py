import typer
from sys import exit
from typing_extensions import Annotated
from experiments_package.experiments import (
    SingleStepSingleOutput,
    SingleStepMultiOutput,
    MultiStep,
    Autoregressive,
)

OUTPUT_PATH = "./outputs"

app = typer.Typer()
experiments = [
    SingleStepSingleOutput("SingleOutput", OUTPUT_PATH),
    SingleStepMultiOutput("MultiOutput", OUTPUT_PATH),
    MultiStep("MultiStep", OUTPUT_PATH),
    Autoregressive("Autoregressive", OUTPUT_PATH),
]


@app.command()
def run_experiment(
    exp: Annotated[
        str,
        typer.Option(
            help="The Experiment to run.",
            autocompletion=lambda: [ex.name for ex in experiments],
        ),
    ] = None,
    all: bool = False,
):
    """Run an experiment with the specified models."""
    if exp:
        if exp not in [ex.name for ex in experiments]:
            typer.echo(
                f"Allowed Values for experiment are: {str([ex.name for ex in experiments])}"
            )
            exit(-1)

        experiment = next((ex for ex in experiments if ex.name == exp))
        typer.echo(f"Run {experiment.name}")
        experiment.run()
    elif all:
        typer.echo("Run all experiments")
        for experiment in experiments:
            experiment.run()


if __name__ == "__main__":
    app()

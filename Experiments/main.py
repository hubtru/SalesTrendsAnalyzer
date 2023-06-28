import typer
from typing_extensions import Annotated
from experiments_package.experiments import Experiment1, Experiment2


app = typer.Typer()
experiments = [
    Experiment1("SingleOutput", "./outputs"),
    Experiment2("MultiOutput", "./outputs"),
]


@app.command()
def run_experiment(
    exp: Annotated[
        str,
        typer.Option(
            help="The Experiment to run.",
            autocompletion=lambda: [ex.name for ex in experiments],
        ),
    ],
):
    """Run an experiment with the specified models."""

    experiment = next((ex for ex in experiments if ex.name == exp))
    typer.echo(f"Run {experiment.name}")
    experiment.run()


if __name__ == "__main__":
    app()

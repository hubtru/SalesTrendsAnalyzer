import experiments_package.models.single_step as single_step
from experiments_package.experiments.experiment1 import Experiment1
from experiments_package.general.config import ProductIds

if __name__ == "__main__":
    experiment = Experiment1("Experiment1", "./outputs")
    experiment.run(
        {
            "Persistence": single_step.Baseline(
                label_index=experiment.data.column_indices[
                    ProductIds.BENS_LUNCHTIME.value
                ],
                forecasting_width=1,
            ),
            "Linear": single_step.Linear(
                window_width=3, label_width=1, feature_size=87
            ),
            "Dense": single_step.Dense(window_width=3, label_width=1, feature_size=87),
            "Multi Step Dense": single_step.MultiStepDense(
                window_width=3, label_width=1, feature_size=87
            ),
            "Convolutional": single_step.Convolutional(
                window_width=3, label_width=1, feature_size=87
            ),
            "RNN": single_step.RNN(window_width=3, label_width=1, feature_size=87),
        }
    )

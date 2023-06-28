from experiments_package.models.Baseline import Baseline
from experiments_package.models.Linear import Linear
from experiments_package.experiments.experiment1 import Experiment1
from experiments_package.general.config import ProductIds

if __name__ == "__main__":
    experiment = Experiment1("Experiment1", "./outputs")
    experiment.run(
        {
            "Persistence": Baseline(
                label_index=experiment.data.column_indices[
                    ProductIds.BENS_LUNCHTIME.value
                ],
                forecasting_width=1,
            ),
            "Linear": Linear(window_width=3, label_width=1, feature_size=87),
        }
    )

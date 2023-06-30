import tensorflow as tf
from experiments_package.general import DatasetOptions, Experiment, ProductIds
import experiments_package.models.single_step as single_step
from experiments_package.models import Baseline


class SingleStepSingleOutput(Experiment):
    def compile_and_fit(self, model):
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )

        history = model.fit(
            self.data.train, epochs=100, validation_data=self.data.val, verbose=0
        )

        return history

    def get_train_settings(self):
        return {
            "Batch Size": 32,
            "Epochs": 100,
            "Learning Rate": 0.001,
            "Optimizer": "Adam",
            "Shuffled Batches": "True",
            "Early Stopping": "False",
            "Window Width": 3,
            "Label Width": 1,
            "Shift": 1,
            "Predicted Labels": ProductIds.BENS_LUNCHTIME.value,
        }

    def get_dataset_options(self):
        return DatasetOptions(
            data_origin="./../Data/merged_cleaned_FE_imputed(v)_w.csv",
            drop_columns=[
                "WeekoftheYear_cos",
                "WeekoftheYear_sin",
                "DayoftheYear_cos",
                "DayoftheYear_sin",
                "Month_cos",
                "Month_sin",
                "Season_Autumn",
                "Season_Spring",
                "Season_Summer",
                "Season_Winter",
            ],
            window_width=3,
            label_width=1,
            shift=1,
            label_columns=[ProductIds.BENS_LUNCHTIME.value],
        )

    def get_models(self):
        return {
            "Persistence": Baseline(
                label_index=self.data.column_indices[ProductIds.BENS_LUNCHTIME.value],
                label_width=1,
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

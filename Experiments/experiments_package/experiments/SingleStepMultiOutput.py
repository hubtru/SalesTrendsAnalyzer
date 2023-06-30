import tensorflow as tf
from experiments_package.general import DatasetOptions, Experiment, ProductIds
from experiments_package.models.single_step import Dense, RNN, ResidualRNN
from experiments_package.models import Baseline


class SingleStepMultiOutput(Experiment):
    def compile_and_fit(self, model):
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )

        history = model.fit(
            self.data.train,
            epochs=100,
            validation_data=self.data.val,
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
            "Predicted Labels": "All",
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
            label_columns=None,
        )

    def get_models(self):
        return {
            "Baseline": Baseline(label_index=None, forecasting_width=1),
            "Dense": Dense(
                window_width=3, label_width=1, feature_size=87, num_labels=87
            ),
            "RNN": RNN(window_width=3, label_width=1, feature_size=87, num_labels=87),
            "ResidualRNN": ResidualRNN(
                window_width=3, label_width=1, feature_size=87, num_labels=87
            ),
        }

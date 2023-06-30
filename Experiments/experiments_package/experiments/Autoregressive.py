import tensorflow as tf
from experiments_package.general import DatasetOptions, Experiment
from experiments_package.models.autoregressive import FeedBack


class Autoregressive(Experiment):
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
            "Window Width": 10,
            "Label Width": 5,
            "Shift": 5,
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
            window_width=10,
            label_width=5,
            shift=5,
            label_columns=None,
        )

    def get_models(self):
        return {
            "FeedBack": FeedBack(units=32, label_width=5, num_labels=87),
        }

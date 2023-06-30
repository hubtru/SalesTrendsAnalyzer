import tensorflow as tf
from experiments_package.general import DatasetOptions, Experiment
from experiments_package.models import Baseline
from experiments_package.models.multi_step import (
    MultiConvolution,
    MultiDense,
    MultiLinear,
    MultiLSTM,
)


class MultiStep(Experiment):
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
            "RepeatBaseline": Baseline(label_index=None, label_width=5),
            "MultiStepLastBasline": Baseline(
                label_index=None, label_width=5, repeat_last=True
            ),
            "Multi Convolution": MultiConvolution(
                conv_width=3, label_width=5, num_labels=87
            ),
            "Multi Dense": MultiDense(
                label_width=5, num_labels=87, num_features=87, window_width=10
            ),
            "Mutli Linear": MultiLinear(label_width=5, num_labels=87),
            "Multi LSTM": MultiLSTM(label_width=5, num_labels=87),
        }

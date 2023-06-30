import tensorflow as tf
from experiments_package.general import DatasetOptions, Experiment
from experiments_package.models import Baseline
from experiments_package.models.multi_step import (
    MultiConvolution,
    MultiDense,
    MultiLinear,
    MultiLSTM,
)


class MultiStepWithoutWeather(Experiment):
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
                "apparent_temperature_mean",
                "precipitation_sum",
                "shortwave_radiation_sum",
                "windgusts_10m_max",
                "temperature_2m_min",
                "weather_unchanged sky state",
                "weather_clouds dissolving",
                "IsSnowfall",
                "temperature_2m_mean",
                "temperature_2m_max",
                "evapotranspiration_res",
                "windspeed_10m_max",
                "weather_clouds developing",
                "apparent_temperature_min",
                "weather_no clouds developing",
                "precipitation_hours",
                "weather_precipitation",
                "apparent_temperature_max",
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
                conv_width=3, label_width=5, num_labels=69
            ),
            "Multi Dense": MultiDense(
                label_width=5, num_labels=69, num_features=69, window_width=10
            ),
            "Mutli Linear": MultiLinear(label_width=5, num_labels=69),
            "Multi LSTM": MultiLSTM(label_width=5, num_labels=69),
        }

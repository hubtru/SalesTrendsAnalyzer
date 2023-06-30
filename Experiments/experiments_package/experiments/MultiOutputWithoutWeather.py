import tensorflow as tf
from experiments_package.general import DatasetOptions, Experiment
from experiments_package.models.single_step import Dense, RNN, ResidualRNN
from experiments_package.models import Baseline


class MultiOutputWithoutWeather(Experiment):
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
            window_width=3,
            label_width=1,
            shift=1,
            label_columns=None,
        )

    def get_models(self):
        return {
            "Baseline": Baseline(label_index=None, label_width=1),
            "Dense": Dense(
                window_width=3, label_width=1, feature_size=69, num_labels=69
            ),
            "RNN": RNN(window_width=3, label_width=1, feature_size=69, num_labels=69),
            "ResidualRNN": ResidualRNN(
                window_width=3, label_width=1, feature_size=69, num_labels=69
            ),
        }

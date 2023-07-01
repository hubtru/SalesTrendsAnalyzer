import experiments_package.models.single_step as single_step
import tensorflow as tf
from experiments_package.general import DatasetOptions, Experiment, ProductIds
from experiments_package.models import Baseline


class SingleOutput7Days(Experiment):
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
            "Window Width": 7,
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
                "4260705920003",
                "4260705920010",
                "4260705920027",
                "4260705920034",
                "4260705920041",
                "4260705920058",
                "4260705920065",
                "4260705920072",
                "4260705920089",
                "4260705920096",
                "4260705920102",
                "4260705920119",
                "4260705920126",
                "4260705920133",
                "4260705920140",
                "4260705920157",
                "4260705920164",
                "4260705920171",
                "4260705920188",
                "4260705920195",
                "4260705920201",
                "4260705920218",
                "4260705920225",
                "4260705920232",
                "4260705920249",
                "4260705920256",
                "4260705920263",
                "4260705920270",
                "4260705920287",
                "4260705920300",
                "4260705920317",
                "4260705920324",
                "4260705920331",
                "4260705920355",
                "4260705920362",
                "4260705920393",
                "4260705920409",
                "4260705920416",
                "4260705920423",
                "4260705920430",
                "4260705920461",
                "4260705920478",
                "4260705920492",
                "4260705920508",
                "4260705920515",
                "4260705920522",
                "4260705920539",
                "4260705920546",
                "4260705920553",
                "4260705920560",
                "4260705920577",
                "4260705920584",
                "4260705920591",
                "4260705920607",
                "4260705920638",
            ],
            window_width=7,
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
                window_width=7, label_width=1, feature_size=32
            ),
            "Dense": single_step.Dense(window_width=7, label_width=1, feature_size=32),
            "Multi Step Dense": single_step.MultiStepDense(
                window_width=7, label_width=1, feature_size=32
            ),
            "Convolutional": single_step.Convolutional(
                window_width=7, label_width=1, feature_size=32
            ),
            "RNN": single_step.RNN(window_width=7, label_width=1, feature_size=32),
        }

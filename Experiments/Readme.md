# Experiments

> Subpackage used for carrying out different experiments streamlined to compare each other.

<br>

## Directory Structure

- **experiments_package**
    - package for the code of the experiments
    - **experiments**
        - place for all experiments that are defined
    - **models**
        - place where all models are located and can be reused in different experiments
    - **general**
        - folder for the setup of experiments and generating results
- **outputs**

    - place to save results of the experiments

- **main.py**
    - Script to run experiments

<br>

# Usage

```shell
python main.py --help
```

e.g.:

```shell
python main.py run <experiment> <model>
```

```shell
python main.py experiment --exp SingleOutput
```

Following things have to be considered while using:

> Running a single model will generate outcomes for this model alone, not touching any other results of experiments.

> Running a whole experiment generates outputs and results for that experiment.

> To produce the "all_results"-csv-file for documentation one has to run the combine-results command. Repeat this step
> after each new experiment or model run in order to update.
> !! Only outputs of running whole experiments will be included in the all-results-file
>
> If there are several data-points for the same experiment and model, they will all be added to the final result data.

<br>

# How to add a new experiment

- Define your models in the **models** directory
- create a new experiment-file in **experiments**
- instantiate a subclass of **general.Experiment**
- fill the abstract methods below
- Register the experiment in "experiments" array of the "main.py" file in main directory

```python
@abstractmethod
def get_dataset_options(self) -> general.DatasetOptions:
    """
    Return the options of for the dataset.

    e.g.:
    """
    return DatasetOptions(
        data_origin="./../Data/xxx.csv",
        drop_columns=["WeekoftheYear_cos"],
        window_width=3,
        label_width=1,
        shift=1,
        label_columns=["15720456723"],
    )


@abstractmethod
def compile_and_fit(self, model) -> History:
    """
    Return the history of the fit-function.
    This method shall includes the compile and the fit function action on the model passed in.
    Use the settings and hyperparameters you want to use for that experiment.

    - Please use verbose=0 in the fit function to declutter output.
    """


@abstractmethod
def get_train_settings(self) -> Dict[str, Any]:
    """
    Returns information about the information that are specific for this algorithm.
    They will be shown in the results table of the experiment.

    Mapping from:
    - (key) => (value) === (parameter name) => (parameter value)

    e.g.:
    """
    return {
        "Batch Size": 32,
        "epochs": 100,
    }

```

> See the DatasetOptions in general.config:

```python
@dataclass
class DatasetOptions:
    window_width: int
    label_width: int
    shift: int
    data_origin: str
    drop_columns: List[str]
    label_columns: Optional[List[str]] = None
```

## How are the weekly sushi savings calculated?

- Take all product Ids which are in the values to predict
- Slide input window over whole dataset, from every output, take the chronologically first prediction and compare it
  with the real value of that day.
- This means that analysis range depends on the shift value used in the window generator.
- Then for every product type it is looked up how much sushi are in that package and the values are scaled to use total
  sushi's wasted/saved
- Every Day gets rounded to the closest integer as only full packages can be produced
- wasted sushi is the number of sushi that was produced too much compared to actual demand
- not_enough sushi is the number of sushi that was produced less than the amount actually sold on that day
- all the values will get add up for all the products predicted, then they will get divided by the number of days (times
  the number of products) to get the average per day for all products and all times

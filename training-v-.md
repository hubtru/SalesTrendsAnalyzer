# training-v-
# Table of Contents
- Introduction
- Installation
- Usage
- Data
- Overview
- Data structure
- Results
- Contributing
## Introduction
The given code is a machine learning model for time series forecasting using various techniques such as Fast Fourier Transform, normalization, data windowing, and splitting. It also includes the creation of TensorFlow datasets and the implementation of different models such as single-step models, multi-output models, multi-step models, and autoregressive models. The performance of each model is evaluated and compared to the baseline model. The code is written in Python and uses TensorFlow and other libraries for data processing and modeling. By using vertical training, the model can provide more targeted and accurate forecasts for a single product, which can be useful for businesses that want to understand the performance of a specific product and make data-driven decisions.

- Loading the data: This involves importing the necessary libraries and loading the dataset that will be used for analysis and prediction.

- Exploratory Data Analysis (EDA): This step involves examining the dataset to understand its features, distribution, and patterns.

- Fast Fourier Transform (FFT): This is a mathematical technique used to transform time-domain data into the frequency-domain.

- Splitting the Data: This involves splitting the data into training, validation, and test sets.

- Normalization: This step involves scaling the data to a smaller range of values to ensure the model can learn effectively.

- Data Windowing & Splitting: This step involves creating windows of data to train and test the model.

- Creating TensorFlow dataset: This involves converting the data into a TensorFlow dataset format to prepare it for model training.

- Single Step Models: This includes various models designed to predict a single output in the future based on a single input.

- Multi-output: This involves designing models that can predict multiple outputs in the future based on a single input.

- Multi-Step Models: This includes models designed to predict multiple outputs in the future based on multiple inputs.

- Baseline: This is the simplest model that can be used as a benchmark for comparing other models.

- Linear Model: This is a type of regression model that assumes a linear relationship between the input and output variables.

- Dense Model: This is a type of neural network model that consists of fully connected layers.

- Convolutional Model: This is a type of neural network model that is commonly used for image processing but can also be used for time series analysis.

- RNN Model: This is a type of neural network model that is specifically designed for sequential data such as time series.

- Performance: This involves evaluating the performance of the different models using various metrics such as mean squared error, mean absolute error, and root mean squared error.

- Autoregressive Models: This includes models designed to predict future values based on the past values of the same variable.
- Vertical Training: Vertical training, also known as single product analysis, involves analyzing data for a specific product, or a subset of products, by focusing on their unique features and values. This approach is useful when you want to understand the performance of a specific product, identify trends or patterns, and make targeted improvements.
For example, let's say you want to analyze the sales performance of a single product in a given period. By using vertical training, you would focus on the specific features of that product, such as its price, promotions, and advertising strategies, and how they affect its sales performance.

**Inputs**: 

- merged_cleaned_FE_imputed(v).csv? 

**Outputs**: 
- The trained models mentioned in Introduction ...? 


## Installation
To use this project, you will need to install the following packages:
- os
- datetime
- IPython
- IPython.display
- matplotlib 
- matplotlib.pyplot 
- numpy 
- pandas 
- seaborn 
- tensorflow 
## Usage

To use this project, you need to download the repository and run the training(v).ipynb file. The file contains code to perform time series analysis on a dataset of daily sales for sushi dishes. It includes steps for loading the data, performing exploratory data analysis, Fast Fourier Transform (FFT) analysis, splitting the data, normalizing the data, windowing the data, and creating a TensorFlow dataset.

The notebook also contains several models for time series forecasting, including single-step models (baseline, linear, dense, convolutional, and RNN), multi-output models, multi-step models (baseline, single-shot, and autoregressive), and performance evaluation metrics. The analysis focuses on daily/weekly sales for each dish.


## Data
### Overview

The data used in this project is sales data for a sushi restaurant over a period of time. The data contains information about the products sold, the date of sale, and the quantity sold.
### Data structure
The merged_cleaned_FE_imputed(v).csv file has the following columns:
- Date: The date of the day for which the data is recorded.
- Month: The month of the year for which the data is recorded.
- DayoftheMonth: The day of the month for which the data is recorded.
- WeekoftheMonth: The week of the month for which the data is recorded.
- DayoftheWeek: The day of the week for which the data is recorded.
- WeekoftheYear: The week of the year for which the data is recorded.
- DayoftheYear: The day of the year for which the data is recorded.
- isWeekend: A binary column that indicates whether the day is a weekend or not.
- isWeekStart: A binary column that indicates whether the day is the start of a week or not.
- isWeekEnd: A binary column that indicates whether the day is the end of a week or not.
- isMonthStart: A binary column that indicates whether the day is the start of a month or not.
- isMonthEnd: A binary column that indicates whether the day is the end of a month or not.
- Season_Autumn, Season_Spring, Season_Summer, Season_Winter: Binary columns that indicate the season for which the data is recorded.
- isHoliday: A binary column that indicates whether the day is a holiday or not.
- StoreID: The ID of the store for which the data is recorded.
- 4260705920003 to 4260705920638: These columns represent the ProductID of different products sold at the store. The values in these columns indicate the quantity of each product sold on the given day.
- total_quantity_day: The total quantity of products sold at the store on the given day.

## Contributing
Contributions to this project are welcome. Please open an issue or pull request if you have any suggestions or would like to contribute code.

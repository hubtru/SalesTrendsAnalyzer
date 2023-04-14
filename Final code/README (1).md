# Table of Contents
- Introduction
- feature_engineering
- data_separator
- data_Loader
##  Introduction
This repository contains three Python scripts that provide useful functionality for data analysis and processing. These scripts are designed to perform specific tasks related to data processing and cleaning, and are intended to be used as building blocks for more complex data analysis workflows.


### Script:   `data_separator.py`

**Description**: This Python script performs data manipulation tasks using the Pandas library. It groups data by StoreID and ProductID, calculates the sum of Quantity for each group, and saves the resulting data into separate CSV files for each StoreID and ProductID.

**Inputs**:

- Data/merged_cleaned.csv: Path to the merged and cleaned data file (CSV format).
- Data/product_mapping.csv: Path to the product mapping data file (CSV format).

**Outputs**: 

- Grouped data: Separate CSV files are generated for each StoreID and ProductID and saved in the specified output directories Data/StoreID_StoreID.csv and Data/ProductID_ProductID.csv respectively.

**Usage**: 
```bash
python  data_separator.py
```

This code is a Python script that performs some data manipulation tasks using the Pandas library.

The groupby_storeid function takes a dataset as input and groups the data by StoreID, sorts the data by Date, drops the StoreID column, and saves the resulting grouped data into separate CSV files for each StoreID.

The weekly_sum function takes a list of store IDs as input, reads the data for each store, groups the data by ProductID and WeekoftheYear, calculates the sum of Quantity for each group, and saves the resulting data into separate CSV files for each StoreID.

The groupby_week function takes a list of store IDs as input, reads the data for each store, groups the data by ProductID and EAN, sorts the data by WeekoftheYear, and saves the resulting data into separate CSV files for each ProductID.

The if __name__ == '__main__': block reads in a merged and cleaned dataset from a CSV file, reads in a product mapping CSV file, creates a dictionary mapping EAN codes to product numbers, gets a list of unique store IDs from the merged dataset, and calls the groupby_storeid, weekly_sum, and groupby_week functions with the list of store IDs as input.

Overall, this code appears to be performing some data processing and transformation tasks on a dataset related to retail stores and products

### Script:  `data_Loader.py`

**Description**: This script converts raw data files to CSV format and cleans the data for further analysis.

**Inputs**:

- folder_path: A string representing the path of the folder containing the raw data files (in a non-CSV format). Default name is: `Data/Raw`
- destination_path: A string representing the path where the converted CSV files will be saved.
- column_names: A list of strings representing the column names of the resulting CSV files.


**Outputs**: 

- Concatenated CSV file: A CSV file containing all the converted raw data files concatenated into a single Pandas DataFrame and saved in the specified output directory (destination_path). Default name is: `Data/merged_raw.csv`.  
- Cleaned CSV file: A cleaned CSV file (with name 'merged_cleaned.csv') containing the cleaned data, saved in the Data directory.

**Usage**: 

```bash
python   data_Loader.py
```

**Functions**:

- convert_to_csv(folder_path, destination_path, column_names): Converts raw data files to CSV format and concatenates them into a single Pandas DataFrame.
- clean_data(df): Cleans the data in the Pandas DataFrame by converting columns to appropriate data types, sorting the DataFrame, and dropping rows with missing values or duplicates.


This code has two main functions: convert_to_csv and clean_data.

convert_to_csv function takes in three arguments:

folder_path - a string that represents the path of the folder containing the raw data files
destination_path - a string that represents the path where the converted CSV files will be saved
column_names - a list of strings that represents the column names of the resulting CSV files
This function reads each raw data file in the folder_path directory, converts it to a Pandas DataFrame, and saves it as a CSV file in the destination_path directory. The resulting CSV files have the same name as the original files, but with .csv as the extension. The function returns a Pandas DataFrame that is the concatenation of all the CSV files in the destination_path directory.

clean_data function takes in a Pandas DataFrame argument:

- df - a Pandas DataFrame that contains the data to be cleaned

This function cleans the data in the DataFrame by doing the following:

- Converts the Date column to a datetime object
- Replaces commas with dots in the Quantity, Quantity_perWeek, Price, and Price_Total_perOrder columns
- Converts the Quantity and Quantity_perWeek columns to integers
- Converts the Price and Price_Total_perOrder columns to floats
- Sorts the DataFrame by the Date column
- Drops any rows that contain missing values or duplicates
- Saves the cleaned DataFrame as a CSV file in the Data directory with the name merged_cleaned.csv

The __main__ block calls both functions with appropriate arguments. It reads raw data files from the Data/Raw directory, converts them to CSV files, concatenates them into a single Pandas DataFrame, cleans the data, and saves the cleaned DataFrame as a CSV file in the Data directory with the name merged_cleaned.csv.


### Script:  `feature_engineering.py`
**Description**: This script processes raw cleaned data and generates the features. 

**Inputs**: 
- `Data/merged_cleaned.csv`: Path to the merged and cleaned data file (CSV format). 

**Outputs**: 
- Calculated features: A CSV file conataining the calculated features, saved in the specified output directory `Data/merged_cleaned_FE_imputed(v).csv`. 

**Usage**: 
```bash
python  feature_engineering.py
```

This code reads in a CSV file ('Data/merged_cleaned.csv') and performs data imputation and feature engineering on it. The imputation is done by filling missing values in the 'Quantity' and 'Price' columns with 0 and the mean of the respective group of 'StoreID' and 'ProductID', respectively. If there are still missing values in the 'Price' column, they are filled with the general average price across all stores and products.

The feature engineering includes creating new features based on the date, such as year, month, day of the month, week of the month, day of the week, week of the year, day of the year, whether it's a weekend, whether it's the start or end of a week/month, the season, and whether it's a holiday in Germany.

Finally, the modified dataframe is saved to a new CSV file ('merged_cleaned_FE_imputed(v).csv') for further analysis.

### Script:  `data_Loader.py'

**Description**: This script converts raw data files to CSV format and cleans the data for further analysis.

**Inputs**:

- folder_path: A string representing the path of the folder containing the raw data files (in a non-CSV format).
- destination_path: A string representing the path where the converted CSV files will be saved.
- column_names: A list of strings representing the column names of the resulting CSV files.


**Outputs**: 

- Concatenated CSV file: A CSV file containing all the converted raw data files concatenated into a single Pandas DataFrame and saved in the specified output directory (destination_path).
- Cleaned CSV file: A cleaned CSV file (with name 'merged_cleaned.csv') containing the cleaned data, saved in the Data directory.

**Usage**: 

```bash
python   data_Loader.py
```

**Functions**:

- convert_to_csv(folder_path, destination_path, column_names): Converts raw data files to CSV format and concatenates them into a single Pandas DataFrame.
- clean_data(df): Cleans the data in the Pandas DataFrame by converting columns to appropriate data types, sorting the DataFrame, and dropping rows with missing values or duplicates.

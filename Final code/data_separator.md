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

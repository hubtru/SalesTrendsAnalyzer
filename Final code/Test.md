### Script:  `test.py`

**Description**: This script loads sales and weather data from CSV files, pivots the sales data to create a new DataFrame with daily sales quantity for each product, calculates the total daily sales quantity, and performs feature engineering to create additional features for machine learning modeling.

**Inputs**: 

- sales_path: A string representing the path of the CSV file containing the sales data.
- weather_path: A string representing the path of the CSV file containing the weather data.

**Outputs**: 

- sales_pivot: A pivoted DataFrame where each row represents a date, each column represents a product ID, and the values represent the quantity of products sold on that date for that product ID.
- sales_total: A DataFrame containing the total sales quantity for each day, saved as a CSV file with the name 'total_sales.csv'.
- features: A DataFrame containing additional engineered features for use in machine learning modeling, saved as a CSV file with the name 'features.csv'.

**Usage**: 
```bash
python  test.py
```

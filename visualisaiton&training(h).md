### Script:   `visualisaiton&training(h).py` 

**Description**: This script performs analysis on sales data for different products in different stores over time.
The script first loads in the cleaned and processed sales data from the CSV file. It then creates a pivot table to organize the data by date and product ID, which is used for further analysis.

The script then generates two types of plots:

Mean quantity of each product sold for each day of the year
Cumulative sales over each week of the year
For each plot, a PNG file is saved in the Plots/ directory.

Finally, the script splits the data into training, validation, and test sets, and normalizes the data using the mean and standard deviation of the training set. The normalized data for each set is saved in a separate CSV file in the Data/ directory.

**Inputs**: 

- Data/cleaned_sales_data.csv: Path to the cleaned and processed sales data file (CSV format).

**Outputs**: 

- Plots: PNG files containing plots of mean quantity of each product sold for each day of the year and cumulative sales over each week of the year. These files are saved in the specified output directory Plots/.
- Normalized data: CSV files containing normalized sales data for training, validation, and test sets. These files are saved in the specified output directory Data/.

**Usage**: 
```bash
python  visualisaiton&training(h).py
```



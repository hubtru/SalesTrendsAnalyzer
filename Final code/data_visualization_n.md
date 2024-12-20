# data_visualization_n
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
This project aims to analyze sales data for a sushi restaurant chain. The data includes daily sales for a variety of sushi dishes, as well as information on the type and EAN (European Article Number) code for each dish. The analysis focuses on  daily/weekly sales for each dish.
 **Inputs**: 
 - merged_cleaned_FE_imputed(v).csv file
 - Sushi Menu.csv file
 
 **Outputs**: 
 - Plots: PNG files containing plots of mean quantity of each product sold for each day of the year and cumulative sales over each week of the year. These files are saved in the specified output directory Plots/.
 
The script allows the user to Choose the store for which the sales data will be analyzed, Choose the number of products to visualize, from one to all available, and select their EAN number from a dropdown menu, Choose the time frame for which the sales data will be analyzed: all available, year, month-year, or day-month-year and Choose the resolution of the analysis: day or week. The code handles errors related to the wrong format of inputs.

PS: If you want to use the code with dropdown menus, you will need to compile the code with the separate code sections. 

## Installation
To use this project, you will need to install the following packages:
- pandas
- matplotlib
You can install these packages using pip:

```sh
pip install pandas matplotlib
```
## Usage

To use this project, you will need to download the repository and run the data_visualization_v.ipynb file. This file contains the code for analyzing the sales data and generating visualizations.

To visualize data with dropdown menus, you will need to run the "data_visualization_dropdown_menu.ipynb" file.
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

The Sushi Menu.csv file has the following columns:

- Nummer (unique dish ID)
- EAN (European Article Number) code
- Type
- sub_type
- Count
- main_ingredient
- sub_ingredient1
- sub_ingredient2
- sub_ingredient3
- sub_ingredient4
- sub_ingredient5
- sub_ingredient6
- sub_ingredient7
- diet_type
- meat_type
- is_new
- Price_per_100g
- Price_per_stuck
- Weight(g)
- Brennwert(KJ)
- Brennwert(Kcal)
- Fett(g)
- gesättigte Fettsäuren(g)
- Kohlenhydrate(g)
- Zucker(g)
- Ballaststoffe(g)
- Eiweiß(g)
- Salz(g)
- Enthält
- Allergic_material
- Religious_kosher
- Zutaten
## Results
The results of the analysis are visualizations that show the sales trends for different products over time. The visualizations are generated using matplotlib and show the cumulative sales for each product on a weekly basis.

## Contributing
Contributions to this project are welcome. Please open an issue or pull request if you have any suggestions or would like to contribute code. It would be helpful to add the possibility to choose the products from the dropdown menu to improve the user experience.





1. Data Loading 
The date looked like the following: 1970-01-01 00:00:00.020230204 so converted it to the following format: 20230204
moreover the separator was comma and not semi-colon so changed it to semi-colon

1. Data cleaning
comma was replaced with dot to prevent errors
columns Quantity and Quantity_perWeek converted to integer
column Price and Price_Total_perOrder converted to float

1. Feature engineering
new columns were created:
- MonthoftheYear
- WeekoftheYear
- WeekoftheMonth
- DayoftheWeek
- DayoftheMonth
- DayoftheYear
- isWeekend
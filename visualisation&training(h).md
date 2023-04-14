
### Script:   `visualisaiton&training(h).ipynb` 

**Description**: This script performs analysis on sales data for different products in different stores over time.
The script first loads in the cleaned and processed sales data from the CSV file. It then creates a pivot table to organize the data by date and product ID, which is used for further analysis. The script then generates two types of plots:
- Mean quantity of each product sold for each day of the year
- Cumulative sales over each week of the year
For each plot, a PNG file is saved in the Plots/ directory.

Finally, the script splits the data into training, validation, and test sets, and normalizes the data using the mean and standard deviation of the training set. The normalized data for each set is saved in a separate CSV file in the Data/ directory. The script preprocesses the data and creates a TensorFlow dataset to train a convolutional neural network (CNN) model on it. The compile_and_fit function is used to compile and train the CNN model using the fit method. The model architecture consists of a 1D convolutional layer with 32 filters and a kernel size of 3, followed by two dense layers, the final one outputting a single value. The model is trained on the conv_window dataset for a maximum of 50 epochs using the compile_and_fit function. The model's loss and mean absolute error are monitored during training, and early stopping is applied using the tf.keras.callbacks.EarlyStopping callback. A wider wide_conv_window dataset is created with a larger input_width and label_width, and the trained model is plotted using the plot method. 

**Inputs**: 

- `Data/cleaned_sales_data.csv`: Path to the cleaned and processed sales data file (CSV format).

**Outputs**: 

- Plots: PNG files containing plots of mean quantity of each product sold for each day of the year and cumulative sales over each week of the year. These files are saved in the specified output directory Plots/.
- Normalized data: CSV files containing normalized sales data for training, validation, and test sets. These files are saved in the specified output directory Data/.

**Usage**: 
```bash
python  visualisaiton&training(h).py
```

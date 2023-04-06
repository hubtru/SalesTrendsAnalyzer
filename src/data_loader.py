import pandas as pd
import os 
import glob


def convert_to_csv(folder_path, destination_path, column_names):
    for filename in os.listdir(folder_path):
        df = pd.read_csv(folder_path + '/' + filename, sep=';', names=column_names)
        df.to_csv(destination_path + '/' + filename[:-4] + '.csv', index=False)

    sum = 0
    for df in glob.glob(os.path.join(destination_path, "*.csv")):
        sum += len(pd.read_csv(df).index)


    all_files = glob.glob(os.path.join(destination_path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)

    df_merged = df_merged.dropna()
    df_merged = df_merged.drop_duplicates()
    df_merged = df_merged.reset_index(drop=True)
    
    if sum - 3 == len(df_merged.index):
        df_merged.to_csv('Data/merged_raw.csv', index=False)
        print('Success')
    else:
        print('length of merged file is not equal to the sum of all files')
        print('length of merged file: ', len(df_merged.index))
        print('sum of all files: ', sum)

    return df_merged

def clean_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    
    df['Quantity'] = df['Quantity'].str.replace(',', '.')
    df['Quantity_perWeek'] = df['Quantity_perWeek'].str.replace(',', '.')
    df['Price'] = df['Price'].str.replace(',', '.')
    df['Price_Total_perOrder'] = df['Price_Total_perOrder'].str.replace(',', '.')

    df['Quantity'] = df['Quantity'].astype(float).astype(int)
    df['Quantity_perWeek'] = df['Quantity_perWeek'].astype(float).astype(int)
    df['Price'] = df['Price'].astype(float)
    df['Price_Total_perOrder'] = df['Price_Total_perOrder'].astype(float)

    df = df.sort_values(by=['Date'])
    df = df.reset_index(drop=True)

    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df.to_csv('Data/merged_cleaned.csv', index=False)


if __name__ == '__main__':
    folder_path = 'Data/Raw'
    destination_path = 'Data/CSV'
    column_names = ['Date', 'StoreID', 'ProductID', 'Quantity', 'Price', 'Quantity_perWeek', 'Price_Total_perOrder']
    clean_data(convert_to_csv(folder_path, destination_path, column_names))
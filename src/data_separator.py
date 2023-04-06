import pandas as pd 


def groupby_storeid(dataset):
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    grouped = dataset.groupby('StoreID')
    for storeID, group in dataset.groupby('StoreID'):
        group = group.drop(columns=['StoreID'])
        group = group.sort_values(by=['Date'])
        group = group.reset_index(drop=True)
        group.to_csv('Data/Store/' + str(storeID) + '.csv', index=False)

        group = group.set_index(['ProductID', 'WeekoftheYear', 'Date'])
        group = group.sort_index(level=0)
        group = group[['Price', 'Quantity', 'Price_Total_perOrder', 'Quantity_perWeek']]
        group.to_csv('Data/Products/{}/'.format(str(storeID)) + str(storeID) + '.csv', index=True)


def weekly_sum(store_ids=None):
    for store_id in store_ids:
        store = pd.read_csv('Data/Store/{}.csv'.format(store_id))
        store['Date'] = pd.to_datetime(store['Date'])
        store['EAN'] = store['ProductID'].map(maps)
        store = store.set_index(['ProductID', 'EAN', 'WeekoftheYear'])
        store = store.sort_index(level=0)

        store['Quantity_sum'] = store.groupby(level=[0, 2])['Quantity'].transform('sum')
        store = store[['Quantity_sum']]
        store = store.reset_index()
        store = store.drop_duplicates(subset=['ProductID', 'EAN', 'WeekoftheYear'])
        store = store.to_csv('Data/Products/{}/weekly_sum.csv'.format(store_id), index=False)


def groupby_week(store_ids=None):
    for store_id in store_ids:
        store = pd.read_csv('Data/Store/{}.csv'.format(store_id))
        store['Date'] = pd.to_datetime(store['Date'])
        store['EAN'] = store['ProductID'].map(maps)
        store = store.groupby(['ProductID', 'EAN'])

        for name, group in store:
            group = pd.DataFrame(group)
            group = group.set_index(['ProductID', 'EAN', 'WeekoftheYear'])
            group = group.sort_index(level=2)
            group.to_csv('Data/Products/{}/weekly/'.format(store_id) + str(name[0]) + '.csv', index=False)



if __name__ == '__main__':
    df = pd.read_csv('Data/merged_cleaned_FE.csv')
    maps = pd.read_csv('Data/products_map.csv')
    maps = maps.set_index('EAN')['Nummer'].to_dict()
    store_ids = df['StoreID'].unique()
    groupby_storeid(df)
    weekly_sum(store_ids)
    groupby_week(store_ids)
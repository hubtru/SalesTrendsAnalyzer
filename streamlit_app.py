import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_rolling_metrics(df, target_col, window_type='rolling', period=7, season=365, alpha=0.2):
    if window_type == 'rolling':
        df[f'{target_col}_rolling_{period}'] = df[target_col].rolling(window=period).mean()
    elif window_type == 'seasonal_rolling':
        seasonal_window = df[target_col].rolling(window=season, min_periods=1)
        df[f'{target_col}_seasonal_rolling_{period}'] = seasonal_window.mean().rolling(window=period).mean()
    elif window_type == 'ewm':
        df[f'{target_col}_ewm_{period}'] = df[target_col].ewm(alpha=alpha).mean().rolling(window=period).mean()
    
    return df


def merge_datasets(products_df, weather_df, sales_df):
    sales_products_df = pd.merge(sales_df, products_df, on='ProductID', how='left')
    merged_df = pd.merge(sales_products_df, weather_df, on='Date', how='left')
    
    return merged_df


def plot_sales_aggregated(sales_df, show_weather=False, window_type=None, period=None, season=None, alpha=None):
    df = sales_df.groupby(['Date', 'StoreID'])['Quantity'].sum().reset_index()
    df = df.merge(sales_df.groupby(['Date', 'StoreID'])['apparent_temperature_mean (°C)'].mean().reset_index(), on=['Date', 'StoreID'])

    if show_weather:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.05, subplot_titles=("Sales", "Temperature"))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Quantity'], name='Quantity'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['apparent_temperature_mean (°C)'], name='Temperature'), row=2, col=1)
        fig.update_layout(title=title, height=600)
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Quantity', row=1, col=1)
        fig.update_yaxes(title_text='Temperature (°C)', row=2, col=1)
    else:
        fig = px.line(df, x='Date', y='Quantity', color='StoreID')

    if window_type and period:
        for store_id in df['StoreID'].unique():
            df_store = df[df['StoreID'] == store_id]
            df_store = calculate_rolling_metrics(df_store, 'Quantity', window_type=window_type, period=period, season=season, alpha=alpha)
            fig.add_trace(go.Scatter(x=df_store['Date'], y=df_store[f'Quantity_{window_type}_{period}'], name=f'{window_type.capitalize()} {period}', line_dash='dash'))

    return fig



def plot_sales_default(sales_df, store_id, product_id, show_weather=False, window_type=None, period=None, season=None, alpha=None):
    df = sales_df.loc[(sales_df['StoreID'] == store_id) & (sales_df['ProductID'] == product_id)]
    
    if show_weather:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.05, subplot_titles=("Sales", "Temperature"))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Quantity'], name='Quantity'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['apparent_temperature_mean (°C)'], name='Temperature'), row=2, col=1)
        fig.update_layout(title=f'Sales for Store {store_id} and Product {product_id}', height=600)
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Quantity', row=1, col=1)
        fig.update_yaxes(title_text='Temperature (°C)', row=2, col=1)
    
    else:
        fig = px.line(df, x='Date', y='Quantity', title=f'Sales for Store {store_id} and Product {product_id}')

    if window_type and period:
        df = calculate_rolling_metrics(df, 'Quantity', window_type=window_type, period=period, season=season, alpha=alpha)
        fig.add_trace(go.Scatter(x=df['Date'], y=df[f'Quantity_{window_type}_{period}'], name=f'{window_type.capitalize()} {period}', line_dash='dash'))

    return fig



if __name__ == '__main__':
    products_df = pd.read_csv('/home/shady/Projects/AI_lab(new)/Data/Sushi Menu.csv')
    products_df['ProductID'] = products_df['Nummer'].astype(str)
    products_df = products_df.drop(columns=['Nummer'])
    products_df['product_name'] = products_df['Type'] + ' ' + products_df['sub_type']
    products_df = products_df.drop(columns=['Type', 'sub_type'])

    weather_df = pd.read_csv('/home/shady/Projects/AI_lab/Data/weather.csv')
    weather_df['Date'] = pd.to_datetime(weather_df['time'])

    sales_df = pd.read_csv('Data/merged_cleaned_FE_imputed(v).csv')
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    sales_df['ProductID'] = sales_df['ProductID'].astype(str)
    sales_df['StoreID'] = sales_df['StoreID'].astype(str)

    merged_df = merge_datasets(products_df, weather_df, sales_df)
    final_df = merged_df[['Date', 'StoreID', 'ProductID', 'Quantity', 'product_name', 'apparent_temperature_mean (°C)']]


    st.set_page_config(layout="wide") 

    data_column, plot_column = st.columns([0.5, 1])

    with data_column:
        st.title('Raw Data')
        st.write(final_df, height=0)


    # show aggregated sales
    with plot_column:
        st.title('Aggregated Sales')
        
        show_weather = st.checkbox('Show Weather', key='wearher_agg')

        window_type = st.selectbox('Select Window Type', ['None', 'rolling', 'seasonal_rolling', 'ewm'], key='window_agg')

        if show_weather and window_type:
            if window_type == 'rolling':
                period = st.slider('Select Period', min_value=1, max_value=30, value=7, step=1)
                season = None
                alpha = None
            elif window_type == 'seasonal_rolling':
                period = st.slider('Select Period', min_value=1, max_value=30, value=7, step=1)
                season = st.slider('Select Season', min_value=1, max_value=30, value=7, step=1)
                alpha = None
            elif window_type == 'ewm':
                period = st.slider('Select Period', min_value=1, max_value=30, value=7, step=1)
                season = None
                alpha = st.slider('Select Alpha', min_value=0.0, max_value=1.0, value=0.5, step=0.01)


            st.plotly_chart(plot_sales_aggregated(final_df, show_weather=show_weather, window_type=window_type, period=period, season=season, alpha=alpha), use_container_width=True)

        else:
            show_weather = False
            window_type = None
            period = None
            season = None
            alpha = None

            st.plotly_chart(plot_sales_aggregated(final_df, show_weather=show_weather, window_type=window_type, period=period, season=season, alpha=alpha), use_container_width=True)



    # Show filtered sales
    with plot_column:
        st.title('Filtered Sales')
        store_id = st.selectbox('Select Store ID', final_df['StoreID'].unique())
        product_id = st.selectbox('Select Product', final_df['ProductID'].unique())

        show_weather = st.checkbox('Show Weather', key='weather_def')
        window_type = st.selectbox('Select Window Type', ['None', 'rolling', 'seasonal_rolling', 'ewm'], key='window_def')

        product_name = final_df.loc[final_df['ProductID'] == product_id, 'product_name'].iloc[0]
        title = f'Sales of {product_name} ({product_id}) in Store {store_id}'

        if show_weather and window_type:
            if window_type == 'rolling':
                period = st.slider('Select Period', min_value=1, max_value=30, value=7, step=1)
                season = None
                alpha = None
            elif window_type == 'seasonal_rolling':
                period = st.slider('Select Period', min_value=1, max_value=30, value=7, step=1)
                season = st.slider('Select Season', min_value=1, max_value=30, value=7, step=1)
                alpha = None
            elif window_type == 'ewm':
                period = st.slider('Select Period', min_value=1, max_value=30, value=7, step=1)
                season = None
                alpha = st.slider('Select Alpha', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

            st.plotly_chart(plot_sales_default(final_df, store_id, product_id, show_weather=show_weather, window_type=window_type, period=period, season=season, alpha=alpha), use_container_width=True)
        
        else:
            show_weather = False
            window_type = None
            period = None
            season = None
            alpha = None

            st.plotly_chart(plot_sales_default(final_df, store_id, product_id, show_weather=show_weather, window_type=window_type, period=period, season=season, alpha=alpha), use_container_width=True)

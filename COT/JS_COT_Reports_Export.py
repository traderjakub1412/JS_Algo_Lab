import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
from datetime import datetime, timedelta


df=pd.read_csv('all_contracts_2023_2024_merged.csv',index_col=0,parse_dates=True)

dfpl = df[:]

# Function to find the most recent Friday
def last_friday():
    today = datetime.now()
    days_to_friday = (4 - today.weekday() + 7) % 7
    last_friday_date = today - timedelta(days=days_to_friday)
    return last_friday_date

# Create a folder named 'Reports' if it doesn't exist
folder_path = 'Reports'
os.makedirs(folder_path, exist_ok=True)

unique_assets = ['EURO FX - CHICAGO MERCANTILE EXCHANGE',
                 'BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE',
                 'CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE',
                 'SWISS FRANC - CHICAGO MERCANTILE EXCHANGE',
                 'JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE',
                 'NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE',
                 'AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE',
                 'BITCOIN - CHICAGO MERCANTILE EXCHANGE',
                 'E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE',
                 'NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE',
                 'E-MINI RUSSELL 2000 INDEX - CHICAGO MERCANTILE EXCHANGE']

for asset in unique_assets:
    asset_df = dfpl[dfpl['Market and Exchange Names'] == asset]

    # Create subplots
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[5, 2,1,1,1,1])

   # Add traces to the subplots
    fig.add_trace(go.Candlestick(x=asset_df.index,
                                open=asset_df['Open'],
                                high=asset_df['High'],
                                low=asset_df['Low'],
                                close=asset_df['Close']), row=1, col=1)

    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['Net_Position_Comm'], mode='lines', name=f'Commercial: {asset_df["Net_Position_Comm"].iloc[-1]}', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['Net_Position_NonComm'], mode='lines', name=f'Non-Commercial: {asset_df["Net_Position_NonComm"].iloc[-1]}', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['Net_Position_NonRept'], mode='lines', name=f'Non-Reportable: {asset_df["Net_Position_NonRept"].iloc[-1]}', line=dict(color='blue')), row=2, col=1)

    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['COT_Index_Comm'], mode='lines', name=f'Commercial Index: {asset_df["COT_Index_Comm"].iloc[-1]}%', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['COT_Index_NonComm'], mode='lines', name=f'Non-Commercial Index: {asset_df["COT_Index_NonComm"].iloc[-1]}%', line=dict(color='green')), row=3, col=1)
    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['COT_Index_NonRept'], mode='lines', name=f'Non-Reportable Index: {asset_df["COT_Index_NonRept"].iloc[-1]}%', line=dict(color='blue')), row=3, col=1)

    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['Open Interest (All)'], mode='lines', name=f'Open Interest: {asset_df["Open Interest (All)"].iloc[-1]}', line=dict(color='black')), row=4, col=1)

    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['COT_Index_OI'], mode='lines', name=f'Open Interest Index: {asset_df["COT_Index_OI"].iloc[-1]}%', line=dict(color='black')), row=5, col=1)

    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['OI_Percentage_Comm'], mode='lines', name=f'Open Interest Percentage Comm Net: {asset_df["OI_Percentage_Comm"].iloc[-1]}%', line=dict(color='blue')), row=6, col=1)

    fig.add_trace(go.Scatter(x=asset_df.index, y=asset_df['OI_Percentage_Short'], mode='lines', name=f'Open Interest Percentage Comm Short: {asset_df["OI_Percentage_Short"].iloc[-1]}%', line=dict(color='red')), row=6, col=1)


    # Update layout with specified height and width
    fig.update_layout(
        title_text=asset,
        showlegend=True,
        height=1000,
        width=1800,
        hovermode='x unified'
    )
    fig.update_traces(xaxis='x')

    fig.update_xaxes(rangeslider_visible=False)

    
    # Get the last Friday's date
    last_friday_date = last_friday()

    
    # Format the date as 'YYYY-MM-DD'
    formatted_date = last_friday_date.strftime('%Y-%m-%d')

    # Save the figure in the 'Reports' folder with date in the file name
    fig_file = os.path.join(folder_path, f"{formatted_date}{asset}_figure.html")
    pio.write_html(fig, file=fig_file, auto_open=False)

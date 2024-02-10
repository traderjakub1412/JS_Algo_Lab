import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt



def bollinger_band(df, col, n, m):
    df = df.copy()
    df[f"BB_M"] = df[col].rolling(window=n).mean()
    df[f"BB_H"] = df[f"BB_M"] + m * df[col].rolling(window=n).std()
    df[f"BB_L"] = df[f"BB_M"] - m * df[col].rolling(window=n).std()
    return df

def sma(df, col, n):
    df[f"SMA_{n}"] = df[col].rolling(window=n).mean()
    return df

def sma_diff(df, col, n, m):
    df = df.copy()
    df[f"SMA_d_{n}"] = df[col].rolling(window=n).mean()
    df[f"SMA_d_{m}"] = df[col].rolling(window=m).mean()
    df[f"SMA_diff"] = df[f"SMA_d_{n}"] - df[f"SMA_d_{m}"]
    return df

def rsi(df, col, n):
    df = df.copy()
    delta = df[col].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=n).mean()
    avg_loss = loss.rolling(window=n).mean()
    rs = avg_gain / avg_loss
    df[f"RSI"] = 100 - (100 / (1 + rs))
    return df

def atr(df, n):
    df = df.copy()
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f"ATR"] = tr.rolling(n).mean()
    return df

def sto_rsi(df, col, n):
    df = rsi(df, col, n)
    df[f"STO_RSI"] = df[f"RSI"].rolling(n).apply(lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)) * 100)
    df[f"STO_RSI_D"] = df[f"STO_RSI"].rolling(3).mean()
    df[f"STO_RSI_K"] = df[f"STO_RSI_D"].rolling(3).mean()
    return df

def ichimoku(df, n1, n2):
    df = df.copy()
    df["BASE"] = (df["high"].rolling(n1).max() + df["low"].rolling(n1).min()) / 2
    df["CONVERSION"] = (df["high"].rolling(n2).max() + df["low"].rolling(n2).min()) / 2
    df["SPAN_A"] = ((df["BASE"] + df["CONVERSION"]) / 2).shift(n2)
    df["SPAN_B"] = ((df["high"].rolling(2*n2).max() + df["low"].rolling(2*n2).min()) / 2).shift(n2)
    return df


def previous_ret(df,col,n):
    df["previous_ret"] = (df[col].shift(int(n)) - df[col]) / df[col]
    return df


def k_enveloppe(df, n=10):
    df[f"EMA_HIGH_{n}"] = df["high"].ewm(span=n).mean()
    df[f"EMA_LOW_{n}"] = df["low"].ewm(span=n).mean()

    df["pivots_high"] = (df["close"] - df[f"EMA_HIGH_{n}"])/ df[f"EMA_HIGH_{n}"]
    df["pivots_low"] = (df["close"] - df[f"EMA_LOW_{n}"])/ df[f"EMA_LOW_{n}"]
    df["pivots"] = (df["pivots_high"]+df["pivots_low"])/2
    return df

def candle_information(df):
    # Candle color
    df["candle_way"] = -1
    df.loc[(df["open"] - df["close"]) < 0, "candle_way"] = 1

    # Filling percentage
    df["filling"] = np.abs(df["close"] - df["open"]) / np.abs(df["high"] - df["low"])

    # Amplitude
    df["amplitude"] = np.abs(df["close"] - df["open"]) / (df["open"] / 2 + df["close"] / 2) * 100

    return df

def data_split(df_model, split, list_X, list_y):

    # Train set creation
    X_train = df_model[list_X].iloc[0:split-1, :]
    y_train = df_model[list_y].iloc[1:split]

    # Test set creation
    X_test = df_model[list_X].iloc[split:-1, :]
    y_test = df_model[list_y].iloc[split+1:]

    return X_train, X_test, y_train, y_test

def quantile_signal(df, n, quantile_level=0.67,pct_split=0.8):

    n = int(n)

    # Create the split between train and test set to do not create a Look-Ahead bais
    split = int(len(df) * pct_split)

    # Copy the dataframe to do not create any intereference
    df = df.copy()

    # Create the fut_ret column to be able to create the signal
    df["fut_ret"] = (df["close"].shift(-n) - df["open"]) / df["open"]

    # Create a column by name, 'Signal' and initialize with 0
    df['Signal'] = 0

    # Assign a value of 1 to 'Signal' column for the quantile with the highest returns
    df.loc[df['fut_ret'] > df['fut_ret'][:split].quantile(q=quantile_level), 'Signal'] = 1

    # Assign a value of -1 to 'Signal' column for the quantile with the lowest returns
    df.loc[df['fut_ret'] < df['fut_ret'][:split].quantile(q=1-quantile_level), 'Signal'] = -1

    return df

def binary_signal(df, n):

    n = int(n)

    # Copy the dataframe to do not create any intereference
    df = df.copy()

    # Create the fut_ret column to be able to create the signal
    df["fut_ret"] = (df["close"].shift(-n) - df["open"]) / df["open"]

    # Create a column by name, 'Signal' and initialize with 0
    df['Signal'] = -1

    # Assign a value of 1 to 'Signal' column for the quantile with the highest returns
    df.loc[df['fut_ret'] > 0, 'Signal'] = 1

    return df


def auto_corr(df, col, n=50, lag=10):
    """
    Calculates the autocorrelation for a given column in a Pandas DataFrame, using a specified window size and lag.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the column for which to compute autocorrelation.
    - col (str): The name of the column in the DataFrame for which to calculate autocorrelation.
    - n (int, optional): The size of the rolling window for calculation. Default is 50.
    - lag (int, optional): The lag step to be used when computing autocorrelation. Default is 10.

    Returns:
    - pd.DataFrame: A new DataFrame with an additional column named 'autocorr_{lag}', where {lag} is the provided lag value. This column contains the autocorrelation values.
    """
    df = df.copy()
    df[f'autocorr_{lag}'] = df[col].rolling(window=n, min_periods=n, center=False).apply(lambda x: x.autocorr(lag=lag), raw=False)
    return df

def DC_market_regime(df, threshold):
    """
    Determines the market regime based on Directional Change (DC) and trend events.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing financial data. The DataFrame should contain a 'close' column 
        with the closing prices, and 'high' and 'low' columns for high and low prices.
    
    threshold : float
        The percentage threshold for DC events.
    
    Returns:
    --------
    df : pandas.DataFrame
        A new DataFrame containing the original data and a new column "market_regime", 
        which indicates the market regime at each timestamp. A value of 1 indicates 
        an upward trend, and a value of 0 indicates a downward trend.
        
    """
    def dc_event(Pt, Pext, threshold):
        """
        Compute if we have a POTENTIAL DC event
        """
        var = (Pt - Pext) / Pext

        if threshold <= var:
            dc = 1
        elif var <= -threshold:
            dc = -1
        else:
            dc = 0

        return dc


    def calculate_dc(df, threshold):
        """
        Compute the start and the end of a DC event
        """

        # Initialize lists to store DC and OS events
        dc_events_up = []
        dc_events_down = []
        dc_events = []
        os_events = []

        # Initialize the first DC event
        last_dc_price = df["close"][0]
        last_dc_direction = 0  # +1 for up, -1 for down

        # Initialize the current Min & Max for the OS events
        min_price = last_dc_price
        max_price = last_dc_price
        idx_min = 0
        idx_max = 0


        # Iterate over the price list
        for i, current_price in enumerate(df["close"]):

            # Update min & max prices
            try:
                max_price = df["high"].iloc[dc_events[-1][-1]:i].max()
                min_price = df["low"].iloc[dc_events[-1][-1]:i].min()
                idx_min = df["high"].iloc[dc_events[-1][-1]:i].idxmin()
                idx_max = df["low"].iloc[dc_events[-1][-1]:i].idxmax()
            except Exception as e:
                pass
                #print(e, dc_events, i)
                #print("We are computing the first DC")

            # Calculate the price change in percentage
            dc_price_min = dc_event(current_price, min_price, threshold)
            dc_price_max = dc_event(current_price, max_price, threshold)


            # Add the DC event with the right index IF we are in the opposite way
            # Because if we are in the same way, we just increase the OS event size
            if (last_dc_direction!=1) & (dc_price_min==1):
                dc_events_up.append([idx_min, i])
                dc_events.append([idx_min, i])
                last_dc_direction = 1

            elif (last_dc_direction!=-1) & (dc_price_max==-1):
                dc_events_down.append([idx_max, i])
                dc_events.append([idx_max, i])
                last_dc_direction = -1

        return dc_events_up, dc_events_down, dc_events


    def calculate_trend(dc_events_down, dc_events_up, df):
        """
        Compute the DC + OS period (trend) using the DC event lists
        """

        # Initialize the variables
        trend_events_up = []
        trend_events_down = []

        # Verify which event occured first (upward or downward movement)

        # If the first event is a downward event
        if dc_events_down[0][0]==0:

            # Iterate on the index 
            for i in range(len(dc_events_down)):

                # If it is the value before the last one we break the loop
                if i==len(dc_events_down)-1:
                    break

                # Calculate the start and end for each trend
                trend_events_up.append([dc_events_up[i][1], dc_events_down[i+1][1]])
                trend_events_down.append([dc_events_down[i][1], dc_events_up[i][1]])

        # If the first event is a upward event
        elif dc_events_up[0][0]==0:

            # Iterate on the index
            for i in range(len(dc_events_up)):

                # If it is the value before the last one we break the loop
                if i==len(dc_events_up)-1:
                    break

                # Calculate the start and end for each trend
                trend_events_up.append([dc_events_down[i][1], dc_events_up[i+1][1]])
                trend_events_down.append([dc_events_up[i][1], dc_events_down[i][1]])

        # Verify the last indexed value for the down ward and the upward trends
        last_up = trend_events_up[-1][1]
        last_down = trend_events_down[-1][1]

        # Find which trend occured last to make it go until now
        if last_down < last_up:
            trend_events_up[-1][1] = len(df)-1
        else:
            trend_events_down[-1][1] = len(df)-1

        return trend_events_down, trend_events_up

    def get_dc_price(dc_events, df):
        dc_events_prices = []
        for event in dc_events:
            prices = [df["close"].iloc[event[0]], df["close"].iloc[event[1]]]
            dc_events_prices.append(prices)
        return dc_events_prices
    
    df = df.copy()
    
    # Extract DC and Trend events
    dc_events_up, dc_events_down, dc_events = calculate_dc(df, threshold=threshold)
    trend_events_down, trend_events_up = calculate_trend(dc_events_down, dc_events_up, df)
    
    df["market_regime"] = np.nan
    for event_up in trend_events_up:
        df.loc[df.index[event_up[1]], "market_regime"] = 1

    for event_down in trend_events_down:
        df.loc[df.index[event_down[1]], "market_regime"] = 0

    df["market_regime"] = df["market_regime"].fillna(method="ffill")
    
    return df


def displacement_detection(df, type_range="standard", strengh=3, period=100):
    """
    This function calculates and adds a 'displacement' column to a provided DataFrame. Displacement is determined based on
    the 'candle_range' which is calculated differently according to the 'type_range' parameter. Then, it calculates the
    standard deviation of the 'candle_range' over a given period and sets a 'threshold'. If 'candle_range' exceeds this 'threshold',
    a displacement is detected and marked as 1 in the 'displacement' column.

    Parameters:
    df (pd.DataFrame): The DataFrame to add the columns to. This DataFrame should have 'open', 'close', 'high', and 'low' columns.
    type_range (str, optional): Defines how to calculate 'candle_range'. 'standard' calculates it as the absolute difference between
                                'close' and 'open', 'extremum' calculates it as the absolute difference between 'high' and 'low'.
                                Default is 'standard'.
    strengh (int, optional): The multiplier for the standard deviation to set the 'threshold'. Default is 3.
    period (int, optional): The period to use for calculating the standard deviation. Default is 100.

    Returns:
    pd.DataFrame: The original DataFrame, but with four new columns: 'candle_range', 'MSTD', 'threshold' and 'displacement'.

    Raises:
    ValueError: If an unsupported 'type_range' is provided.
    """
    df = df.copy()

    # Choose your type_range
    if type_range == "standard":
        df["candle_range"] = np.abs(df["close"] - df["open"])
    elif type_range == "extremum":
        df["candle_range"] = np.abs(df["high"] - df["low"])
    else:
        raise ValueError("Put a right format of type range")

    # Compute the STD of the candle range
    df["MSTD"] = df["candle_range"].rolling(period).std()
    df["threshold"] = df["MSTD"] * strengh

    # Displacement if the candle range is above the threshold
    df["displacement"] = np.nan
    df.loc[df["threshold"] < df["candle_range"], "displacement"] = 1
    df["variation"] = df["close"] - df["open"]

    # Specify the way of the displacement
    df["green_displacement"] = 0
    df["red_displacement"] = 0

    df.loc[(df["displacement"] == 1) & (0 < df["variation"]), "green_displacement"] = 1
    df.loc[(df["displacement"] == 1) & (df["variation"] < 0), "red_displacement"] = 1

    # Shift by one because we only know that we have a displacement at the end of the candle (BE CAREFUL)
    df["green_displacement"] = df["green_displacement"].shift(1)
    df["red_displacement"] = df["red_displacement"].shift(1)

    df["high_displacement"] = np.nan
    df["low_displacement"] = np.nan

    df.loc[df["displacement"] == 1, "high_displacement"] = df["high"]
    df.loc[df["displacement"] == 1, "low_displacement"] = df["low"]

    df["high_displacement"] = df["high_displacement"].fillna(method="ffill")
    df["low_displacement"] = df["low_displacement"].fillna(method="ffill")

    return df


def gap_detection(df, lookback=2):
    """
    Detects and calculates the bullish and bearish gaps in the given DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with columns 'high' and 'low' representing the high and low prices for each period.
    - lookback (int, optional): Number of periods to look back to detect gaps. Default is 2.

    Returns:
    - pd.DataFrame: DataFrame with additional columns:
        * 'Bullish_gap_sup': Upper boundary of the bullish gap.
        * 'Bullish_gap_inf': Lower boundary of the bullish gap.
        * 'Bearish_gap_sup': Upper boundary of the bearish gap.
        * 'Bearish_gap_inf': Lower boundary of the bearish gap.
        * 'Bullish_gap_size': Size of the bullish gap.
        * 'Bearish_gap_size': Size of the bearish gap.

    The function first identifies the bullish and bearish gaps by comparing the current period's high/low prices
    with the high/low prices of the lookback period. It then calculates the size of each gap and forward-fills any
    missing values in the gap boundaries.
    """
    df = df.copy()
    df["Bullish_gap_sup"] = np.nan
    df["Bullish_gap_inf"] = np.nan

    df["Bearish_gap_sup"] = np.nan
    df["Bearish_gap_inf"] = np.nan

    df["Bullish_gap"] = 0
    df["Bearish_gap"] = 0

    df.loc[df["high"].shift(lookback) < df["low"], "Bullish_gap_sup"] = df["low"]
    df.loc[df["high"].shift(lookback) < df["low"], "Bullish_gap_inf"] = df["high"].shift(lookback)
    df.loc[df["high"].shift(lookback) < df["low"], "Bullish_gap"] = 1

    df.loc[df["high"] < df["low"].shift(lookback), "Bearish_gap_sup"] = df["low"].shift(lookback)
    df.loc[df["high"] < df["low"].shift(lookback), "Bearish_gap_inf"] = df["high"]
    df.loc[df["high"] < df["low"].shift(lookback), "Bearish_gap"] = 1

    df["Bullish_gap_size"] = df["Bullish_gap_sup"] - df["Bullish_gap_inf"]
    df["Bearish_gap_size"] = df["Bearish_gap_sup"] - df["Bearish_gap_inf"]

    # Fill the missing values by the last one
    df[["Bullish_gap_sup", "Bullish_gap_inf",
        "Bearish_gap_sup", "Bearish_gap_inf"]] = df[["Bullish_gap_sup", "Bullish_gap_inf",
                                                     "Bearish_gap_sup", "Bearish_gap_inf"]].fillna(method="ffill")

    return df


def kama(df, col, n=10):
    """
    Calculates the Kaufman Adaptive Moving Average (KAMA) for a specified column
    in a DataFrame and adds it as a new column named 'kama_{n}'.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the column for which KAMA is to be calculated.
    col : str
        The name of the column for which KAMA will be calculated.
    n : int
        The window period for KAMA calculation. Default is 10.
    
    Returns:
    --------
    df : pandas.DataFrame
        A new DataFrame with the 'kama_{n}' column added.
    """
    
    df = df.copy()
    
    # Change
    df['change'] = df[col] - df[col].shift(n)
    
    # Volatility
    df['volatility'] = df[col].diff().abs().rolling(window=n).sum()
    
    # Efficiency Ratio (ER)
    df['ER'] = df['change'].abs() / df['volatility']
    
    # Smoothing Constant (SC)
    df['SC'] = ((df['ER'] * (2.0 / (2 + 1) - 2.0 / (30 + 1)) + 2.0 / (30 + 1)) ** 2).clip(0,1) # Ensuring values are between 0 and 1
    
    # KAMA
    df['kama'] = df[col]
    for i in range(n+1, len(df)):
        df.loc[df.index[i], 'kama'] = df.loc[df.index[i-1], 'kama'] + \
            df.loc[df.index[i], 'SC'] * (df.loc[df.index[i], col] - df.loc[df.index[i-1], 'kama'])
    
    # Clean up auxiliary columns and rename the 'kama' column
    df.drop(columns=['change', 'volatility', 'ER', 'SC'], inplace=True)
    df.rename(columns={'kama': f'kama_{n}'}, inplace=True)
    
    return df


def kama_market_regime(df, col, n, m):
    """
    Calculates the Kaufman's Adaptive Moving Average (KAMA) to determine market regime.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame containing price data or other numeric series.
    - col (str): The column name in the DataFrame to apply KAMA.
    - n (int): The period length for the first KAMA calculation.
    - m (int): The period length for the second KAMA calculation.

    Returns:
    - pd.DataFrame: DataFrame with additional columns "kama_diff" and "kama_trend" indicating the market trend.
    """
    
    df = df.copy()
    df = kama(df, col, n)
    df = kama(df, col, m)
    
    df["kama_diff"] = df[f"kama_{m}"] - df[f"kama_{n}"]
    df["kama_trend"] = -1
    df.loc[0<df["kama_diff"], "kama_trend"] = 1
    
    return df


def log_transform(df, col, n):
    """
    Applies a logarithmic transformation to a specified column in a DataFrame 
    and calculates the percentage change of the log-transformed values over a 
    given window size.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the column to be logarithmically transformed.
    col : str
        The name of the column to which the logarithmic transformation is to be applied.
    n : int
        The window size over which to calculate the percentage change of the log-transformed values.

    Returns:
    --------
    df : pandas.DataFrame
        A new DataFrame containing two new columns:
        1. log_{col}: Log-transformed values of the specified column.
        2. ret_log_{n}: Percentage change of the log-transformed values over the window size n.
    """
    df = df.copy()
    df[f"log_{col}"] = np.log(df[col])
    df[f"ret_log_{n}"] = df[f"log_{col}"].pct_change(n)
    
    return df


def derivatives(df,col):
    """
    Calculates the first and second derivatives of a given column in a DataFrame 
    and adds them as new columns 'velocity' and 'acceleration'.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the column for which derivatives are to be calculated.
        
    col : str
        The column name for which the first and second derivatives are to be calculated.

    Returns:
    --------
    df : pandas.DataFrame
        A new DataFrame with 'velocity' and 'acceleration' columns added.

    """
    
    df = df.copy()

    df["velocity"] = df[col].diff().fillna(0)
    df["acceleration"] = df["velocity"].diff().fillna(0)
    
    return df


from statsmodels.tsa.stattools import adfuller

def rolling_adf(df, col, window_size=30):
    """
    Calculate the Augmented Dickey-Fuller test statistic, p-value, and critical values on a rolling window.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the column on which to perform the ADF test.
    col : str
        The name of the column on which to perform the ADF test.
    window_size : int
        The size of the rolling window.

    Returns:
    --------
    df : pandas.DataFrame
        A new DataFrame with additional columns containing rolling ADF test statistic, p-value, and critical values.
    """
    df = df.copy()
    
    # Create empty series to store rolling ADF test statistics, p-values, and critical values
    rolling_adf_stat = pd.Series(dtype='float64', index=df.index)
    rolling_p_values = pd.Series(dtype='float64', index=df.index)
    rolling_critical_values = pd.Series(dtype='float64', index=df.index)

    # Loop through the DataFrame by `window_size` and apply `adfuller`.
    for i in range(window_size, len(df)):
        window = df[col].iloc[i-window_size:i]
        adf_result = adfuller(window)
        adf_stat = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]

        # Store the calculated statistics in respective series
        rolling_adf_stat.at[df.index[i]] = adf_stat
        rolling_p_values.at[df.index[i]] = p_value
        rolling_critical_values.at[df.index[i]] = critical_values['5%']  # Adjust the critical value level if needed

    # Add the rolling ADF test statistic, p-value, and critical values columns to the original DataFrame
    df['rolling_adf_stat'] = rolling_adf_stat
    df['rolling_p_value'] = rolling_p_values
    df['rolling_critical_value'] = rolling_critical_values
    
    return df


def spread(df):
    """
    Calculates the spread between the 'high' and 'low' columns of a given DataFrame 
    and adds it as a new column named 'spread'.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the 'high' and 'low' columns for which the spread is to be calculated.

    Returns:
    --------
    df : pandas.DataFrame
        A new DataFrame with the 'spread' column added.
    """
    df = df.copy()
    df["spread"] = df["high"] - df["low"]
    
    return df


def moving_parkinson_estimator(df, window_size=30):
    """
    Calculate Parkinson's volatility estimator based on high and low prices.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'high' and 'low' columns for each trading period.

    Returns:
    --------
    volatility : float
        Estimated volatility based on Parkinson's method.
    """
    def parkinson_estimator(df):
        N = len(df)
        sum_squared = np.sum(np.log(df['high'] / df['low']) ** 2)

        volatility = math.sqrt((1 / (4 * N * math.log(2))) * sum_squared)
        return volatility
    
    df = df.copy()
    # Create an empty series to store mobile volatility
    rolling_volatility = pd.Series(dtype='float64')

    # Browse the DataFrame by window size `window_size` and apply `parkinson_estimator`.
    for i in range(window_size, len(df)):
        window = df.loc[df.index[i-window_size]: df.index[i]]
        volatility = parkinson_estimator(window)
        rolling_volatility.at[df.index[i]] = volatility

    # Add the mobile volatility series to the original DataFrame
    df['rolling_volatility_parkinson'] = rolling_volatility
    
    return df


def moving_yang_zhang_estimator(df, window_size=30):
    """
    Calculate Parkinson's volatility estimator based on high and low prices.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'high' and 'low' columns for each trading period.

    Returns:
    --------
    volatility : float
        Estimated volatility based on Parkinson's method.
    """
    def yang_zhang_estimator(df):
        N = len(window)
    
        term1 = np.log(window['high'] / window['close']) * np.log(window['high'] / window['open'])
        term2 = np.log(window['low'] / window['close']) * np.log(window['low'] / window['open'])

        sum_squared = np.sum(term1 + term2)
        volatility = np.sqrt(sum_squared / N)

        return volatility
    
    df = df.copy()
    
    # Create an empty series to store mobile volatility
    rolling_volatility = pd.Series(dtype='float64')

    # Browse the DataFrame by window size `window_size` and apply `yang_zhang_estimator`.
    for i in range(window_size, len(df)):
        window = df.loc[df.index[i-window_size]: df.index[i]]
        volatility = yang_zhang_estimator(window)
        rolling_volatility.at[df.index[i]] = volatility

    # Add the mobile volatility series to the original DataFrame
    df['rolling_volatility_yang_zhang'] = rolling_volatility
    
    return df

def get_D_W_M(df):
    """ Getting the day, week and month from the index of the dataframe

    Args:
        df (_type_): _input DataFrame_

    Returns:
        _type_: _Output day_of_week_int, week_number and month  _
    """
    df = df.copy()
    df['day_of_week_int'] = df.index.dayofweek
    df['week_number'] = df.index.isocalendar().week
    df['month'] = df.index.month
    return df

def fractals(df,n):
    
    df = df.copy()

    df["PH"] = df['high'].rolling(n).max()
    df["PL"] = df['low'].rolling(n).min()

    return df

def create_lagged_column(df, column_name, lag=1):
    """
    Create a lagged column for a specific column in the dataframe.
    
    Parameters:
    - df: The dataframe.
    - column_name: The name of the column to be lagged.
    - lag: The number of positions to lag. Default is 1.
    
    Returns:
    - df: Dataframe with the lagged column.
    """
    lag_col_name = f"{column_name}_lag_{lag}"
    df[lag_col_name] = df[column_name].shift(lag)
    return df


def resample_dataframe(df, tf='30T'):
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }
    
    
    return df.resample(tf).apply(ohlc_dict).dropna()


def macd(df, col ,short_window=12, long_window=26, signal_window=9):
    """
    Compute MACD
    
    Parameters:
    - data: DataFrame with 'close' column for closing prices
    - short_window: Short term EMA window, default is 12
    - long_window: Long term EMA window, default is 26
    - signal_window: Signal EMA window, default is 9
    
    Returns:
    - DataFrame with 'macd' and 'signal_line' columns
    """
    
    # Compute Short and Long term EMAs
    short_ema = df[col].ewm(span=short_window, adjust=False).mean()
    long_ema = df[col].ewm(span=long_window, adjust=False).mean()
    
    # Compute MACD line
    df['macd'] = short_ema - long_ema
    
    # Compute Signal line
    df['signal_line'] = df['macd'].ewm(span=signal_window, adjust=False).mean()
    
    return df

def calculate_pivot_points(df):
    """
    Calculate pivot points based on high, low, and close values.
    
    Parameters:
    - high (float): The high price for the period.
    - low (float): The low price for the period.
    - close (float): The closing price for the period.

    Returns:
    - dict: A dictionary containing the pivot point and the support and resistance levels (S1, S2, R1, R2).
    """
    df = df.copy()
    # Calculate the main pivot point
    df["PP"] = (df['high'] + df['low'] + df['close']) / 3.0

    # Calculate resistance levels
    df['R1'] = 2 * df['PP'] - df['low']
    df['R2'] = df['PP'] + (df['high'] - df['low'])

    # Calculate support levels
    df['S1'] = 2 * df['PP'] - df['high']
    df['S2'] = df['PP'] - (df['high'] - df['low'])

    return df


def find_fractals(df):
    # Identify bullish fractals
    bullish = np.where(
        (df['low'] < df['low'].shift(1)) & 
        (df['low'] < df['low'].shift(2)) & 
        (df['low'] < df['low'].shift(-1)) & 
        (df['low'] < df['low'].shift(-2)), df['low'], None)

    # Identify bearish fractals
    bearish = np.where(
        (df['high'] > df['high'].shift(1)) & 
        (df['high'] > df['high'].shift(2)) & 
        (df['high'] > df['high'].shift(-1)) & 
        (df['high'] > df['high'].shift(-2)), df['high'], None)

    # Add the results to the dataframe
    df['PL'] = bullish
    df['PH'] = bearish
    #forwards fill the fractals
    df['PL']=df['PL'].ffill()
    df['PH']=df['PH'].ffill()
    

    return df

def plot_trades(history_book, df, start_time=None, end_time=None):
    """
    Plots the entry and exit points of trades on the price chart within a specified time range,
    and also plots the cumulative PnL and drawdown over time on smaller subplots in the same row.

    :param history_book: DataFrame containing trade history with entry and exit times and prices.
    :param df: DataFrame containing the price data.
    :param start_time: The start time for the plot. If None, plotting starts from the beginning.
    :param end_time: The end time for the plot. If None, plotting goes until the end.
    """
    # Create a figure with specified subplot sizes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1, 1]},sharex=True)

    # Filter the dataframes based on the specified time range
    if start_time is not None:
        df = df[df.index >= start_time]
        history_book = history_book[history_book['Entry Time'] >= start_time]
    if end_time is not None:
        df = df[df.index <= end_time]
        history_book = history_book[history_book['Entry Time'] <= end_time]

    # Plotting the price data
    ax1.plot(df['close'], color='blue', alpha=0.5)  # Change 'Close' to your relevant column name
    ax1.set_title('Price Chart')

    # Iterating over the trades to plot entry and exit points
    for index, trade in history_book.iterrows():
        entry_time = pd.to_datetime(trade['Entry Time'])
        exit_time = pd.to_datetime(trade['Exit Time'])
        entry_price = trade['Entry Price']
        exit_price = trade['Exit Price']

        # Plotting entry and exit points
        ax1.scatter(entry_time, entry_price, color='green' if trade['Type'] == 'Long' else 'red', marker='o')  # Green for Long, Red for Short entry
        ax1.scatter(exit_time, exit_price, color='black', marker='x')  # Black 'X' for exit

        # Connecting entry and exit with a dotted line
        ax1.plot([entry_time, exit_time], [entry_price, exit_price], color='black', linestyle=':', alpha=0.7)

    ax1.set_title('Trade Entry and Exit Points')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')

    # Plotting the cumulative PnL on the second subplot
    ax2.plot(history_book['Exit Time'], history_book['Cumulative Profit'], label='Cumulative PnL', color='purple')
    ax2.set_title('Cumulative Profit and Loss')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cumulative PnL')

    # Calculate and Plot Inverse Drawdown on the third subplot
    cumulative_pips = history_book['PnL'].cumsum()  # Assuming 'PnL' is the correct column for PnL in pips
    running_max = np.maximum.accumulate(cumulative_pips)
    drawdowns_in_pips = running_max - cumulative_pips
    inverse_drawdowns = -drawdowns_in_pips  # Inverting the drawdown values

    # Plotting the inverse drawdown
    ax3.fill_between(history_book['Exit Time'], inverse_drawdowns, 0, where=inverse_drawdowns<=0, color='red', alpha=0.5)
    ax3.plot(history_book['Exit Time'], inverse_drawdowns, label='Inverse Drawdown', color='red')
    ax3.set_title('Inverse Drawdown')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Drawdown in Pips')

    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.tight_layout()
    plt.show()

def calculate_performance_metrics(history_book):
    # Start and End Times
    start_time = history_book['Entry Time'].min()
    end_time = history_book['Exit Time'].max()

    # Win Rate
    profitable_trades = history_book['PnL'] > 0
    win_rate = np.mean(profitable_trades) * 100  # In percentage

    # Buy and Sell Counts
    buy_count = sum(history_book['Type'] == 'Long')
    sell_count = sum(history_book['Type'] == 'Short')

    # Maximum Drawdown Calculation
    cumulative_returns = history_book['PnL'].cumsum()
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = running_max - cumulative_returns
    max_drawdown = drawdowns.max()
    # average drawdown
    average_drawdown = drawdowns.mean() 

    # Calculate Maximum Drawdown Duration and Average Drawdown Duration
    is_drawdown = cumulative_returns < running_max
    drawdown_durations = []
    current_drawdown_start = None

    for i in range(len(is_drawdown)):
        if is_drawdown[i]:
            if current_drawdown_start is None:
                current_drawdown_start = history_book['Entry Time'].iloc[i]
        else:
            if current_drawdown_start is not None:
                duration = history_book['Entry Time'].iloc[i] - current_drawdown_start
                drawdown_durations.append(duration)
                current_drawdown_start = None

    # Handle case where last period is a drawdown
    if current_drawdown_start is not None:
        duration = end_time - current_drawdown_start
        drawdown_durations.append(duration)

    max_drawdown_duration = max(drawdown_durations, default=pd.Timedelta(0))
    avg_drawdown_duration = pd.Series(drawdown_durations).mean()
    max_drawdown_duration_str = str(max_drawdown_duration).split('.')[0]  # Convert Timedelta to string and remove microseconds
    avg_drawdown_duration_str = str(avg_drawdown_duration).split('.')[0]  # Convert Timedelta to string and remove microseconds

    # Total Profit
    total_profit = history_book['PnL'].sum() * 10000

    # Average Time in Trade
    avg_time_in_trade = (history_book['Exit Time'] - history_book['Entry Time']).mean()
    avg_time_str = str(avg_time_in_trade).split('.')[0]  # Convert Timedelta to string and remove microseconds

    # Total Number of Trades
    total_trades = len(history_book)

    # Average wins and losses
    wins = history_book[history_book['PnL'] > 0]
    losses = history_book[history_book['PnL'] <= 0]
    win_rate = len(wins) / len(history_book)
    average_win = wins['PnL'].mean()
    average_loss = losses['PnL'].mean()

    return {
        'Start Time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'End Time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'Win Rate (%)': round(win_rate, 2),
        'Maximum Drawdown (pips)': round(max_drawdown, 4),
        'Average Drawdown (pips)': round(average_drawdown, 4),
        'Maximum Drawdown Duration': max_drawdown_duration_str,
        'Average Drawdown Duration': avg_drawdown_duration_str,
        'Total Number of Trades': total_trades,
        'Buy Count': buy_count,
        'Sell Count': sell_count,
        'Total Profit (pips)': round(total_profit, 5),
        'Average Time in Trade': avg_time_str,
        'Average win' : round(average_win,4),
        'Average loss' : round(average_loss,4)

    } 


def combined_analysis(history_book, df, start_time=None, end_time=None):
    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(history_book)
    print("Performance Metrics:")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value}")

    # Plot trades
    plot_trades(history_book, df, start_time, end_time)



def plot_trades_grid(history_book, df, start_time=None, end_time=None):
    """
    Plots the entry and exit points of trades on the price chart within a specified time range,
    and also plots the cumulative PnL and drawdown over time on smaller subplots in the same row.

    :param history_book: DataFrame containing trade history with entry and exit times and prices.
    :param df: DataFrame containing the price data.
    :param start_time: The start time for the plot. If None, plotting starts from the beginning.
    :param end_time: The end time for the plot. If None, plotting goes until the end.
    """
    # Create a figure with specified subplot sizes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1, 1]},sharex=True)

    # Filter the dataframes based on the specified time range
    if start_time is not None:
        df = df[df.index >= start_time]
        history_book = history_book[history_book['Entry Time'] >= start_time]
    if end_time is not None:
        df = df[df.index <= end_time]
        history_book = history_book[history_book['Entry Time'] <= end_time]

    # Plotting the price data
    ax1.plot(df['close'], color='blue', alpha=0.5)  # Change 'Close' to your relevant column name
    ax1.set_title('Price Chart')

    # Iterating over the trades to plot entry and exit points
    for index, trade in history_book.iterrows():
        entry_time = pd.to_datetime(trade['Entry Time'])
        exit_time = pd.to_datetime(trade['Exit Time'])
        entry_price = trade['Entry Price']
        exit_price = trade['Exit Price']

        # Plotting entry and exit points
        ax1.scatter(entry_time, entry_price, color='green' if trade['Type'] == 'Long' else 'red', marker='o')  # Green for Long, Red for Short entry
        ax1.scatter(exit_time, exit_price, color='black', marker='x')  # Black 'X' for exit

        # Connecting entry and exit with a dotted line
        ax1.plot([entry_time, exit_time], [entry_price, exit_price], color='black', linestyle=':', alpha=0.7)

    ax1.set_title('Trade Entry and Exit Points')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')

    # Plotting the cumulative PnL on the second subplot
    ax2.plot(history_book['Exit Time'], history_book['Cumulative Profit'], label='Cumulative PnL', color='purple')
    ax2.set_title('Cumulative Profit and Loss')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cumulative PnL')

    # Calculate and Plot Inverse Drawdown on the third subplot
    drawdowns_in_pips = history_book['Max Drawdown']
    inverse_drawdowns = -drawdowns_in_pips  # Inverting the drawdown values

    # Plotting the inverse drawdown
    ax3.fill_between(history_book['Exit Time'], inverse_drawdowns, 0, where=inverse_drawdowns<=0, color='red', alpha=0.5)
    ax3.plot(history_book['Exit Time'], inverse_drawdowns, label='Inverse Drawdown', color='red')
    ax3.set_title('Inverse Drawdown')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Drawdown in Pips')

    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.tight_layout()
    plt.show()

        
def calculate_performance_metrics_grid(history_book):
    # Start and End Times
    start_time = history_book['Entry Time'].min()
    end_time = history_book['Exit Time'].max()

    # Win Rate
    profitable_trades = history_book['PnL'] > 0
    win_rate = np.mean(profitable_trades) * 100  # In percentage

    # Buy and Sell Counts
    buy_count = sum(history_book['Type'] == 'Long')
    sell_count = sum(history_book['Type'] == 'Short')

    # Maximum Drawdown Calculation
    max_drawdown = history_book['Max Drawdown'].max()

    # Total Profit
    total_profit = history_book['PnL'].sum() * 10000

    # Average Time in Trade
    avg_time_in_trade = (history_book['Exit Time'] - history_book['Entry Time']).mean()
    avg_time_str = str(avg_time_in_trade).split('.')[0]  # Convert Timedelta to string and remove microseconds

    # Total Number of Trades
    total_trades = len(history_book)

    # Average wins and losses
    wins = history_book[history_book['PnL'] > 0]
    losses = history_book[history_book['PnL'] <= 0]
    win_rate = len(wins) / len(history_book)
    average_win = wins['PnL'].mean()
    average_loss = losses['PnL'].mean()

    return {
        'Start Time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'End Time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'Win Rate (%)': round(win_rate, 2),
        'Maximum Drawdown (pips)': round(max_drawdown, 4),
        'Total Number of Trades': total_trades,
        'Buy Count': buy_count,
        'Sell Count': sell_count,
        'Total Profit (pips)': round(total_profit, 5),
        'Average Time in Trade': avg_time_str,
        'Average win' : round(average_win,4),
        'Average loss' : round(average_loss,4)

    } 


def adf_plot(df, start, end):
    df = df[start:end]
    fig, axes = plt.subplots(4, 1, figsize=(20, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})

    axes[0].plot(df['close'], color='grey', alpha=0.5)
    axes[0].plot(df['Signal'], color='blue')
    axes[1].plot(df['rolling_adf_stat'], color='black')
    axes[2].plot(df['rolling_p_value'], color='black')
    axes[3].plot(df['rolling_critical_value'], color='black')
   
    plt.show()


from statsmodels.tsa.stattools import adfuller

def rolling_adf(df, col, window_size=30, threshold=0.1):
    """
    Calculate the Augmented Dickey-Fuller test statistic, p-value, and critical values on a rolling window.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the column on which to perform the ADF test.
    col : str
        The name of the column on which to perform the ADF test.
    window_size : int
        The size of the rolling window.
    threshold : float
        The threshold for the p-value to trigger the signal.

    Returns:
    --------
    df : pandas.DataFrame
        A new DataFrame with additional columns containing rolling ADF test statistic, p-value, critical values, and signals.
    """
    df = df.copy()
    
    # Create empty series to store rolling ADF test statistics, p-values, and critical values
    rolling_adf_stat = pd.Series(dtype='float64', index=df.index)
    rolling_p_values = pd.Series(dtype='float64', index=df.index)
    rolling_critical_values = pd.Series(dtype='float64', index=df.index)
    signals = pd.Series(dtype='float64', index=df.index)

    # Loop through the DataFrame by `window_size` and apply `adfuller`.
    for i in range(window_size, len(df)):
        window = df[col].iloc[i-window_size:i]
        adf_result = adfuller(window)
        adf_stat = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]

        # Store the calculated statistics in respective series
        rolling_adf_stat.at[df.index[i]] = adf_stat
        rolling_p_values.at[df.index[i]] = p_value
        rolling_critical_values.at[df.index[i]] = critical_values['5%']  # Adjust the critical value level if needed

        # Check if the condition for the signal is met
        if adf_stat <= critical_values['5%'] and p_value <= threshold:
            signals.at[df.index[i]] = df[col].iloc[i]
        else:
            signals.at[df.index[i]] = 0

    # Add the rolling ADF test statistic, p-value, critical values, and signals columns to the original DataFrame
    df['rolling_adf_stat'] = rolling_adf_stat
    df['rolling_p_value'] = rolling_p_values
    df['rolling_critical_value'] = rolling_critical_values
    df['signal'] = signals
    
    return df

def calculate_up_down(df):
    """
    Calculate the 'up' and 'down' columns based on the 'change' column in the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the 'change' column.

    Returns:
    --------
    df : pandas.DataFrame
        A new DataFrame with additional 'up' and 'down' columns.
    """
    df = df.copy()

    df['up'] = 0.0
    df['down'] = 0.0

    for i in range(1, len(df)):
        current_row = df.index[i]
        previous_row = df.index[i - 1]

        if df.at[current_row, 'change'] > 0:
            df.at[current_row, 'up'] = df.at[current_row, 'change'] + df.at[previous_row, 'up']
            df.at[current_row, 'down'] = 0
        elif df.at[current_row, 'change'] < 0:
            df.at[current_row, 'down'] = df.at[current_row, 'change'] + df.at[previous_row, 'down']
            df.at[current_row, 'up'] = 0
        else:
            df.at[current_row, 'up'] = 0 
            df.at[current_row, 'down'] = 0
    return df


def plot_up_down(df, start=10, end=100):
    """
    Plot cumulative change and 'up'/'down' columns for a specified range.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'change', 'up', and 'down' columns.
    start : int
        Start index for plotting.
    end : int
        End index for plotting.

    Returns:
    --------
    None (displays the plot).
    """
    df = df[start:end]
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 8), sharex=True)

    axes[0].plot(df.index, df['change'].cumsum(), label='EUR', color='blue')
    axes[0].legend()
    axes[0].set_ylabel('EUR Prices')

    axes[1].plot(df.index, df['up'], label='up', color='blue')
    axes[1].plot(df.index, df['down'], label='down', color='red')
    axes[1].legend()
    axes[1].set_ylabel('Up Down')

    plt.show()


def update_count_columns(df):
    """
    Update 'count_up' and 'count_down' columns based on the 'up' and 'down' columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'up' and 'down' columns.

    Returns:
    --------
    df : pandas.DataFrame
        A new DataFrame with updated 'count_up' and 'count_down' columns.
    """
    df = df.copy()

    df['count_up'] = 0
    df['count_down'] = 0

    for i in range(1, len(df)):
        cur_index = df.index[i]
        prev_index = df.index[i - 1]

        if df.at[prev_index, 'up'] == 0 and df.at[cur_index, 'up'] > 0:
            df.at[cur_index, 'count_up'] = df.at[prev_index, 'count_up'] + 1
        elif df.at[prev_index, 'up'] > 0 and df.at[cur_index, 'up'] > 0:
            df.at[cur_index, 'count_up'] = df.at[prev_index, 'count_up'] + 1
        elif df.at[prev_index, 'up'] > 0 and df.at[cur_index, 'up'] == 0:
            df.at[cur_index, 'count_up'] = 0
        elif df.at[prev_index, 'down'] == 0 and df.at[cur_index, 'down'] < 0:
            df.at[cur_index, 'count_down'] = df.at[prev_index, 'count_down'] - 1
        elif df.at[prev_index, 'down'] < 0 and df.at[cur_index, 'down'] < 0:
            df.at[cur_index, 'count_down'] = df.at[prev_index, 'count_down'] - 1
        elif df.at[prev_index, 'down'] < 0 and df.at[cur_index, 'down'] == 0:
            df.at[cur_index, 'count_down'] = 0
    return df


def features_counting(df, window, std):
    """
    Calculate spike features based on 'up' and 'down' columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'up' and 'down' columns.
    window : int
        Rolling window size.
    std : float
        Standard deviation multiplier.

    Returns:
    --------
    df : pandas.DataFrame
        A new DataFrame with additional spike-related columns.
    """
    df = df.copy()

    df['spike_up'] = np.where(df['up'] / df['count_up'] > 0, df['up'] / df['count_up'], 0)
    df['spike_down'] = np.where(df['down'] / df['count_down'] > 0, df['down'] / df['count_down'], 0)

    df['spike_up_mean'] = df['spike_up'].rolling(window).mean()
    df['spike_up_std'] = df['spike_up'].rolling(window).std()
    df['spike_up_band'] = df['spike_up_mean'] + std * df['spike_up_std']

    df['spike_down_mean'] = df['spike_down'].rolling(window).mean()
    df['spike_down_std'] = df['spike_down'].rolling(window).std()
    df['spike_down_band'] = df['spike_down_mean'] + std * df['spike_down_std']
    return df

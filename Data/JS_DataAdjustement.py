import pandas as pd
import numpy as np
# Downloaded data from MT5 manualy (larger inputs possible)
##################### INPUT DATAS HIGHER AND LOWER TIMEFRAME ##########################################

hdf = pd.read_csv('FixTimeBars/EURUSD_H1_202001020000_202401031800.csv', sep = '\t')
ldf = pd.read_csv('FixTimeBars/EURUSD_M1_202001020005_202401031840.csv', sep = '\t')

#######################################################################################################


def clean_df_minutes(df):
    df = df.copy()
    df['time'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])

    df.drop(['<DATE>', '<TIME>', '<VOL>', '<SPREAD>'], axis=1, inplace=True)

    df = df[['time', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']]

    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

    df.set_index('time', inplace=True)

    return df


hdf = clean_df_minutes(hdf)
ldf = clean_df_minutes(ldf)


def find_timestamp_extremum(df, df_lower_timeframe):
    """
    :param: df_lowest_timeframe
    :return: self._data with three new columns: Low_time (TimeStamp), High_time (TimeStamp), High_first (Boolean)
    """
    df = df.copy()
    df = df.loc[df_lower_timeframe.index[0]:]

    # Set new columns
    df["low_time"] = np.nan
    df["high_time"] = np.nan

    # Loop to find out which of the high or low appears first
    for i in range(len(df) - 1):

        # Extract values from the lowest timeframe dataframe
        start = df.iloc[i:i + 1].index[0]
        end = df.iloc[i + 1:i + 2].index[0]
        row_lowest_timeframe = df_lower_timeframe.loc[start:end].iloc[:-1]

        # Extract Timestamp of the max and min over the period (highest timeframe)
        try:
            high = row_lowest_timeframe["high"].idxmax()
            low = row_lowest_timeframe["low"].idxmin()

            df.loc[start, "low_time"] = low
            df.loc[start, "high_time"] = high

        except Exception as e:
            print(e)
            df.loc[start, "low_time"] = None
            df.loc[start, "high_time"] = None

    # Verify the number of row without both TP and SL on same time
    percentage_good_row = len(df.dropna()) / len(df) * 100
    percentage_garbage_row = 100 - percentage_good_row

    # if percentage_garbage_row<95:
    print(f"WARNINGS: Garbage row: {'%.2f' % percentage_garbage_row} %")

    df = df.iloc[:-1]

    return df


df = find_timestamp_extremum(hdf, ldf)


save_path = 'FixTimeBars/' + input("Write the name of new csv:") + '.csv'

if len(save_path) > 0:
    df.to_csv(save_path)
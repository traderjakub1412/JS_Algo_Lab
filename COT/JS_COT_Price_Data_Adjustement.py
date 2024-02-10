import pandas as pd
import yfinance as yf
import cot_reports as cot


def merge_cot_and_price(contract_code, instrument_code, start_year, end_year, save_csv=False):
    # COT data
    dfs = []
    for i in range(start_year, end_year + 1):
        single_year = pd.DataFrame(cot.cot_year(i, cot_report_type='legacy_fut'))
        dfs.append(single_year)

    df_cot = pd.concat(dfs)

    asset_cot = df_cot[df_cot['CFTC Contract Market Code (Quotes)'] == contract_code]
    asset_cot = asset_cot[['As of Date in Form YYYY-MM-DD', 'Market and Exchange Names',
                           'Open Interest (All)', 'Noncommercial Positions-Long (All)', 'Noncommercial Positions-Short (All)',
                           'Commercial Positions-Long (All)', 'Commercial Positions-Short (All)',
                           'Nonreportable Positions-Long (All)', 'Nonreportable Positions-Short (All)']].copy()

    # Net Positions
    asset_cot['Net_Position_NonComm'] = asset_cot['Noncommercial Positions-Long (All)'] - asset_cot['Noncommercial Positions-Short (All)']
    asset_cot['Net_Position_Comm'] = asset_cot['Commercial Positions-Long (All)'] - asset_cot['Commercial Positions-Short (All)']
    asset_cot['Net_Position_NonRept'] = asset_cot['Nonreportable Positions-Long (All)'] - asset_cot['Nonreportable Positions-Short (All)']

    # Datetime and index
    asset_cot['As of Date in Form YYYY-MM-DD'] = pd.to_datetime(asset_cot['As of Date in Form YYYY-MM-DD'])
    asset_cot = asset_cot.set_index('As of Date in Form YYYY-MM-DD').sort_index()

    # COT Index for noncom, com, retail and oi
    def calculate_cot_index(series):
        low = series.min()
        high = series.max()
        current_week = series.iloc[-1]

        cot_index = (current_week - low) / (high - low)
        cot_index = round(cot_index * 100, 1)

        return cot_index

    asset_cot['COT_Index_NonComm'] = asset_cot['Net_Position_NonComm'].rolling(26).apply(calculate_cot_index)
    asset_cot['COT_Index_Comm'] = asset_cot['Net_Position_Comm'].rolling(26).apply(calculate_cot_index)
    asset_cot['COT_Index_NonRept'] = asset_cot['Net_Position_NonRept'].rolling(26).apply(calculate_cot_index)
    asset_cot['COT_Index_OI'] = asset_cot['Open Interest (All)'].rolling(26).apply(calculate_cot_index)
    asset_cot['OI_Percentage_Comm'] = (asset_cot['Net_Position_Comm'] / asset_cot['Open Interest (All)'])
    asset_cot['OI_Percentage_Comm'] = asset_cot['OI_Percentage_Comm'].rolling(26).apply(calculate_cot_index)
    asset_cot['OI_Percentage_Short'] = (asset_cot['Commercial Positions-Short (All)'] /asset_cot['Open Interest (All)'])
    asset_cot['OI_Percentage_Short'] = asset_cot['OI_Percentage_Short'].rolling(26).apply(calculate_cot_index)

    # Price data
    asset_price = yf.download(instrument_code, start=f"{start_year}-01-01", end=f"{end_year}-12-31", progress=False)

    # Merged DataFrame
    merged_df = pd.merge_asof(asset_price, asset_cot, left_index=True, right_index=True, direction='nearest')
    
    merged_df[['Open Interest (All)',
       'Noncommercial Positions-Long (All)',
       'Noncommercial Positions-Short (All)',
       'Commercial Positions-Long (All)', 'Commercial Positions-Short (All)',
       'Nonreportable Positions-Long (All)',
       'Nonreportable Positions-Short (All)', 'Net_Position_NonComm',
       'Net_Position_Comm', 'Net_Position_NonRept', 'COT_Index_NonComm',
       'COT_Index_Comm', 'COT_Index_NonRept', 'COT_Index_OI']] = merged_df[['Open Interest (All)',
       'Noncommercial Positions-Long (All)',
       'Noncommercial Positions-Short (All)',
       'Commercial Positions-Long (All)', 'Commercial Positions-Short (All)',
       'Nonreportable Positions-Long (All)',
       'Nonreportable Positions-Short (All)', 'Net_Position_NonComm',
       'Net_Position_Comm', 'Net_Position_NonRept', 'COT_Index_NonComm',
       'COT_Index_Comm', 'COT_Index_NonRept', 'COT_Index_OI']].shift(5)

    if save_csv:
        csv_filename = f"{instrument_code}_{contract_code}_{start_year}_{end_year}_merged.csv"
        merged_df.to_csv(csv_filename, index=True)
        print(f"CSV file saved as {csv_filename}")

    return merged_df

def download_and_merge_data(contracts_info, start_year, end_year, save_csv=False):
    merged_dfs = []

    for contract_info in contracts_info:
        contract_code = contract_info['contract_code']
        instrument_code = contract_info['instrument_code']

        # Download and merge data for each contract and instrument
        merged_df = merge_cot_and_price(contract_code, instrument_code, start_year, end_year, save_csv=False)
        merged_dfs.append(merged_df)

    # Concatenate all merged DataFrames
    final_merged_df = pd.concat(merged_dfs)

    if save_csv:
        csv_filename = f"all_contracts_{start_year}_{end_year}_merged.csv"
        final_merged_df.to_csv(csv_filename, index=True)
        print(f"CSV file saved as {csv_filename}")

    return final_merged_df

# Usage:

# For single instrument
"""
contract_code  = '13874A'
instrument_code  = 'ES=F'
start_year  = 2015
end_year  = 2024

result = merge_cot_and_price(contract_code , instrument_code , start_year , end_year , save_csv=True)
print(result.tail(10))
print('Data adjustement done!')
"""
# For all list
contracts_info = [
    {'contract_code': '099741', 'instrument_code': '6E=F'},  # Euro FX
    {'contract_code': '096742', 'instrument_code': '6B=F'},  # British Pound
    {'contract_code': '090741', 'instrument_code': '6C=F'},  # Canadian Dollar
    {'contract_code': '092741', 'instrument_code': '6S=F'},  # Swiss Franc
    {'contract_code': '097741', 'instrument_code': '6J=F'},  # Japanese Yen
    {'contract_code': '112741', 'instrument_code': '6N=F'},  # NZ Dollar
    {'contract_code': '232741', 'instrument_code': '6A=F'},  # Australian Dollar
    {'contract_code': '133741', 'instrument_code': 'BTC-USD'},  # Bitcoin
    {'contract_code': '13874A', 'instrument_code': 'ES=F'},  # E-MINI S&P 500
    {'contract_code': '209742', 'instrument_code': 'NQ=F'},  # NASDAQ MINI
    {'contract_code': '239742', 'instrument_code': 'RTY=F'},  # Russell E-MINI
    {'contract_code': '050642', 'instrument_code': 'CB=F'},  # Butter (Cash Settled)
    {'contract_code': '054642', 'instrument_code': 'HE=F'},  # Lean Hogs
    {'contract_code': '057642', 'instrument_code': 'LE=F'},  # Live Cattle
    {'contract_code': '058644', 'instrument_code': 'LBR=F'},  # Lumber
]


start_year = 2023
end_year = 2024

result = download_and_merge_data(contracts_info, start_year, end_year, save_csv=True)
print(result.tail(10))
print('Data adjustment done!')



"""
EURO FX - CHICAGO MERCANTILE EXCHANGE                                Code-099741, 6E=F,
BRITISH POUND - CHICAGO MERCANTILE EXCHANGE                          Code-096742, 6B=F,
CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE                        Code-090741, 6C=F,
SWISS FRANC - CHICAGO MERCANTILE EXCHANGE                            Code-092741, 6S=F, 
JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE                           Code-097741, 6J=F,
NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE                              Code-112741, 6N=F,
AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE                      Code-232741, 6A=F,
BITCOIN - CHICAGO MERCANTILE EXCHANGE                                Code-133741, BTC-USD,
E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE                         Code-13874A, ES=F,
NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE                            Code-209742, NQ=F,
RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE                         Code-239742, RTY=F,
BUTTER (CASH SETTLED) - CHICAGO MERCANTILE EXCHANGE                  Code-050642, CB=F,
LEAN HOGS - CHICAGO MERCANTILE EXCHANGE                              Code-054642, HE=F,
LIVE CATTLE - CHICAGO MERCANTILE EXCHANGE                            Code-057642, LE=F,
LUMBER - CHICAGO MERCANTILE EXCHANGE                                 Code-058644, LBR=F

"""
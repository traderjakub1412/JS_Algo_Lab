from Strategies.TreePcaQuantile import *
from MainScripts.CombinatorialPurgedCV import *


df = pd.read_csv("../Data/FixTimeBars/EURUSD_2020_2023_1H_READY.csv", index_col="time", parse_dates=True)


params = {
    "tp": [0.005 ],
    "sl": [-0.005 ],
    "look_ahead_period": 20,
    "sma_slow": 120,
    "sma_fast": 30,
    "rsi": 21,
    "atr": 15,
    "cost": 0.0001,
    "leverage": 5,
    "list_X": ["SMA_diff", "RSI", "ATR", "candle_way", "filling", "amplitude", "SPAN_A", "SPAN_B", "BASE", "STO_RSI",
               "STO_RSI_D", "STO_RSI_K", "previous_ret"],
    "train_mode": True,
}

BT = Backtest(data=df, TradingStrategy=TreePcaQuantile, parameters=params, run_directly=True,title='TreePcaQuantile')


def featureEngineering(df):

    import pandas as pd
    import numpy as np

    
    # Formatting the Date, there's probably a better way to do this, but this works I guess..
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    #Switching NaNs to zeros so we dont get infinites
    df.fillna(0, inplace=True)
    

    # Add a column called Change for the main value, which is the percent change from the previous day's close
    df['Close_Change'] = df['Close'].pct_change()
    df['Open_Change'] = df['Open'].pct_change()
    df['High_Change'] = df['High'].pct_change()
    df['Low_Change'] = df['Low'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    

    #drop dividends and splits
    df.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)

    ## Feature Engineering using TA Library
    #import ta library
    import ta

    #Awesome Oscillator
    from ta.momentum import AwesomeOscillatorIndicator
    df['AO'] = ta.momentum.AwesomeOscillatorIndicator(df['High'], df['Low'], 5, 34).awesome_oscillator()

    # # Kaufman's Adaptive Moving Average
    # from ta.momentum import KAMAIndicator
    # df['KAMA'] = ta.momentum.KAMAIndicator(df['Close'], 10, 2, 30).kama()

    # # Percentage Price Oscillator
    # from ta.momentum import PercentagePriceOscillator
    # df['PPO'] = ta.momentum.PercentagePriceOscillator(df['Close'], 12, 26).ppo()

    # # Percentage Volume Oscillator
    # from ta.momentum import PercentageVolumeOscillator
    # df['PVO'] = ta.momentum.PercentageVolumeOscillator(df['Volume'], 12, 26).pvo()

    # Rate of Change
    from ta.momentum import ROCIndicator
    df['ROC'] = ta.momentum.ROCIndicator(df['Close'], 12).roc()

    # Relative Strength Index
    from ta.momentum import RSIIndicator
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()

    # Stochastic RSI
    from ta.momentum import StochRSIIndicator
    df['StochRSI'] = ta.momentum.StochRSIIndicator(df['Close'], 14, 3, 3).stochrsi()

    # Stochastic Oscillator
    from ta.momentum import StochasticOscillator
    df['StochOsc'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], 14, 3, 3).stoch()

    # True Strength Index
    from ta.momentum import TSIIndicator
    df['TSI'] = ta.momentum.TSIIndicator(df['Close'], 25, 13).tsi()

    # Chaikin Money Flow
    from ta.volume import ChaikinMoneyFlowIndicator
    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], 20).chaikin_money_flow()

    # # Ease of Movement
    # from ta.volume import EaseOfMovementIndicator
    # df['EoM'] = ta.volume.EaseOfMovementIndicator(df['High'], df['Low'], df['Close'], df['Volume'], 14).ease_of_movement()

    # # Force Index
    # from ta.volume import ForceIndexIndicator
    # df['FI'] = ta.volume.ForceIndexIndicator(df['Close'], df['Volume'], 13).force_index()

    # # Money Flow Index
    # from ta.volume import MFIIndicator
    # df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], 14).money_flow_index()

    # # Negative Volume Index
    # from ta.volume import NegativeVolumeIndexIndicator
    # df['NVI'] = ta.volume.NegativeVolumeIndexIndicator(df['Close'], df['Volume']).negative_volume_index()

    # # On Balance Volume
    # from ta.volume import OnBalanceVolumeIndicator
    # df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

    # # Volume Price Trend
    # from ta.volume import VolumePriceTrendIndicator
    # df['VPT'] = ta.volume.VolumePriceTrendIndicator(df['Close'], df['Volume']).volume_price_trend()

    # # Volume Weighted Average Price
    # from ta.volume import volume_weighted_average_price
    # df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])

    # Average True Range
    from ta.volatility import AverageTrueRange
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 14).average_true_range()

    #Bollinger Bands
    from ta.volatility import BollingerBands
    df['BB_H'] = ta.volatility.BollingerBands(df['Close'], 20, 2).bollinger_hband()
    df['BB_L'] = ta.volatility.BollingerBands(df['Close'], 20, 2).bollinger_lband()
    df['BB_M'] = ta.volatility.BollingerBands(df['Close'], 20, 2).bollinger_mavg()
    df['BB_W'] = ta.volatility.BollingerBands(df['Close'], 20, 2).bollinger_wband()

    # Take the distance between BB_H/L with Close price and calculate its ratio with the width of the Bollinger Bands 
    df['BB_HL'] = (df['BB_H'] - df['Close']) / df['BB_W']
    df['BB_LH'] = (df['Close'] - df['BB_L']) / df['BB_W']

    #Drop bollinger bands
    df.drop(['BB_H', 'BB_L', 'BB_M', 'BB_W'], axis=1, inplace=True)

    # # Donchian Channel
    # from ta.volatility import DonchianChannel
    # df['DC_H'] = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], 20).donchian_channel_hband()
    # df['DC_L'] = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], 20).donchian_channel_lband()
    # df['DC_M'] = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], 20).donchian_channel_mband()
    # df['DC_P'] = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], 20).donchian_channel_pband()
    # df['DC_W'] = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], 20).donchian_channel_wband()

    # # Keltner Channel
    # from ta.volatility import KeltnerChannel
    # df['KC_H'] = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], 20, 2).keltner_channel_hband()
    # df['KC_L'] = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], 20, 2).keltner_channel_lband()
    # df['KC_M'] = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], 20, 2).keltner_channel_mband()
    # df['KC_P'] = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], 20, 2).keltner_channel_pband()
    # df['KC_W'] = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], 20, 2).keltner_channel_wband()


    # # Ulcer Index
    # from ta.volatility import UlcerIndex
    # df['UI'] = ta.volatility.UlcerIndex(df['Close'], 14).ulcer_index()

    # # Average Directional Movement Index
    # from ta.trend import ADXIndicator
    # df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], 14).adx()
    # df['ADX_-DI'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], 14).adx_neg()
    # df['ADX_+DI'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], 14).adx_pos()

    # # Aroon Indicator
    # from ta.trend import AroonIndicator
    # df['Aroon_Down'] = ta.trend.AroonIndicator(df['Close'], 14).aroon_down()
    # df['Aroon_Up'] = ta.trend.AroonIndicator(df['Close'], 14).aroon_up()
    # df['Aroon_Ind'] = ta.trend.AroonIndicator(df['Close'], 14).aroon_indicator()

    # Commodity Channel Index
    from ta.trend import CCIIndicator
    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close'], 14).cci()

    # # Detrended Price Oscillator
    # from ta.trend import DPOIndicator
    # df['DPO'] = ta.trend.DPOIndicator(df['Close'], 14).dpo()

    # Exponential Moving Average (5, 20, 50, 100, 200)
    from ta.trend import EMAIndicator
    df['EMA_5'] = ta.trend.EMAIndicator(df['Close'], 5).ema_indicator()
    df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], 20).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], 50).ema_indicator()
    df['EMA_100'] = ta.trend.EMAIndicator(df['Close'], 100).ema_indicator()
    df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], 200).ema_indicator()

    # EMA ratios
    df['Ratio_EMA5'] = df['Close'] / df['EMA_5']
    df['Ratio_EMA20'] = df['EMA_5'] / df['EMA_20']
    # df['Ratio_EMA50'] = df['EMA_20'] / df['EMA_50']
    # df['Ratio_EMA100'] = df['EMA_50'] / df['EMA_100']
    # df['Ratio_EMA200'] = df['EMA_100'] / df['EMA_200']

    # Ratio of the change between Ratio_EMA and its previous value
    df['Ratio_EMA5_Change'] = df['Ratio_EMA5'].pct_change()
    df['Ratio_EMA20_Change'] = df['Ratio_EMA20'].pct_change()
    # df['Ratio_EMA50_Change'] = df['Ratio_EMA50'].pct_change()
    # df['Ratio_EMA100_Change'] = df['Ratio_EMA100'].pct_change()
    # df['Ratio_EMA200_Change'] = df['Ratio_EMA200'].pct_change()

    #drop EMAs
    df.drop(['EMA_5', 'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200'], axis=1, inplace=True)


    # # KST Oscillator
    # from ta.trend import KSTIndicator
    # df['KST'] = ta.trend.KSTIndicator(df['Close'], 14, 15, 16, 30, 10, 10, 10, 15).kst()
    # df['KST_Signal'] = ta.trend.KSTIndicator(df['Close'], 14, 15, 16, 30, 10, 10, 10, 15).kst_sig()
    # df['KST_Diff'] = ta.trend.KSTIndicator(df['Close'], 14, 15, 16, 30, 10, 10, 10, 15).kst_diff()

    # MACD
    from ta.trend import MACD
    df['MACD'] = ta.trend.MACD(df['Close'], 12, 26, 9).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close'], 12, 26, 9).macd_signal()
    df['MACD_Diff'] = ta.trend.MACD(df['Close'], 12, 26, 9).macd_diff()

    # # Mass index
    # from ta.trend import MassIndex
    # df['Mass_Index'] = ta.trend.MassIndex(df['High'], df['Low'], 9, 25).mass_index()

    # Simple Moving Average (5, 20, 50, 100, 200)
    # from ta.trend import SMAIndicator
    # df['SMA_5'] = ta.trend.SMAIndicator(df['Close'], 5).sma_indicator()
    # df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], 20).sma_indicator()
    # df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], 50).sma_indicator()
    # df['SMA_100'] = ta.trend.SMAIndicator(df['Close'], 100).sma_indicator()
    # df['SMA_200'] = ta.trend.SMAIndicator(df['Close'], 200).sma_indicator()

    # SMA ratios ?

    # # Schaff Trend Cycle
    # from ta.trend import STCIndicator
    # df['STC'] = ta.trend.STCIndicator(df['Close']).stc()

    # # Trix
    # from ta.trend import TRIXIndicator
    # df['TRIX'] = ta.trend.TRIXIndicator(df['Close'], 15).trix()

    # # Vortex Indicator
    # from ta.trend import VortexIndicator
    # df['Vortex_Indicator'] = ta.trend.VortexIndicator(df['High'], df['Low'], df['Close'], 14).vortex_indicator_diff()
    # df['Vortex_Indicator_Pos'] = ta.trend.VortexIndicator(df['High'], df['Low'], df['Close'], 14).vortex_indicator_pos()
    # df['Vortex_Indicator_Neg'] = ta.trend.VortexIndicator(df['High'], df['Low'], df['Close'], 14).vortex_indicator_neg()
    # df['Vortex_Indicator_Diff'] = ta.trend.VortexIndicator(df['High'], df['Low'], df['Close'], 14).vortex_indicator_diff()

    # Drop NA -> Maybe 0s instead?
    df.fillna(0, inplace=True)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df
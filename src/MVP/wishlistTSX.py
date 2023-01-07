def wishlistTSX(TSX_info):
    # Create a new column called 'targetLowProfit' = targetLowPrice / currentPrice * 100
    TSX_info['targetLowProfit'] = TSX_info['targetLowPrice'] / TSX_info['currentPrice'] * 100

    # Create a new column called 'targetHighProfit' = targetHighPrice / currentPrice * 100
    TSX_info['targetHighProfit'] = TSX_info['targetHighPrice'] / TSX_info['currentPrice'] * 100

    # Make a top list
    TSX_top = TSX_info[(TSX_info['recommendationKey'].str.contains('buy')) & (TSX_info['numberOfAnalystOpinions'] >= 3)]

    # only keep the columns we need longname, quotetype, symbol, sector, industry, numberOfAnalystOpinions, recommendationKey, targetLowProfit, targetHighProfit, shortRatio, sharesoutstanding, revenuePerShare, fiftytwoweeklow, fiftytwoweekhigh, fiftydayaverage, averagedailyvolume10day, twohundreddayaverage
    TSX_top = TSX_top[['longName', 'quoteType', 'symbol', 'sector', 'industry', 'currentPrice', 'numberOfAnalystOpinions', 'recommendationKey', 'targetLowProfit', 'targetLowPrice', 'targetHighProfit', 'targetHighPrice', 'shortRatio', 'sharesOutstanding', 'revenuePerShare', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'fiftyDayAverage', 'averageDailyVolume10Day', 'twoHundredDayAverage']]

    #remove stocks under 1$
    TSX_top = TSX_top[TSX_top['currentPrice'] >= 1]

    #remove stocks with targetLowProfit less than 110%
    TSX_top = TSX_top[TSX_top['targetLowProfit'] >= 110]

    #remove stocks with targetLowProfit greater than 1000%
    TSX_top = TSX_top[TSX_top['targetLowProfit'] <= 1000]

    #remove stocks with averageDailyVolume10Day less than 100K
    TSX_top = TSX_top[TSX_top['averageDailyVolume10Day'] >= 100000]

    #remove stocks with less than 100M shares outstanding
    TSX_top = TSX_top[TSX_top['sharesOutstanding'] >= 100000000]

    # remove stocks where currentPrice is less than half of fiftyTwoWeekhigh
    TSX_top = TSX_top[TSX_top['currentPrice'] >= TSX_top['fiftyTwoWeekHigh'] *0.4]

    # remove stock where current price is below 80% of twoHundredDayAverage
    TSX_top = TSX_top[TSX_top['currentPrice'] >= TSX_top['twoHundredDayAverage'] * 0.9]

    #show TSX_top order by targetLowProfit descending
    TSX_top.sort_values(by='targetLowProfit', ascending=False).shape

    return TSX_top
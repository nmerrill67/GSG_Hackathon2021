#!/usr/bin/env python3
import os
import numpy as np
from matplotlib import pyplot as plt

# Read a csv downloaded from here https://www.cryptodatadownload.com/data
# If  not in USD or USDT, provide btc_to_usd which is the return of this 
# function on that csv.
# If not_before is set, don't include any data before this unnix timestamp
def read_crypto_csv(fname, btc_to_usd=None, not_before=0):
    with open(fname, 'r') as f:
        data = f.readlines() 
    assert len(data) > 2

    # Remove junk lines
    data = data[2:]
    if data[-1] == '':
        data = data[:-1]
    n = len(data)

    # NOTE: It seems volume is just for Poloniex, so don't use
    ret = {
        "unix_time": [],
        "str_time": [],
        "open": [],
        "close": [],
        "low": [],
        "high": []
    }

    # Extract into dict
    units = None
    for line in data:
        line_spl = line.split(',')

        # Determine units
        if units is None:
            units = line_spl[2].split('/')[1][:3]
            assert units in ['BTC', 'USD']
            if units == 'BTC':
                assert btc_to_usd is not None # Provide the conversion
        
        unix = int(line_spl[0])
        if unix >= not_before:
            to_usd_ratio = 1
            if units == 'BTC':
                btc_to_usd_row = np.argwhere(btc_to_usd["unix_time"] == unix)
                if btc_to_usd_row.shape[0] == 0:
                    continue # Can't find this time in the conversion table

                # Just use the closing value
                to_usd_ratio = btc_to_usd["close"][btc_to_usd_row[0]]
            
            # Insert the data if we can get it all in USD
            ret["unix_time"].append(unix)
            ret["str_time"].append(line_spl[1])
            ret["open"].append(float(line_spl[3]) * to_usd_ratio)
            ret["high"].append(float(line_spl[4]) * to_usd_ratio)
            ret["low"].append(float(line_spl[5]) * to_usd_ratio)
            ret["close"].append(float(line_spl[6]) * to_usd_ratio)
    
    # Convert all the lists to numpy arrays for better access later
    # Also reverse the lists so that they're oldest to newest
    for k,v in ret.items():
        ret[k] = np.array(v[::-1])
    return ret

if __name__ == '__main__':
    data_dir = 'data/hist_data'
    btc_usd_fl = os.path.join(data_dir, 'Poloniex_BTCUSDT_1h.csv')
    doge_btc_fl = os.path.join(data_dir, 'Poloniex_DOGEBTC_1h.csv')
    eth_btc_fl = os.path.join(data_dir, 'Poloniex_ETHBTC_1h.csv')

    not_before = 1618963200 # 4/20/2021 midnight

    btc_usd = read_crypto_csv(btc_usd_fl, not_before=not_before)
    doge_usd = read_crypto_csv(doge_btc_fl, btc_to_usd=btc_usd, not_before=not_before)
    eth_usd = read_crypto_csv(eth_btc_fl, btc_to_usd=btc_usd, not_before=not_before)
    
    plt.subplot(311)
    plt.plot(btc_usd["str_time"], btc_usd["close"], label='BTC', color='r')
    plt.legend()
    plt.xticks([])

    plt.subplot(312)
    plt.plot(doge_usd["str_time"], doge_usd["close"], label='DOGE', color='g')
    plt.legend()
    plt.xticks([])
    plt.ylabel('Closing Price (USD)')

    plt.subplot(313)
    plt.plot(eth_usd["str_time"], eth_usd["close"], label='ETH', color='b')
    plt.legend()
    locs, labels = plt.xticks()
    labels = btc_usd["str_time"]
    print(f"Read in {len(labels)} BTC market entries")
    plt.xticks([locs[0], locs[len(locs)//2], locs[-1]],
               [labels[0], labels[len(labels)//2], labels[-1]])

    plt.show()

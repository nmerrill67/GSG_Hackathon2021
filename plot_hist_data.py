#!/usr/bin/env python3
import os
from matplotlib import pyplot as plt
from utils import *

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

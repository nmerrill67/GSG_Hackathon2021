import numpy as np
import pandas as pd

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




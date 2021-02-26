#%% Imports
import math
import numpy as np
import statistics
import pandas as pd
import scipy.stats as scs
# from pylab import mpl, plt
import random
import os
from matplotlib.pylab import mpl, plt

#%% Read data into code
data = pd.read_csv('crypto-markets.csv')

data.describe()

new_data = data.loc[data['date'] > '2017 - 01 - 01'] # Sorterer dato, så datoer før 01-01-2017 ikke bruges :D
new_data.info()


#%% Exercise 1
print("See external file LINK")

#%% Exercise 2

#%% Exercise 2 preperation
random.seed(42069)

#%% Exercise 2.1
new_data_1 = new_data[['date', 'name', 'volume']] # Get data needed

# Use to print out nicely :D
new_data_median_sorted = new_data_1.groupby("name").agg({"volume": ["median"]}).sort_values(by=("volume", "median"), ascending=False) # Group by name, aggregate volume by median and sort by volume as primary and then median as secondary
# print(new_data_median_sorted)
crypto_curr = new_data_median_sorted.head(50) # Get 50 first results - the top 50 because of sorting
print(crypto_curr) # Print results
# 2017-06-01 - 2018-01-01 (Close values)

#%% Exercise 2.2
pd_data = new_data[['date', 'name', 'close']] # Filter to date, name and close

# Choosing a random date
unique_days = pd_data["date"].unique() # filter unique days in data
n_unique_days = len(unique_days) # count unique days

n_randdom_day = random.randint(0,n_unique_days-1) # Select random entry number
unique_day = unique_days[n_randdom_day] # get entry

is_date = pd_data['date']==unique_day # create filtering list
pd_data = pd_data[is_date] # Filter data by date
#r_d_pd_data = pd_data["date" == unique_day] # filter by random day


# Choose a random N of cryptos
top_currencies = crypto_curr.index.values # top 50 curencies
random_top_curr = set(random.choices(top_currencies, k=random.randint(1,50))) # Select random number of currensies





#n.randdom.day = random.randint(0,n.unique_days)
#unique_day = unique_days[n.randdom.day] # get entry



#new_data_median_sorted.

#len
#pd_data = pd_data.pivot_table(index = pd_data.index, columns = 'name', values = 'close')
#log_ret = np.log(pd_data/pd_data.shift(1))
#log_ret = log_ret.dropna(axis = 1, how = 'all').dropna(axis = 0, how = 'all')

data_cmc = pd.read_csv('crypto-markets.csv')
data_cmc['date'] = pd.to_datetime(data_cmc['date'])
data_cmc.index = pd.DatetimeIndex(data_cmc['date'])
symbol = 'bitcoin'
data_symbol = pd.DataFrame(data_cmc['close'].loc[data_cmc['name'] == symbol])
print(data_symbol)
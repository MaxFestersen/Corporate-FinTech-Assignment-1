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
new_data_1 = new_data[['date', 'name', 'volume']]

# Use to print out nicely :D
new_data_median_sorted = new_data_1.groupby("name").agg({"volume": ["median"]}).sort_values(by=("volume", "median"), ascending=False)
print(new_data_median_sorted)
new_data_median_sorted.head(50)

# 2017-06-01 - 2018-01-01 (Close values)
#%% Exercise 2


pd_data = new_data[['date', 'name', 'close']]









unique_days = pd_data["date"].unique() # filter unique days in data
n.unique_days = unique_days.count() # count unique days
random.seed(42069)
n.randdom.day = random.randint(0,n.unique_days)
unique_day = unique_days[n.randdom.day] # get entry
r.d.pd_data = pd_data["date" = unique_day] # filter by random day
print(random.randint(0,50))

new_data_median_sorted.

#len
pd_data = pd_data.pivot_table(index = pd_data.index, columns = 'name', values = 'close')
log_ret = np.log(pd_data/pd_data.shift(1))
log_ret = log_ret.dropna(axis = 1, how = 'all').dropna(axis = 0, how = 'all')


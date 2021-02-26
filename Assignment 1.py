#%% Imports
import math
import numpy as np
import statistics
import pandas as pd
import scipy.stats as scs
# from pylab import mpl, plt
import random
from matplotlib.pylab import mpl, plt
from datetime import datetime
from dateutil.relativedelta import relativedelta

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
#%% Exercise 2.2 a
pd_data = new_data[['date', 'name', 'close']] # Filter to date, name and close

# Choosing a random date
unique_days = pd_data["date"].unique() # filter unique days in data
last_day = datetime.strptime(unique_days[-1], "%Y-%m-%d") # Get last date entry in date format (to make date calculations)
last_available_date = last_day - relativedelta(months=+6) # Get last date that is at least 6 months from end date
last_available_date = last_available_date.strftime("%Y-%m-%d") # Format as string
unique_days_filter = unique_days <= last_available_date # create filter list
unique_days = unique_days[unique_days_filter] # filter unavailable dates (not with 6 months after)
n_unique_days = len(unique_days) # count unique days
n_randdom_day = random.randint(0,n_unique_days-1) # Select random entry number
unique_day = unique_days[n_randdom_day] # get entry
unique_day_until = datetime.strptime(unique_day, "%Y-%m-%d") + relativedelta(months=+6) # Find 6 months after random date
unique_day_until = unique_day_until.strftime("%Y-%m-%d") # Format as string

# Print random date information:
print("The random chosen date is: " + unique_day)


# Choose a random N of cryptos
random_top_curr = crypto_curr.index.values # top 50 curencies - from 2.1
for n in range(random.randint(0,49)): # Randomly run from 0 to 49 times (so at least 1 value is left)
    random_top_curr = np.delete(random_top_curr, random.randint(0,len(random_top_curr)-1), 0) # remove a random value

weigth_random_top_curr = len(random_top_curr) # Amount of currencsies
print("There are ", weigth_random_top_curr, " currencies.\nThe random top currencies are:")
random_top_curr

#%% Exercise 2.2 a filtering
# Filter the days
is_dates = pd_data['date'] >= unique_day # create filtering list from random date
pd_data = pd_data[is_dates] # Filter by the random date and after
is_dates = pd_data['date'] <= unique_day_until # create filtering list until 6 months after random date
pd_data = pd_data[is_dates] # Filter by the random date and after

# Filter by top currensies
curr_in = pd_data['name'].isin(random_top_curr) # Create filter list by random top currensies
pd_data = pd_data[curr_in] # Apply filter list

#%% Exercise 2.2 b
weigth = 1/weigth_random_top_curr # calculate 1/N weights

ret = pd_data_1/pd_data_1.shift(1)
ret = ret.dropna(axis = 1, how = 'all').dropna(axis = 0, how = 'all')
pd_data_1 = pd_data.pivot_table(index = 'date', columns = 'name', values = 'close') # Set names as columns, close as values with date as index

#plt.plot(ret)

ret_data = pd_data_1.pct_change()[1:]
print(ret_data.head())

weighted_returns = (weigth * ret_data)
print(weighted_returns)

port_ret = weighted_returns.sum(axis = 1)
print(port_ret)

overall_sum_port = np.sum(port_ret)
print(overall_sum_port)

ret.cov()
'''
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.hist(port_ret, bins = 60)
ax1.set_xlabel('Portfolio returns')
ax1.set_ylabel("Freq")
ax1.set_title("Portfolio Returns calculated manually")
plt.show();
'''
#n.randdom.day = random.randint(0,n.unique_days)
#unique_day = unique_days[n.randdom.day] # get entry



#new_data_median_sorted.

#len


data_cmc = pd.read_csv('crypto-markets.csv')
data_cmc['date'] = pd.to_datetime(data_cmc['date'])
data_cmc.index = pd.DatetimeIndex(data_cmc['date'])
symbol = 'bitcoin'
data_symbol = pd.DataFrame(data_cmc['close'].loc[data_cmc['name'] == symbol])
print(data_symbol)

#%% Exercise 2.c
results = {} # generate dictionary to carry information for each iteration
#for i in 1:10000{
#    results[] = results{i = i, portfolio_return = x, portfolio_volatility = y, Sharpe ratio=z}
#}

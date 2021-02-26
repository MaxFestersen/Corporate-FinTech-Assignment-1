#%% Imports
import math
import numpy as np
import statistics
import pandas as pd
import scipy.stats as scs
# from pylab import mpl, plt
import random
from matplotlib.pylab import mpl, plt
import datetime
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
top_currencies = crypto_curr.index.values # top 50 curencies - from 2.1
#random_top_curr = set(random.choices(top_currencies, k=random.randint(1,50))) # Select random number of currensies
for i in range(random.randint(0,49)):
    print("hej")
    random_top_curr = np.delete(top_currencies, random.randint(0,len(top_currencies)), 0)

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
n = 1/weigth_random_top_curr



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
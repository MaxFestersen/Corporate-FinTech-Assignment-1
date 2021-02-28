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
import webbrowser # Skal kun bruges til at vise hjemmeside til opgave 3
import scipy.optimize as sco
#import PyPortfolioOpt

#%% Design
plt.style.use('fivethirtyeight')

#%% Read data into code
data = pd.read_csv('crypto-markets.csv')
#data.describe()

new_data = data.loc[data['date'] > '2017 - 01 - 01'] # Sorterer dato, så datoer før 01-01-2017 ikke bruges :D
#new_data.info()


#%% Exercise 1
print("See external file LINK")

#%% Exercise 2

#%% Exercise 2 preperation
random.seed(42069)
results = { # generate dictionary to carry information for each iteration
    "i" : [], # Iteration
    "N": [], # Number of cryptos
    "portfolio_return" : [],
    "portfolio_volatility" : [],
    "sharpe_ratio" : []
}
results["i"].append(0) # First result is created outside of loop

# Set inital data
pd_data = new_data[['date', 'name', 'close']] # Filter to date, name and close

# Preperation to choose random date
unique_days = pd_data["date"].unique() # filter unique days in data
last_day = datetime.strptime(unique_days[-1], "%Y-%m-%d") # Get last date entry in date format (to make date calculations)
last_available_date = last_day - relativedelta(months=+6) # Get last date that is at least 6 months from end date
last_available_date = last_available_date.strftime("%Y-%m-%d") # Format as string
unique_days_filter = unique_days <= last_available_date # create filter list
unique_days = unique_days[unique_days_filter] # filter unavailable dates (not with 6 months after)
n_unique_days = len(unique_days) # count unique days


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
# Choosing a random date
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

weight_random_top_curr = len(random_top_curr) # Amount of currencsies

print("There are ", weight_random_top_curr, " currencies.\nThe random top currencies are:")
print(random_top_curr)

#%% Exercise 2.2 a filtering
# Filter the days
is_dates = pd_data['date'] >= unique_day # create filtering list from random date
pd_data = pd_data[is_dates] # Filter by the random date and after
is_dates = pd_data['date'] <= unique_day_until # create filtering list until 6 months after random date
pd_data = pd_data[is_dates] # Filter by the random date and after

# Filter by top currensies
curr_in = pd_data['name'].isin(random_top_curr) # Create filter list by random top currencies
pd_data = pd_data[curr_in] # Apply filter list

#%% Exercise 2.2 b
pd_data_1 = pd_data.pivot_table(index = 'date', columns = 'name', values = 'close') # Set names as columns, close as values with date as index
ret = np.log(pd_data_1/pd_data_1.shift(1)) # Calculate log-return of different stocks

ret = ret.dropna(axis = 1, how = 'all').dropna(axis = 0, how = 'all') # Handling missing values, drop column/row if empty
for item in random_top_curr:
    if(item not in ret.columns):
        print(item + " removed because it was empty for holding period.")

if (len(ret.columns) < weight_random_top_curr):
    print("N is now: " + str(len(ret.columns)))

weight = np.array(1/len(ret.columns)) # calculate 1/N weights

results["N"].append(len(ret.columns)) # Append Number of columns to results dictionary

weight_array = np.full((len(ret.columns), 1), weight) # Create array with weights

# Portfolio return, Portfolio volatility, and Sharpe ratio
# Portfolio volatility (for holding period)
#ret.mean() * 126 # Calculate mean of return (from half of an year)
#ret.cov() * 126 # Calculate covariance matrix for stocks

#np.sum(ret.mean().values * weight_array) * 126
#np.dot(weight_array.T, np.dot(ret.cov() * 126, weight_array))
#math.sqrt(np.dot(weight_array.T, np.dot(ret.cov() * 126, weight_array )))

def port_ret(weight, retmean):
    return np.sum(retmean * weight) * 126

def port_vol(weight, retcov):
    p_var = np.dot(
               weight.T,
               retcov * 126
           )
    #p_var = np.dot(
    #           weight_array.T,
    #           ret.cov() * 126
    #       )
    p_var = np.sqrt(p_var)
    p_var[np.isnan(p_var)] = 0
    #p_var = np.array(p_var)
    #vals = []
    #for arr in p_var:
    #    vals.append(sum(arr))
    return(p_var)

# Add to array (for ex2 c and ex2 d)
results["portfolio_return"].append(port_ret(weight, ret.mean()))
results["portfolio_volatility"].append(np.sum(port_vol(weight_array, ret.cov())))

# Sharpe ratio (for holding period)
Sharpe_Ratio = ret.mean()/ret.std()
Holding_SR = (126**0.5) * sum(Sharpe_Ratio)
results["sharpe_ratio"].append(Holding_SR)

#%% Exercise 2.2.c
#for i in range(1,10000):
for i in range(1,5):
    # Set inital data
    pd_data = new_data[['date', 'name', 'close']]  # Filter to date, name and close

    results["i"].append(i)

    # Filtering
    # Choosing a random date
    n_randdom_day = random.randint(0, n_unique_days - 1)  # Select random entry number
    unique_day = unique_days[n_randdom_day]  # get entry
    unique_day_until = datetime.strptime(unique_day, "%Y-%m-%d") + relativedelta(months=+6)  # Find 6 months after random date
    unique_day_until = unique_day_until.strftime("%Y-%m-%d")  # Format as string

    # Choose a random N of cryptos
    random_top_curr = crypto_curr.index.values  # top 50 curencies - from 2.1
    for n in range(random.randint(0, 49)):  # Randomly run from 0 to 49 times (so at least 1 value is left)
        random_top_curr = np.delete(random_top_curr, random.randint(0, len(random_top_curr) - 1), 0)  # remove a random value

    weight_random_top_curr = len(random_top_curr)  # Amount of currencsies

    # Filter the days
    is_dates = pd_data['date'] >= unique_day  # create filtering list from random date
    pd_data = pd_data[is_dates]  # Filter by the random date and after
    is_dates = pd_data['date'] <= unique_day_until  # create filtering list until 6 months after random date
    pd_data = pd_data[is_dates]  # Filter by the random date and after

    # Filter by top currensies
    curr_in = pd_data['name'].isin(random_top_curr)  # Create filter list by random top currencies
    pd_data = pd_data[curr_in]  # Apply filter list

    # Formating and calculations
    pd_data_1 = pd_data.pivot_table(index='date', columns='name', values='close')  # Set names as columns, close as values with date as index
    ret = np.log(pd_data_1 / pd_data_1.shift(1))  # Calculate log-return of different stocks

    ret = ret.dropna(axis=1, how='all').dropna(axis=0, how='all')  # Handling missing values, drop column/row if empty
    if len(ret.columns) == 0: # if there are no columns set vals as NaN
        results["N"].append(0)
        results["portfolio_return"].append(0)
        results["portfolio_volatility"].append(0)
        results["sharpe_ratio"].append(0)
    else:
        weight = np.array(1 / len(ret.columns))  # calculate 1/N weights
        results["N"].append(len(ret.columns))

        weight_array = np.full((len(ret.columns), 1), weight)

        # Portfolio return, Portfolio volatility, and Sharpe ratio
        # Portfolio return (for holding period)
        results["portfolio_return"].append(port_ret(weight, ret.mean()))

        # Portfolio volatility (for holding period)
        results["portfolio_volatility"].append(np.sum(port_vol(weight_array, ret.cov())))

        # Sharpe ratio (for holding period)
        Sharpe_Ratio = ret.mean() / ret.std()
        Holding_SR = (126 ** 0.5) * sum(Sharpe_Ratio)
        results["sharpe_ratio"].append(Holding_SR)

#%% Excersice 2.2.d
ex2 = plt.scatter(results["N"], results["portfolio_volatility"])
#ex2_return = plt.scatter(results["N"], results["portfolio_return"])
plt.show()

#%% Excersice 2.3
#webbrowser.open('https://randerson112358.medium.com/python-for-finance-portfolio-optimization-66882498847')  # Go to example.com

#%% Excersice 2.3.a
results_2_3 = { # generate dictionary to carry information for each iteration
    "i" : [], # Iteration
    "N": [], # Number of cryptos
    "portfolio_return" : [],
    "portfolio_volatility" : [],
    "sharpe_ratio" : []
}
results_2_3["i"].append(0) # First result is created outside of loop

# Set inital data
pd_data = new_data[['date', 'name', 'close']] # Filter to date, name and close

new_data_1 = new_data[['date', 'name', 'volume']] # Get data needed

# Choosing a random date
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

weight_random_top_curr = len(random_top_curr) # Amount of currencsies

print("There are ", weight_random_top_curr, " currencies.\nThe random top currencies are:")
print(random_top_curr)

# Filter the days
is_dates = pd_data['date'] >= unique_day # create filtering list from random date
pd_data = pd_data[is_dates] # Filter by the random date and after
is_dates = pd_data['date'] <= unique_day_until # create filtering list until 6 months after random date
pd_data = pd_data[is_dates] # Filter by the random date and after

# Filter by top currencies
curr_in = pd_data['name'].isin(random_top_curr) # Create filter list by random top currencies
pd_data = pd_data[curr_in] # Apply filter list

pd_data_1 = pd_data.pivot_table(index = 'date', columns = 'name', values = 'close') # Set names as columns, close as values with date as index
ret = pd_data_1/pd_data_1.shift(1) # Calculate log-return of different stocks

ret = ret.dropna(axis = 1, how = 'all').dropna(axis = 0, how = 'all') # Handling missing values, drop column/row if empty
for item in random_top_curr:
    if(item not in ret.columns):
        print(item + " removed because it was empty for holding period.")

if (len(ret.columns) < weight_random_top_curr):
    print("N is now: " + str(len(ret.columns)))

#%% Calculations for exercise
results["N"].append(len(ret.columns)) # Append Number of columns to results dictionary

# Sharpe ratio (for holding period)
Sharpe_Ratio = ret.mean()/ret.std() # Calculate sharpe ratio for each currency
Holding_SR = (126**0.5) * Sharpe_Ratio # Account for holding period
results["sharpe_ratio"].append(sum(Holding_SR))

weight = np.array(1/len(ret.columns)) # Calculate 1/N weights
weight_array = np.full((len(ret.columns), 1), weight) # Create array with weight N times

def min_func_sharpe(weight, retmean, weight_array, retcov):
    min_sharpe = -port_ret(weight, retmean) / np.sum(port_vol(weight_array, retcov))
    return(min_sharpe)

def min_func_sharpe(weight):
    min_sharpe = -port_ret(weight, ret.mean()) / np.sum(port_vol(weight_array, ret.cov()))
    return(min_sharpe)

#test = min_func_sharpe(weight, ret.mean(), weight_array, ret.cov())
cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(len(ret.columns)))

#opts = sco.minimize(min_func_sharpe, weight, ret.mean(), weight_array, ret.cov(), method='SLSQP', bounds=bnds, constraints=cons)
opts = sco.minimize(min_func_sharpe, weight, method='SLSQP', bounds=bnds, constraints=cons)

optv = sco.minimize(port_vol, weight_array,
                    method='SLSQP', bounds=bnds,
                    constraints=cons)

#returns = ret.pct_change()  # SIMPLE RETURN AND NOT LOG RETURN
#cov_matrix_semi_annual = returns.cov() * 126 # Account for holding period











# Forsøg 1
#port_variance = np.dot(weights.T, np.dot(cov_matrix_semi_annual, weights))
#port_volatility = np.sqrt(port_variance)
#portfolioSimpleSemiAnnualReturn = np.sum(returns.mean()*weights) * 126

#percent_var = str(np.round(port_variance, 2) * 100) + '%'
#percent_vols = str(np.round(port_volatility, 2) * 100) + '%'
#percent_ret = str(np.round(portfolioSimpleSemiAnnualReturn, 2)*100)+'%'
#print("Expected annual return : "+ percent_ret) # Det her er ikke rigtigt
#print('Annual volatility/standard deviation/risk : '+percent_vols) # Det her virker heller ikke rigtigt
#print('Annual variance : '+percent_var) # Det her virker heller ikke rigtigt.

#from pypfopt.efficient_frontier import EfficientFrontier
#from pypfopt import risk_models
#from pypfopt import expected_returns

#mu = expected_returns.mean_historical_return(ret) #returns.mean() * 252
#S = risk_models.sample_cov(ret) #Get the sample covariance matrix

#ef = EfficientFrontier(mu, S)

#weights = ef.max_sharpe() #Maximize the Sharpe ratio, and get the raw weights
#cleaned_weights = ef.clean_weights()

#print(cleaned_weights) #Note the weights may have some rounding error, meaning they may not add up exactly to 1 but should be close
#ef.portfolio_performance(verbose=True)
# Forsøg 1 slut

# Portfolio return, Portfolio volatility, and Sharpe ratio
def port_ret(weight, retmean):
    return np.sum(retmean * weight.T) * 126

def port_vol(weight, retcov):
    p_var = np.dot(weight,
                   np.dot(
                       weight.T,
                       retcov
                   ))
    p_var = np.sqrt(p_var)
    #vals = []
    #for arr in p_var:
    #    np.nan_to_num(arr, copy=True, nan=0, posinf=None, neginf=None)
    #    vals.append(sum(arr))
    return(p_var)

# Add to array (for ex2 c and ex2 d)
results["portfolio_return"].append(port_ret(weight_array, ret.mean()))
results["portfolio_volatility"].append(port_vol(weight_array, ret.cov()))

#%% Excersice 2.3.b


#%% Excersice 2.3.c


#%% Excersice 2.3.d
#ex3 = plt.plot(results["N"], results_ex3["portfolio_volatility"]) # En af jer må lige afprøve. Kan ikke selv få plots til at køre i pycharm. Mvh. Max.


#%% Excersice 2.4
#fig, ex4 = plt.subplots(2)
#fig.suptitle('Comparing results from 2. and 3.')
#ex4[0].ex2
#ex4[1].ex3

print("The end")

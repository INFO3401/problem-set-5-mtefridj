import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from utils import *
import datetime as dt
def correctDateFormat(df, column_name):
    temp_df = df.melt(id_vars=list(df.columns[:2]), value_vars=list(df.columns[4:]),var_name='Date',
                      value_name = column_name)
    temp_df['Date'] = pd.to_datetime(temp_df['Date'])
    return temp_df
def aggregateCountry(df, target_col, country_col):
    target_country = df.loc[df["Country/Region"] == country_col]
    return target_country.groupby(['Date', target_col], as_index=False).sum()
def topCorrelations(df, targ_col, top_n):
    list_of_countries1 = list(df['Country/Region'].unique())
    list_of_countries = []
    for z in list_of_countries1:
        if df[df['Country/Region'] == z]['Confirmed'].sum() > 500:
            list_of_countries.append(z)
    list_of_corr = {}
    for index,y in enumerate(list_of_countries):
        #print('in the first for loop')
        start_country = y
        start_country_data = aggregateCountry(df, targ_col, start_country)
        for i, x in enumerate(list_of_countries[index + 1:]):
            target_country = x
            target_country_data = aggregateCountry(df, targ_col, target_country)
            #print(target_country)
            correlation_temp = start_country_data[targ_col].corr(target_country_data[targ_col])
            list_of_corr[start_country + '/' + target_country] = correlation_temp
    sorted_list_corr = sorted(list_of_corr.values())
    best_sorted = sorted_list_corr[:top_n]
    return best_sorted
def computeTemporalLinearRegression(df, x_val, y_val):
    x_col = df[x_val].map(dt.datetime.toordinal).values.reshape(-1, 1)
    y_col = df[y_val].values.reshape(-1, 1)

    regr = LinearRegression()
    regr.fit(x_col, y_col)
    y_hat = regr.predict(y_col)
    fitScore = r2_score(y_col, y_hat)
    print("Linear Regression Fit: " + str(fitScore))
    return [regr.coef_[0][0], regr.intercept_[0]]
def runTemporalLogisticRegression(data, x, y):
    # Process the data
    x_col = data[x].map(dt.datetime.toordinal)
    y_col = data[y]

    # Give the curve a crappy fit to start with
    # In this case, we'll start with x0 as the median and define a straight
    # line between 0 and 1. The curve_fit function will adjust the line
    # to minimize the residuals.
    p0 = [np.median(x_col), 1, min(y_col)]
    params, pcov = curve_fit(logistic, x_col, y_col, p0)

    # Show the fit with the actual data in blue and the model in red. Note that
    # m = params[1] and b = params[2].
    plt.scatter(data[x], y_col, color='lightblue')
    plt.plot(data[x], logistic(x_col, params[0], params[1], params[2]), color='red', linewidth=2)
    plt.show()

    # Compute the fit using R2
    # Recall that the function is 1 - (sum of squares residuals / sum of squares total)
    residuals = y_col - logistic(x_col, params[0], params[1], params[2])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_col - np.mean(y_col))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print("Logistic Regression Fit: " + str(r_squared))

    return params
def logistic(x, x0, m, b):
    y = 1.0 / (1.0 + np.exp(-m*(x - x0) + b))
    return (y)
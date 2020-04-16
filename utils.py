import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import scipy
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sb
from scipy.optimize import curve_fit
def helloWorld():
    print("hello world")
def localandCleanData(filename):
    credit_db = pd.read_csv(filename)
    fixed_db = credit_db.fillna(0)
    return fixed_db

def computeConfidenceInterval(data):
    npArray = 1.0 * np.array(data)
    stdErr = scipy.stats.sem(npArray)
    n = len(data)
    return stdErr * scipy.stats.t.ppf((1+.95)/2.0, n -1)
def runTTest(col1, col2):
    results = scipy.stats.ttest_ind(col1,col2)
    new_results = {'T value' : results[0], 'P-value': results[1]}
    return new_results
def mergeData(df1, df2, column_name):
    new_merge = df1.merge(df2, how='right', on=column_name)
    return new_merge
def plotTimeline(df, x, y):
    ax = plt.gca()
    plt.plot(df[x], df[y])
    plt.setp(ax.get_xticklabels(), rotation=50, horizontalalignment='right')
    ax.tick_params(axis='x', which='major', pad=10)
    plt.show()
def plotMultipleTimelines(df, x_val, y_val, hue_val):
    plt.style.use('ggplot')
    df.plot.line(x=x_val, y=[y_val, hue_val], figsize=(10,7))
    plt.show()
def computeCorrelation(cand1, cand2, df):
    cand1_data = df[cand1]
    cand2_data = df[cand2]
    correlation = cand1_data.corr(cand2_data)
    return correlation

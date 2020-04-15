import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import scipy
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
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

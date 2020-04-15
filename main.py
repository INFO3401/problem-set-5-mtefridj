from utils import *
import pandas as pd
import scipy.stats
import statsmodels


covid_df = localandCleanData(r'C:\Users\btefr\Documents\GitHub\COVID-19\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_confirmed_global.csv')
print(covid_df.head())
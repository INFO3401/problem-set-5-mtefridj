from utils import *
import pandas as pd
import scipy.stats
import statsmodels
from covid import *

covid_df = localandCleanData(r'C:\Users\btefr\Documents\GitHub\COVID-19\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_confirmed_global.csv')
confirmed_df = correctDateFormat(covid_df,'Confirmed')
recovered_df_start = localandCleanData(r'C:\Users\btefr\Documents\GitHub\COVID-19\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_recovered_global.csv')
recovered_df = correctDateFormat(recovered_df_start,'Recovered')
death_df_start = localandCleanData(r'C:\Users\btefr\Documents\GitHub\COVID-19\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_deaths_global.csv')

death_df = correctDateFormat(death_df_start, 'Deaths')
#8
merge_data_other = mergeData(confirmed_df,recovered_df,['Country/Region', 'Date'])
#9
final_df = mergeData(merge_data_other, death_df,['Country/Region', 'Date'])
final_df.astype({'Confirmed' : 'int32', 'Deaths':'int32', 'Recovered': 'int32'})
print(final_df.head(20))
#11
plotTimeline(final_df,"Date", "Confirmed")
#12
#plotMultipleTimelines(final_df, "Country/Region", "Confirmed", "Date")
target_skew = aggregateCountry(final_df, "Confirmed", "United Kingdom")
#new_list = topCorrelations(final_df,'Confirmed', 5)
#print(new_list)
linear_reg_uk = computeTemporalLinearRegression(target_skew, "Date", "Confirmed")
print(linear_reg_uk)
brazil_case = aggregateCountry(final_df, 'Confirmed', 'Brazil')
linear_reg_bz = computeTemporalLinearRegression(brazil_case, "Date", "Confirmed")
print(linear_reg_bz)

logistic_curve_uk = runTemporalLogisticRegression(target_skew, 'Date', 'Confirmed')
other_uk_death = aggregateCountry(final_df, 'Deaths', 'United Kingdom')
other_uk_recovered = aggregateCountry(final_df, 'Recovered', 'United Kingdom')
logistic_curve_uk_death = runTemporalLogisticRegression(other_uk_death, 'Date', 'Deaths')
logistic_curve_uk_rec = runTemporalLogisticRegression(other_uk_recovered, 'Date', 'Recovered')

# 22
unknown_country = aggregateCountry(final_df, 'Confirmed', 'South Korea')
logistic_curve_sk = runTemporalLogisticRegression(unknown_country, 'Date', 'Confirmed')
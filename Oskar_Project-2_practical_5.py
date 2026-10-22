#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:18:31 2026

@author: Oski
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pymannkendall as mk
from scipy.signal import periodogram

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg

from scipy import stats
import statsmodels.formula.api as smf


PROJECT_DIR   = Path("/Users/Oski/Desktop/DaMod_computer_labs/week-02-data-wrangling")
DATA_DIR = PROJECT_DIR / "data" / "raw"
FIGURE_DIR    = PROJECT_DIR / "figures"

#data loading 
file_q_basel = DATA_DIR / "q_basel.csv"
output_data = pd.read_csv(file_q_basel, parse_dates=["datetime"])
output_data = output_data.set_index('datetime')
output_data.head()

discharge = output_data[['obs']]
discharge.head()

#checking for missing values 
discharge_na = discharge.isna().sum()
discharge_na.to_frame('Missing values')


#------------------------Part 1----------------------------#

#VISUAL INSPECTION#
    #plotting timeseries 
plt.plot(discharge.obs, color = "tab:blue")
plt.xlabel("Year")
plt.ylabel("Discharge m3 d-1" )
plt.title("Discharge time series")
plt.show()

#LINEAR REGRESSION#
    #preforming linear regression
    

discharge['days'] = np.arange(len(discharge))
q_val = discharge.obs.values


slope, intercept = np.polyfit(discharge["days"], q_val,1)
predicted_q = intercept + slope * discharge["days"]

discharge_model = smf.ols(
    "obs ~ days",
    data=discharge).fit()
print(discharge_model.summary())

        ##ploting regression 
plt.plot(discharge.obs, color = "tab:blue", lw = 0.5)
plt.plot(discharge.index, predicted_q, color = "black", lw =2, label = "linear regression")
plt.xlabel("Year")
plt.ylabel("Discharge m3 d-1" )
plt.title("Discharge time series")
plt.show()

        ###pearsons 

stats.pearsonr(discharge["days"],q_val)
stats.pearsonr(q_val, predicted_q)

        ####Mean discharge 
    
mean_q = discharge.obs.mean()
print(mean_q)
percent_change = slope/mean_q * 100
print(percent_change)

#NON PARAMETRIC TEST#

discharge_kendall = mk.original_test(discharge.obs)
print(f"Trend (according to MK test) = {discharge_kendall.trend}")
print(f" Significance level (according to MK test)= {discharge_kendall.p}")
print(f"Kendall Tau (according to MK test) = {discharge_kendall.Tau}")
print(f" Sen slope= {discharge_kendall.slope}")


Sen_percent_change = discharge_kendall.slope/mean_q * 100
print(Sen_percent_change)

#MOVING AVERAGES#

Q_ma365 = discharge.obs.rolling(window= 365 ,center=True).mean()
Q_ma90 = discharge.obs.rolling(window= 90 ,center=True).mean()

plt.plot(discharge.obs, color = "tab:blue", lw = 0.5, label = "Daily discharge")
plt.plot(Q_ma365, color = "red", lw = 1, label = "365-day moving average")
plt.plot(Q_ma90, color = "yellow",label = "90-day moving average", lw = 1)
plt.legend()
plt.xlabel("Year")
plt.ylabel("Discharge m3 d-1" )
plt.title("Discharge time series")
plt.show()


#PERIODOGRAM 
Q_flat = discharge.obs.values - discharge.obs.mean()
freq, power = periodogram(Q_flat, fs=1)
period = 1 / freq[1:]

fig, ax = plt.subplots(figsize = (6,4))
ax.plot(period,power[1:])
ax.set_xlim(0,1000)
plt.show()

#ACF 
plot_acf(discharge.obs, lags = 1000)
plot_acf(discharge.obs, lags = 30)


#PACF
plot_pacf(discharge.obs, lags = 6)

#Seaonal decomp
q_decompose = seasonal_decompose(discharge.obs, period=365)
fig =q_decompose.plot()



resid_line = fig.axes[-1].lines[0]  
resid_line.set_marker(".")
resid_line.set_markersize(2)

plt.tight_layout()
plt.show()


#fitting an AR model 
model_AR = AutoReg(discharge.obs, lags= 3)
results_AR = model_AR.fit()
print(results_AR.summary())
Q_fit = results_AR.fittedvalues

Q=discharge.obs.loc["1981-01-04":"2000-12-31"]
plt.plot(discharge.obs, color = "blue", label = "Original discharge")
plt.plot(Q_fit, color = "orange", label = "AR model discharge")
plt.xlabel("Discharge m3 d-1")
plt.ylabel("Time (years)")
plt.legend()
plt.show()


#Forcasting 
steps = 30
forecast = results_AR.forecast(steps=steps)
forecast.index = pd.date_range(
    start=discharge.index[-1] + pd.Timedelta(days=1),
    periods=steps,
    freq="D"
)

#plotting forecasted values 

q_first_yr = discharge.obs.loc["1999-12-31" : "2000-12-31"]
fitted_first_yr = Q_fit.loc["1999-12-31" : "2000-12-31"]

plt.plot(q_first_yr, color = "blue", label = "Original discharge")
plt.plot(fitted_first_yr, color = "orange", label = "AR model discharge")
plt.plot(forecast, color = "black", label = "AR model discharge")
plt.xlabel("Discharge m3 d-1")
plt.ylabel("Time (years)")
plt.legend()
plt.show()



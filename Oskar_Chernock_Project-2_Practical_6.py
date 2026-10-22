#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:32:16 2026

@author: Oski
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.signal import periodogram
import pymannkendall as mk
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats

# directory settings
PROJECT_DIR   = Path("/Users/Oski/Desktop/DaMod_computer_labs/week-02-data-wrangling")
DATA_DIR = PROJECT_DIR / "data" / "raw"
FIGURE_DIR    = PROJECT_DIR / "figures"

gw_raw = pd.read_csv(DATA_DIR /'GWL_Dataset.csv')

gw_raw["DATE"] = pd.to_datetime(dict(year=gw_raw["year"], month=gw_raw["month"], day=1))
gw = gw_raw.set_index("DATE")["gwl"]
gw = gw.asfreq('MS')
gw.head()

gw_pd = pd.DataFrame(data = gw)


#EXPLORING THE DATA SET
#plotting time series
plt.plot(gw, color = "tab:blue")
plt.xlabel("Year")
plt.ylabel("Ground water level m" )
plt.title("Discharge time series")
plt.show()


#calculating mean and standard deviation 

gw.mean()
gw_pd.std(ddof=1)

#preforming linear regression 

gw_pd['months'] = np.arange(len(gw_pd))
gwl = gw_pd.gwl.values


slope, intercept = np.polyfit(gw_pd["months"], gwl,1)
predicted_gwl = intercept + slope * gw_pd["months"]

#preforming annual moving averages 

gwl_annual = gw_pd.gwl.rolling(window= 12 ,center=True).mean()

#preforming biannual MA

gwl_bi= gw_pd.gwl.rolling(window= 6 ,center=True).mean()

#Mann-Kendell test trend with sens slope estimator 

gwl_kendall = mk.original_test(gw_pd.gwl)
print(f"Trend (according to MK test) = {gwl_kendall.trend}")
print(f" Significance level (according to MK test)= {gwl_kendall.p}")
print(f"Kendall Tau (according to MK test) = {gwl_kendall.Tau}")
print(f" Sen slope= {gwl_kendall.slope}")

        ##ploting regression 
plt.plot(gw_pd.gwl,color = "tab:blue", lw = 1, label = "Time series")
plt.plot(gw_pd.index, predicted_gwl, color = "black", lw =0.5, label = "Linear regression")
plt.plot(gw_pd.index, gwl_annual, color = "orange", lw = 0.5, label = "Annual rolling average")
plt.plot(gw_pd.index, gwl_bi, color = "red", lw = 0.5, label = "Bi-annual rolling average")
plt.legend()
plt.xlabel("Time (months)")
plt.ylabel("Groundwater (m)" )
plt.title("Groundwater time series")
plt.show()


#periodogram 

gwl_flat = gw_pd.gwl.values - gw_pd.gwl.mean()

freq, power = periodogram(gwl_flat, fs=1)
# Convert frequency to period (days)
period = 1 / freq[1:]

fig, ax = plt.subplots(figsize = (6,4))
ax.plot(period,power[1:], '-o')
ax.set_xlim(0,36)
plt.show()


#Seasonal decompistion 
gw_decompose = seasonal_decompose(gw_pd.gwl, period=12)
gw_decompose.plot()

deseason = gw_pd.gwl - gw_decompose.seasonal

plt.plot(gw_pd.gwl, color =  "tab:blue", lw = 1, label = "Discharge time series")
plt.plot(deseason, color =  "orange", lw = 2, label = "Discharge deseasoned")
plt.plot(gw_decompose.trend, color = "black", lw = 1, label = "Trend")
plt.xlabel("Time (years)")
plt.ylabel("Discharge m3 d-1")
plt.title("Deseasoned discharge vs discharge")
plt.legend()
plt.show()

#Dicky fuller tests 
ad_results = adfuller(gw_pd.gwl)
print(ad_results)
print("ADF test result", ad_results [0])
print("ADF pvalue", ad_results [1])
print("Number of lags", ad_results [2])

first_order_gw =gw_pd.gwl.diff()

first_order_gw = first_order_gw.dropna()
ad_diff_results = adfuller(first_order_gw)
print(ad_diff_results)
print("ADF test result", ad_diff_results[0])
print("ADF pvalue", ad_diff_results[1])
print("Number of lags", ad_diff_results[2])

print()
plt.plot(first_order_gw)
plt.xlabel("Time(years)")
plt.ylabel("Differenced discharge")
plt.show()

#Autocorrelation 
plot_acf(gw_pd.gwl, lags = 180)
plot_acf(first_order_gw, lags = 180)

plot_acf(gw_pd.gwl, lags = 10)
plot_acf(first_order_gw, lags = 10)



#Partial Autocorrelation
plot_pacf(gw_pd.gwl, lags = 10)
plot_pacf(first_order_gw, lags = 10)

#Running the ARIMA model 
model = ARIMA(gw, order=(3,1,2)) # setup the ARIMA model object
model_fit= model.fit() #fit the model to the gw-data
print(model_fit.summary()) # print out the summary of the model fitting process
gw_fit = model_fit.fittedvalues # extract modelled gw-leve


gw_fit_fix =gw_fit.loc["2002-01-01" : "2016-10-01"]

plt.plot(gw_pd.gwl)
plt.plot(gw_fit)
plt.ylim([391.8,397.2])
plt.show()

plt.figure(figsize=(12,5))
plt.plot(gw_pd.gwl,'o--', label='Observed', color='gray')
plt.plot(gw_fit, label="ARIMA fit",ls='--', color="C2", linewidth=2)

plt.xlabel("Date")
plt.ylabel("Groundwater Level")
plt.title("ARIMA Model of Groundwater Levels")
plt.legend(loc = 'upper left')
plt.ylim([391.8,397.2])
plt.grid(True)

forecast_steps = 24
forecast = model_fit.get_forecast(steps=forecast_steps)
fc_mean = forecast.predicted_mean      
fc_95 = forecast.conf_int(alpha=0.05)
fc_80 = forecast.conf_int(alpha = 0.2)

plt.figure(figsize=(12,5))
plt.plot(gw_pd.gwl,'o--', label='Observed', color='gray')
plt.plot(gw_fit, label="ARIMA fit",ls='--', color="C2", linewidth=2)
plt.plot(fc_mean.index, fc_mean, 'o--', label = "Forecast", color = 'red')
plt.fill_between(fc_95.index, fc_95.iloc[:,0], fc_95.iloc[:,1], label = "95% CI", color = 'orange', alpha = 0.3)
plt.fill_between(fc_80.index, fc_80.iloc[:,0], fc_80.iloc[:,1], label = "80% CI",  color = 'blue', alpha = 0.1)

plt.xlabel("Date")
plt.ylabel("Groundwater Level")
plt.title("ARIMA Model of Groundwater Levels")
plt.legend(loc = 'upper left')
plt.ylim([391.8,397.2])
plt.grid(True)


#SARIMA 

seasonal_order = (0,1,1,12)
sarima_1_model = ARIMA(gw, order=(3,1,2),seasonal_order = seasonal_order
)
sarima_1_model_fit = sarima_1_model.fit()
print(sarima_1_model_fit.summary())
gw_fit_sarima_1 = sarima_1_model_fit.fittedvalues

forecast_sarima_1 = sarima_1_model_fit.get_forecast(steps=forecast_steps)
fc_sarima_1 = forecast_sarima_1.predicted_mean       
fc_ci_95_sarima_1 = forecast_sarima_1.conf_int(alpha=0.05)   # 95% Confidence Interval
fc_ci_80_sarima_1 = forecast_sarima_1.conf_int(alpha=0.2) 

plt.figure(figsize=(12,5))
plt.plot(gw,'o--', label='Observed', color='gray')
plt.plot(gw_fit_sarima_1, label="SARIMA fit",ls='--', color="C2", linewidth=2)
plt.plot(fc_sarima_1.index, fc_sarima_1, 'o--',label='SARIMA Forecast', color='red')
plt.plot(fc_mean.index, fc_mean, 'o--',label='ARIMA Forecast', color='darkred')
plt.fill_between(fc_ci_95_sarima_1.index, fc_ci_95_sarima_1.iloc[:,0], fc_ci_95_sarima_1.iloc[:,1], color='pink', alpha=0.3,label='95% CI')

plt.xlabel("Date")
plt.ylabel("Groundwater Level")
plt.title("SARIMA Model of Groundwater Levels")
plt.legend(loc = 'upper left')
plt.ylim([391.8,397.2])
plt.grid(True) 

# Model evaluation 

##Setup 
forecast_steps = 24
gw_train = gw.iloc[:-forecast_steps]
gw_test = gw.iloc[-forecast_steps:]

#Model fitting
model_sarima_2 = ARIMA(gw_train, order=(3,1,2), seasonal_order=seasonal_order)
modelfit_sarima_2 = model_sarima_2.fit()
print(modelfit_sarima_2.summary())
gw_fit_sarima_2 = modelfit_sarima_2.fittedvalues

#Forecasting
forecast_sarima_2 = modelfit_sarima_2.get_forecast(steps=forecast_steps)
fc_sarima_2 = forecast_sarima_2.predicted_mean       
fc_ci_95_sarima_2 = forecast_sarima_2.conf_int(alpha=0.05)  
fc_ci_80_sarima_2 = forecast_sarima_2.conf_int(alpha=0.2)


plt.figure(figsize=(12,5))
plt.plot(gw_train,'o--', label='Observed (trained)', color='gray')
plt.plot(gw_test,'o--', label='Observed (test)', color='k')
plt.plot(gw_fit_sarima_2, label="SARIMA fit",ls='--', color="C2", linewidth=2)
plt.plot(fc_sarima_2.index, fc_sarima_2, 'o--',label='SARIMA Forecast', color='red')
plt.fill_between(fc_ci_95_sarima_2.index, fc_ci_95_sarima_2.iloc[:,0], fc_ci_95_sarima_2.iloc[:,1], color='pink', alpha=0.3,label='95% CI')

plt.xlabel("Date")
plt.ylabel("Groundwater Level")
plt.title("SARIMA Model of Groundwater Levels")
plt.legend(loc = 'upper left')
plt.ylim([391.8,397.2])
plt.grid(True)

#12 month forecast model
forecast_steps_12 = 12

gw_train_12 = gw.iloc[:-forecast_steps_12]
gw_test_12 = gw.iloc[-forecast_steps_12:]


model_sarima_12 = ARIMA(gw_train_12, order=(3, 1, 2), seasonal_order=seasonal_order)

modelfit_sarima_12 = model_sarima_12.fit()
print(modelfit_sarima_12.summary())


gw_fit_sarima_12 = modelfit_sarima_12.fittedvalues

forecast_sarima_12 = modelfit_sarima_12.get_forecast(
    steps=forecast_steps_12
)

fc_sarima_12 = forecast_sarima_12.predicted_mean
fc_ci_95_sarima_12 = forecast_sarima_12.conf_int(alpha=0.05)

#plotting validation 
plt.figure(figsize=(12, 5))

plt.plot(gw_train_12,'o--',label='Observed (trained)',color='gray')
plt.plot(gw_test_12,'o--',label='Observed (test)', color='black')
plt.plot(gw_fit_sarima_12,'--',label='SARIMA fit',color='C2',linewidth=2)
plt.plot(fc_sarima_12.index,fc_sarima_12,'o--',label='SARIMA forecast',color='red')

plt.fill_between(fc_ci_95_sarima_12.index,fc_ci_95_sarima_12.iloc[:, 0],
    fc_ci_95_sarima_12.iloc[:, 1],color='pink',alpha=0.3,label='95% CI')

plt.xlabel("Date")
plt.ylabel("Groundwater Level")
plt.title("SARIMA Model: 12-Month Forecast Validation")
plt.legend(loc='upper left')
plt.ylim([391.8, 397.2])
plt.grid(True)

#comparing 12 month vs 24 month 
plt.figure(figsize=(12, 5))
plt.plot(gw_test,'o--',lw=2,label='Observed (test)',color='black')
plt.plot(fc_sarima_2,'o--',lw=2,label='SARIMA forecast 24 month',color='red')
plt.plot(fc_sarima_12,'o--',lw=2,label='SARIMA forecast 12 month',color='green')

plt.xlabel("Date")
plt.ylabel("Groundwater Level")
plt.title("SARIMA Forecast 12 vs 24 month")
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('gw_sarima_12vs24.png',dpi=300)



#Refining model 


##Original model 
seasonal_order = (0,1,1,12)
sarima_1_model = ARIMA(gw, order=(3,1,2),seasonal_order = seasonal_order
)
sarima_1_model_fit = sarima_1_model.fit()
print(sarima_1_model_fit.summary())


##3rd model 
seasonal_order = (0,1,1,12)
sarima_3_model = ARIMA(gw, order=(3,1,1),seasonal_order = seasonal_order
)
sarima_3_model_fit = sarima_3_model.fit()
print(sarima_3_model_fit.summary())

model_sarima_3 = ARIMA(gw_train, order=(3,1,1), seasonal_order=seasonal_order)
modelfit_sarima_3 = model_sarima_3.fit()
print(modelfit_sarima_3.summary())

##4th model
seasonal_order = (0,1,1,12)
sarima_4_model = ARIMA(gw, order=(2,1,1),seasonal_order = seasonal_order
)
sarima_4_model_fit = sarima_4_model.fit()
print(sarima_4_model_fit.summary())

##4th model| Potentially right one 
seasonal_order = (0,1,1,12)
sarima_4_model = ARIMA(gw, order=(2,1,1),seasonal_order = seasonal_order
)
sarima_4_model_fit = sarima_4_model.fit()
print(sarima_4_model_fit.summary())

gw_fit_sarima_4 = sarima_4_model_fit.fittedvalues

#####4th model preformance

model_sarima_4 = ARIMA(gw_train, order=(2,1,1), seasonal_order=seasonal_order)
modelfit_sarima_4 = model_sarima_4.fit()
print(modelfit_sarima_4.summary())


##5th model 
seasonal_order = (0,1,1,12)
sarima_5_model = ARIMA(gw, order=(1,1,1),seasonal_order = seasonal_order
)
sarima_5_model_fit = sarima_5_model.fit()
print(sarima_5_model_fit.summary())

model_sarima_5 = ARIMA(gw_train, order=(1,1,1), seasonal_order=seasonal_order)
modelfit_sarima_5 = model_sarima_5.fit()
print(modelfit_sarima_5.summary())

##6th model
seasonal_order = (0,1,1,12)
sarima_6_model = ARIMA(gw, order=(0,1,1),seasonal_order = seasonal_order
)
sarima_6_model_fit = sarima_6_model.fit()
print(sarima_6_model_fit.summary())

model_sarima_6 = ARIMA(gw_train, order=(0,1,1), seasonal_order=seasonal_order)
modelfit_sarima_6 = model_sarima_6.fit()
print(modelfit_sarima_6.summary())


##7thmodel Right one 
seasonal_order = (0,1,1,12)
sarima_7_model = ARIMA(gw, order=(1,2,1),seasonal_order = seasonal_order
)
sarima_7_model_fit = sarima_7_model.fit()
print(sarima_7_model_fit.summary())

model_sarima_7 = ARIMA(gw_train, order=(1,2,1), seasonal_order=seasonal_order)
modelfit_sarima_7 = model_sarima_7.fit()
print(modelfit_sarima_7.summary())


forecast_sarima_7 = sarima_7_model_fit.get_forecast(steps=forecast_steps)
fc_sarima_7 = forecast_sarima_7.predicted_mean       
fc_ci_95_sarima_7 = forecast_sarima_7.conf_int(alpha=0.05)  
fc_ci_80_sarima_7 = forecast_sarima_7.conf_int(alpha=0.2)


plt.figure(figsize=(12, 5))
plt.plot(fc_sarima_1,'o--',lw=2,label='SARIMA forecast model 1',color='red')
plt.plot(fc_sarima_7,'o--',lw=2,label='SARIMA forecast model 7',color='green')

plt.xlabel("Date")
plt.ylabel("Groundwater Level")
plt.title("SARIMA Forecast model 1 vs 7")
plt.legend(loc='upper left')
plt.grid(True)



##8thmodel TOO FAR
seasonal_order = (0,1,1,12)
sarima_8_model = ARIMA(gw, order=(2,1,1),seasonal_order = seasonal_order
)
sarima_8_model_fit = sarima_8_model.fit()
print(sarima_8_model_fit.summary())

model_sarima_8 = ARIMA(gw_train, order=(2,1,1), seasonal_order=seasonal_order)
modelfit_sarima_8 = model_sarima_8.fit()
print(modelfit_sarima_8.summary())

forecast_sarima_8 = sarima_8_model_fit.get_forecast(steps=forecast_steps)
fc_sarima_8 = forecast_sarima_8.predicted_mean       
fc_ci_95_sarima_8 = forecast_sarima_8.conf_int(alpha=0.05)  
fc_ci_80_sarima_8 = forecast_sarima_8.conf_int(alpha=0.2)

plt.figure(figsize=(12, 5))
plt.plot(gw_test,'o--',lw=2,label='Observed (test)',color='black')
plt.plot(fc_sarima_1,'o--',lw=2,label='SARIMA forecast model 1',color='red')
plt.plot(fc_sarima_8,'o--',lw=2,label='SARIMA forecast model 8',color='green')

plt.xlabel("Date")
plt.ylabel("Groundwater Level")
plt.title("SARIMA Forecast 12 vs 24 month")
plt.legend(loc='upper left')
plt.grid(True)
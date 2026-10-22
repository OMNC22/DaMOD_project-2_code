#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 22:36:08 2026

@author: abvanwoerden
"""

# import packages
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

# directory settings
PROJECT_DIR   = Path("/Users/abvanwoerden/Documents/ESW/DaMod/project-02")
DATA_DIR = PROJECT_DIR / "data" / "raw"
FIGURE_DIR    = PROJECT_DIR / "figures"

gw_raw = pd.read_csv(DATA_DIR /'GWL_Dataset.csv')
display(gw_raw)

gw_raw["DATE"] = pd.to_datetime(dict(year=gw_raw["year"], month=gw_raw["month"], day=1))
gw = gw_raw.set_index("DATE")["gwl"]
gw = gw.asfreq('MS')
gw.head()

plt.plot(gw.index,gw,linestyle="-", marker="o", markersize=3,color="black")
plt.xlabel("Date")
plt.ylabel("Groundwater levels")
plt.grid(True)

len_gw = len(gw)
mean_gw = gw.mean()
std_gw = gw.std()
gw_na = gw.isna().sum()

print('Length of GW series: ', len_gw)
print('Mean of GW series: ', mean_gw)
print('Standard deviation of GW series: ', std_gw)
print('Number of missing observations:', gw_na)

x = np.arange(len(gw))
y = gw.values

# Linear regression (degree 1 polynomial)
slope_LR, intercept = np.polyfit(x, y, 1)
trend_LW = slope_LR * x + intercept
r2 = np.corrcoef(y, trend_LW)[0, 1]**2

print(f"Slope = {slope_LR:.6f}")
print(f"Intercept = {intercept:.3f}")
print(f"R² = {r2:.3f}")


result_mk = mk.original_test(gw)
print(result_mk)
sen_line = result_mk.intercept + result_mk.slope * x


ma6 = gw.rolling(window=6, center=True).mean() # bi-annual
ma12 = gw.rolling(window=12, center=True).mean() # annual

fig, ax = plt.subplots(figsize=(12,4))

ax.plot(gw, color='C0', lw=2, label='Measured levels')
#ax.plot(gw.index, y, label='Measured levels')
ax.plot(gw.index, trend_LW, 'C1', lw=2, label='Linear trend')
ax.plot(gw.index, sen_line,lw=2,color='C2',label=f'Sen slope ({result_mk.slope:.3f})')
ax.plot(ma6, label='Biannual', color='C3', lw=2)
ax.plot(ma12,'--', label='Annual', color='C4', lw=2)

ax.set_xlabel('Date')
ax.set_ylabel("Groundwater Level")
ax.grid(True)
ax.legend(loc='upper left')
ax.set_title('Trend Analysis')
ax.text(
    0.98, 0.3,
    f"Slope (LR) = {slope_LR:.3f}\n"
    f"Intercept = {intercept:.2f}\n"
    f"Mean  = {mean_gw:.2f}",
    transform=ax.transAxes,
    ha='right',
    va='top',
    bbox=dict(boxstyle='round,pad=0.4',facecolor='lightyellow', edgecolor='gray',alpha=0.9)
)
ax.set_xlim(gw.index.min(),gw.index.max())# Rotate labels if needed


# Remove mean
Q_flat = gw.values - gw.mean()
# Periodogram
freq, power = periodogram(Q_flat, fs=1)  # fs=1 sample/day

# Convert frequency to period (days)
period = 1 / freq[1:]
power = power[1:]

# Plot
plt.figure(figsize=(10,4))
plt.plot(period, power,'--o',c = 'coral')
plt.xlabel('Period (month)')
plt.ylabel('Power')
plt.title('Periodogram of GW levels')
plt.grid(True)
plt.xlim(0, 30)


# Classical additive decomposition
decomposition = seasonal_decompose(gw, model='additive', period=12)   # monthly data → yearly cycle
# Plot the components
fig = decomposition.plot()

# Change colors of each subplot line
colors = ['C0', 'black', 'C2', 'C1']
for ax, color in zip(fig.axes, colors):
    ax.lines[0].set_color(color)
    ax.lines[0].set_linewidth(1.5)
for ax in fig.axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# Access residual subplot (usually the last axis)
res_ax = fig.axes[-1]
res_ax.lines[0].set_markersize(2)
res_ax.lines[0].set_marker('.')
fig.suptitle('Seasonal decomposition of GW levels')


# Classical decomposition
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

deseasonalized = gw - seasonal

plt.figure(figsize=(12,5))
plt.plot(gw, color='gray', lw=1, label='Original')
plt.plot(deseasonalized, color='tomato', lw =2, label='Deseasonalized')
plt.plot(trend, color='blue', lw = 2, label='Trend')
plt.title("Groundwater Series - Original vs Deseasonalized")
plt.xlabel("Date")
plt.ylabel("Groundwater Level")
plt.legend()
plt.grid(True)

result = adfuller(gw)

adf_stat = result[0]
p_value = result[1]
used_lags = result[2]
n_obs = result[3]

print("ADF statistic:", adf_stat)
print("p-value:", p_value)
print("Used lags:", used_lags)
print("Number of observations:", n_obs)


gw_diff = gw.diff().dropna()

plt.figure(figsize=(5,4))
plt.plot(gw_diff.index,gw_diff,linestyle="-", marker="o", markersize=3,color="0.4")
plt.xlabel("Date")
plt.ylabel("Differenced Groundwater levels")
plt.title("First-order differenced groundwater series")
plt.grid(True)

result = adfuller(gw_diff)
p_value = result[1]
print("p-value:", p_value)

# ACF of original series
plt.figure(figsize=(12,4))
plot_acf(gw, lags=50)
plt.xlabel("Lag (month)")
plt.ylabel("ACF")
plt.title("ACF - Original Groundwater Series")

# PACF of original series
plt.figure(figsize=(12,4))
plot_pacf(gw, lags=20, method='ywm')  # 'ywm' is stable for PACF
plt.xlabel("Lag (month)")
plt.ylabel("PACF")
plt.title("PACF - Original Groundwater Series")


# ACF of differenced series
plt.figure(figsize=(12,4))
plot_acf(gw_diff, lags=25)
plt.xlabel("Lag (month)")
plt.ylabel("ACF")
plt.title("ACF - Differenced Groundwater Series")

# PACF of differenced series
plt.figure(figsize=(12,4))
plot_pacf(gw_diff, lags=20, method='ywm')
plt.xlabel("Lag (month)")
plt.ylabel("PACF")
plt.title("PACF - Differenced Groundwater Series")

model = ARIMA(gw, order=(3,1,2))
model_fit= model.fit()
print(model_fit.summary())
gw_fit = model_fit.fittedvalues


plt.figure(figsize=(12,5))
plt.plot(gw,'o--', label='Observed', color='gray')
plt.plot(gw_fit, label="ARIMA fit",ls='--', color="C2", linewidth=2)

plt.xlabel("Date")
plt.ylabel("Groundwater Level")
plt.title("ARIMA Model of Groundwater Levels")
plt.legend(loc = 'upper left')
plt.ylim([391.8,397.2])
plt.grid(True)

forecast_steps = 24
forecast = model_fit.get_forecast(steps=forecast_steps)
fc_mean = forecast.predicted_mean       # predicted mean groundwater levels
fc_ci_95 = forecast.conf_int(alpha=0.05)   # 95% Confidence Interval
fc_ci_80 = forecast.conf_int(alpha=0.2)   # 80% Confidence Interval


plt.figure(figsize=(12,5))
plt.plot(gw,'o--', label='Observed', color='gray')
plt.plot(gw_fit, label="ARIMA fit",ls='--', color="C2", linewidth=2)
plt.plot(fc_mean.index, fc_mean, 'o--',label='Forecast', color='red')
plt.fill_between(fc_ci_95.index, fc_ci_95.iloc[:,0], fc_ci_95.iloc[:,1], color='pink', alpha=0.3,label='95% CI')
plt.fill_between(fc_ci_80.index, fc_ci_80.iloc[:,0], fc_ci_80.iloc[:,1], color='blue', alpha=0.1,label='80% CI')
plt.xlabel("Date")
plt.ylabel("Groundwater Level")
plt.title("ARIMA Model of Groundwater Levels")
plt.legend(loc = 'upper left')
plt.ylim([391.8,397.2])
plt.grid(True)


ARIMA(
    data,
    order=(p,d,q),
    seasonal_order=(P,D,Q,s)
)


seasonal_order = (0,1,1,12)


model_sarima_1 = ARIMA(gw, order=(3,1,2), seasonal_order=seasonal_order)
modelfit_sarima_1 = model_sarima_1.fit()
print(modelfit_sarima_1.summary())
gw_fit_sarima_1 = modelfit_sarima_1.fittedvalues

forecast_sarima_1 = modelfit_sarima_1.get_forecast(steps=forecast_steps)
fc_sarima_1 = forecast_sarima_1.predicted_mean       
fc_ci_95_sarima_1 = forecast_sarima_1.conf_int(alpha=0.05)   # 95% Confidence Interval
fc_ci_80_sarima_1 = forecast_sarima_1.conf_int(alpha=0.2)   # 80% Confidence Interval


plt.figure(figsize=(12,5))
plt.plot(gw,'o--', label='Observed', color='gray')
plt.plot(gw_fit_sarima_1, label="SARIMA fit",ls='--', color="C2", linewidth=2)
plt.plot(fc_sarima_1.index, fc_sarima_1, 'o--',label='SARIMA Forecast', color='red')
plt.plot(fc_mean.index, fc_mean, 'o--',label='ARIMA Forecast', color='darkred')
plt.fill_between(fc_ci_95_sarima_1.index, fc_ci_95_sarima_1.iloc[:,0], fc_ci_95_sarima_1.iloc[:,1], color='pink', alpha=0.3,label='95% CI')
#plt.fill_between(fc_ci_80.index, fc_ci_80.iloc[:,0], fc_ci_80.iloc[:,1], color='blue', alpha=0.1,label='80% CI')
#plt.axhline(mean_gw, color="black", linestyle=":", linewidth=2, label="Long-term mean",zorder = 0)
plt.xlabel("Date")
plt.ylabel("Groundwater Level")
plt.title("SARIMA Model of Groundwater Levels")
plt.legend(loc = 'upper left')
plt.ylim([391.8,397.2])
plt.grid(True)

forecast_steps = 24

gw_train = gw.iloc[:-forecast_steps] # or specify by date: gw[:'2013-12-01']
gw_test = gw.iloc[-forecast_steps:]  # or specify by date: gw['2014-01-01':]

# gw_train = gw[:'2014-12-01']
# gw_test = gw['2015-01-01':]
# gw_train = gw[:'2014-10-01']
# gw_test = gw['2014-11-01':]
#print(gw.index[-24])


model_sarima_2 = ARIMA(gw_train, order=(3,1,2), seasonal_order=seasonal_order)#(1,1,1,12))
modelfit_sarima_2 = model_sarima_2.fit()
print(modelfit_sarima_2.summary())
gw_fit_sarima_2 = modelfit_sarima_2.fittedvalues

forecast_sarima_2 = modelfit_sarima_2.get_forecast(steps=forecast_steps)
fc_sarima_2 = forecast_sarima_2.predicted_mean       
fc_ci_95_sarima_2 = forecast_sarima_2.conf_int(alpha=0.05)   # 95% Confidence Interval
fc_ci_80_sarima_2 = forecast_sarima_2.conf_int(alpha=0.2)   # 80% Confidence Interval


plt.figure(figsize=(12,5))
plt.plot(gw_train,'o--', label='Observed (trained)', color='gray')
plt.plot(gw_test,'o--', label='Observed (test)', color='k')
plt.plot(gw_fit_sarima_2, label="SARIMA fit",ls='--', color="C2", linewidth=2)
plt.plot(fc_sarima_2.index, fc_sarima_2, 'o--',label='SARIMA Forecast', color='red')
plt.fill_between(fc_ci_95_sarima_2.index, fc_ci_95_sarima_2.iloc[:,0], fc_ci_95_sarima_2.iloc[:,1], color='pink', alpha=0.3,label='95% CI')
#plt.fill_between(fc_ci_80.index, fc_ci_80.iloc[:,0], fc_ci_80.iloc[:,1], color='blue', alpha=0.1,label='80% CI')
#plt.axhline(mean_gw, color="black", linestyle=":", linewidth=2, label="Long-term mean",zorder = 0)
plt.xlabel("Date")
plt.ylabel("Groundwater Level")
plt.title("SARIMA Model of Groundwater Levels")
plt.legend(loc = 'upper left')
plt.ylim([391.8,397.2])
plt.grid(True)


# Define a shorter validation period
forecast_steps_12 = 12

gw_train_12 = gw.iloc[:-forecast_steps_12]
gw_test_12 = gw.iloc[-forecast_steps_12:]

# Fit the SARIMA model using the shorter training dataset
model_sarima_12 = ARIMA(
    gw_train_12,
    order=(3, 1, 2),
    seasonal_order=seasonal_order
)

modelfit_sarima_12 = model_sarima_12.fit()
print(modelfit_sarima_12.summary())

# Fitted values
gw_fit_sarima_12 = modelfit_sarima_12.fittedvalues

# Twelve-month forecast
forecast_sarima_12 = modelfit_sarima_12.get_forecast(
    steps=forecast_steps_12
)

fc_sarima_12 = forecast_sarima_12.predicted_mean
fc_ci_95_sarima_12 = forecast_sarima_12.conf_int(alpha=0.05)


# Plot the 12-month validation
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


# Compare the first 12 months of the existing 24-month forecast
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






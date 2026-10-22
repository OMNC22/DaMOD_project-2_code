#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:23:29 2026

@author: abvanwoerden
"""

# import packages
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


# directory settings
PROJECT_DIR   = Path("/Users/abvanwoerden/Documents/ESW/DaMod/project-02")
DATA_DIR = PROJECT_DIR / "data" / "raw"
FIGURE_DIR    = PROJECT_DIR / "figures"
file_q_basel = DATA_DIR / "q_basel.csv"

output_data = pd.read_csv(file_q_basel, parse_dates=["datetime"])
output_data = output_data.set_index('datetime')
output_data.head()

discharge = output_data[['obs']]
discharge.head()

time = output_data[['datetime']]
time.head()

# Check for missing values
discharge_na = discharge.isna().sum()

# Visualize the results
discharge_na.to_frame('Missing values')


# Plot observed discharge
fig, ax = plt.subplots(figsize=(12, 5))
discharge.plot(ax=ax, color="steelblue", linewidth=0.6, label="Observed discharge")

ax.set_title("Observed Discharge, 1981-2000")
ax.set_xlabel("Date")
ax.set_ylabel("Discharge (m³/s)")   # adjust units to your data
ax.legend()

plt.show()

### Variables for linear regression
days = np.arange(len(discharge))
q_val = discharge.obs.values

slope, intercept = np.polyfit(days,q_val, 1)
trend = intercept + slope * days

fig, ax = plt.subplots(figsize=(12, 5))
discharge.plot(ax=ax, color="steelblue", linewidth=0.6, label="Observed discharge")
ax.plot(discharge.index, trend, color="red", linewidth=2, label="Linear trend")

ax.set_title("Observed Discharge with linear trend")
ax.set_xlabel("Date")
ax.set_ylabel("Discharge (m³/s)")   # adjust units to your data
ax.legend()

plt.show()



from scipy import stats

res = stats.linregress(days, q_val)

slope = res.slope            # m³/s per day
intercept = res.intercept
r = res.rvalue
p = res.pvalue
stderr = res.stderr

trend = slope * days + intercept
#1.3
mean_discharge = discharge.obs.mean()
percentage_change = (-0.009 / mean_discharge) * 100

print(mean_discharge)
print(percentage_change)

print(f"Slope = {slope:.3f}")
print(f"Intercept = {intercept:.3f}")
print(f"Pearson correlation coefficient: {r:.3f}")
print(f"p-value: {p:.3e}")


result = mk.original_test(discharge.obs)
print(f"Trend (according to MK Test) = {result.trend}")
print(f"Kendall's Tau = {result.Tau}")
print(f"p-value = {result.p}")
print(f"Sen's slope = {result.slope}")

#1.4
Q_ma365 = discharge.obs.rolling(window=365, center=True).mean()
Q_ma90 = discharge.obs.rolling(window=90, center=True).mean()

plt.plot(discharge.index, discharge.obs, label="Daily discharge")
plt.plot(discharge.index, Q_ma365, label="365-day moving average")
plt.plot(discharge.index, Q_ma90, label="90-day moving average")

plt.xlabel("Date")
plt.ylabel("Discharge")
plt.legend()
plt.show()


#2.1

# Remove the long-term mean
Q_flat = discharge.obs.values - discharge.obs.mean()

# Periodogram
freq, power = periodogram(Q_flat, fs=1)

# Convert frequency to period (days)
period = 1 / freq[1:]

# Find dominant period
dominant_period = period[np.argmax(power[1:])]

print(f"Dominant period = {dominant_period:.1f} days")

# Plot periodogram, limited to 1000 days
plt.plot(period, power[1:])
plt.xlim(0, 1000)
plt.xlabel("Period (days)")
plt.ylabel("Power")
plt.show()


#2.2

plot_acf(
    discharge.obs,
    lags=1095
)
plt.xlabel("Lag (days)")
plt.show()

#2.3
plot_pacf(
    discharge.obs,
    lags=20
)
plt.xlabel("Lag (days)")
plt.show()

#3
discharge_decompose = seasonal_decompose(
    discharge.obs,
    model="additive",
    period=365
)

discharge_decompose.plot()
plt.show()

#4.2
model_AR = AutoReg(discharge.obs, lags=1)
results_AR = model_AR.fit()

print(results_AR.summary())

Q_fit = results_AR.fittedvalues

plt.plot(discharge.index, discharge.obs, label="Observed discharge")
plt.plot(discharge.index[1:], Q_fit, label="AR(1) fitted discharge")

plt.xlabel("Date")
plt.ylabel("Discharge")
plt.legend()
plt.show()

#4.3
steps=30
forecast = results_AR.forecast(steps=steps)
forecast.index = pd.date_range(
    start=discharge.index[-1] + pd.Timedelta(days=1),
    periods=steps,
    freq="D"
)

plt.figure(figsize=(12,5))
plt.plot(discharge.index[-365:],discharge.obs[-365:],lw=3, label="Observed")
plt.plot(Q_fit.index[-365:], Q_fit[-365:],lw = 2, ls = '--', label="AR prediction")
plt.plot(forecast.index,forecast,c='C2',lw = 2, ls = '--', label="AR forecast")
plt.axhline(Q_fit, color="black", linestyle=":", linewidth=1, label="Long-term mean")

#plt.xlim(discharge.index[-365:].min(),forecast.index.max())# Rotate labels if needed
plt.legend(loc = 'upper left')
plt.xlabel("Time")
plt.ylabel("Discharge (m³/s)")
plt.title("Auto-regressive model forecast for discharge")



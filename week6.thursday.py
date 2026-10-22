#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:40:19 2026

@author: abvanwoerden
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, f1_score
import statsmodels.formula.api as smf

final_table = pd.read_csv(
    "/Users/abvanwoerden/Documents/ESW/DaMod/week-02-data-wrangling/data/processed/camels_gb_1991_2020_analysis_ready.csv",
    dtype={"gauge_id": str},
)
print(final_table.shape)

# %% 
final_table["flow_variability"] = final_table["Q95"] / final_table["Q5"]

# define "flashy" as the flow variability exceeding a threshold
threshold = 40
final_table["flashy"] = (final_table["flow_variability"] > threshold).astype(int)

final_table["flashy"].value_counts()


# %% 1.
climate = ["aridity", "frac_snow", "high_prec_freq"]
soils = ["conductivity_hypres", "porosity_hypres", "tawc"]
land_cover = ["urban_perc_2015", "dwood_perc_2015"]
topography = ["dpsbar", "elev_mean"]

features = climate + soils + land_cover + topography
print("Using features:", features)

model_data = final_table[features + ["flashy"]].dropna()
x = model_data[features]
y = model_data["flashy"]

# hold out 25% of catchments for testing, never used during training
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="logistic",
    max_iter=2000,
    random_state=0,
)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp_train = mlp.predict(X_train_scaled)
y_pred_mlp_test = mlp.predict(X_test_scaled)

mlp_train_accuracy = accuracy_score(y_train, y_pred_mlp_train)
mlp_test_accuracy = accuracy_score(y_test, y_pred_mlp_test)

mlp_train_f1 = f1_score(y_train, y_pred_mlp_train)
mlp_test_f1 = f1_score(y_test, y_pred_mlp_test)

mlp_confusion_train = confusion_matrix(y_train, y_pred_mlp_train)
mlp_confusion_test = confusion_matrix(y_test, y_pred_mlp_test)

print("MLP train accuracy:", mlp_train_accuracy)
print("Confusion matrix:\n", mlp_confusion_train)
print("MLP train F1 Score: ", mlp_train_f1)

print("MLP test accuracy:", mlp_test_accuracy)
print("Confusion matrix:\n", mlp_confusion_test)
print("MLP test F1 Score: ", mlp_test_f1)

# %% explore different configurations
features = climate + soils + land_cover + topography

model_data = final_table[features + ["flashy"]].dropna()
X = model_data[features]
y = model_data["flashy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="tanh",
    max_iter=2000,
    random_state=0,
)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp_train = mlp.predict(X_train_scaled)
y_pred_mlp_test = mlp.predict(X_test_scaled)

mlp_train_accuracy = accuracy_score(y_train, y_pred_mlp_train)
mlp_test_accuracy = accuracy_score(y_test, y_pred_mlp_test)

mlp_train_f1 = f1_score(y_train, y_pred_mlp_train)
mlp_test_f1 = f1_score(y_test, y_pred_mlp_test)

mlp_confusion_train = confusion_matrix(y_train, y_pred_mlp_train)
mlp_confusion_test = confusion_matrix(y_test, y_pred_mlp_test)

print("MLP train accuracy:", mlp_train_accuracy)
print("Confusion matrix:\n", mlp_confusion_train)
print("MLP train F1 Score: ", mlp_train_f1)

print("MLP test accuracy:", mlp_test_accuracy)
print("Confusion matrix:\n", mlp_confusion_test)
print("MLP test F1 Score: ", mlp_test_f1)

# %% 2.


hidden_sizes = [1, 2, 4, 8, 16]

complexity_rows = []
for size in hidden_sizes:
    model = MLPClassifier(hidden_layer_sizes=(size,), activation="tanh", max_iter=2000, random_state=0)
    model.fit(X_train_scaled, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
    test_accuracy  = accuracy_score(y_test, model.predict(X_test_scaled))

    train_f1 = f1_score(y_train, model.predict(X_train_scaled), zero_division=0)
    test_f1  = f1_score(y_test, model.predict(X_test_scaled), zero_division=0)

    complexity_rows.append({
        "hidden_size": size,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_F1": train_f1,
        "test_F1": test_f1,
    })

complexity_df = pd.DataFrame(complexity_rows)
print(complexity_df)

fig_mlp_accuracy = plt.figure(figsize=(6, 4))
plt.plot(complexity_df["hidden_size"], complexity_df["train_accuracy"], marker="o", label="Train accuracy")
plt.plot(complexity_df["hidden_size"], complexity_df["test_accuracy"], marker="o", label="Test accuracy")
plt.xscale("log")
plt.xlabel("Hidden layer size")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1.0])
plt.title("MLP complexity curve")
plt.legend()
plt.grid()
plt.show()

fig_mlp_f1 = plt.figure(figsize=(6, 4))
plt.plot(complexity_df["hidden_size"], complexity_df["train_F1"], marker="o", label="Train F1")
plt.plot(complexity_df["hidden_size"], complexity_df["test_F1"], marker="o", label="Test F1")
#plt.xscale("log")
plt.xlabel("Hidden layer size")
plt.ylabel("F1 score")
plt.title("MLP complexity curve")
#plt.ylim([.5,1])
plt.legend()
plt.grid()
plt.show()


# %% 3.


rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=0,
)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

rf_test_accuracy = accuracy_score(y_test, y_pred_rf)
rf_F1 = f1_score(y_test, y_pred_rf)
print("Test accuracy:", rf_test_accuracy)
print("Test F1:", rf_F1)


importances = pd.Series(rf_clf.feature_importances_, index=features).sort_values()

fig_rf_importances = plt.figure(figsize=(6, 4.5))
importances.plot.barh(color="tab:blue")
plt.xlabel("Feature importance")
plt.title("Random Forest: which attributes drive flashiness?")
plt.show()



max_depths = [1, 2, 3, 4, 6, 8, 12, None]

rf_complexity_rows = []
for depth in max_depths:
    model = RandomForestClassifier(n_estimators=300, max_depth=depth, random_state=0)
    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy  = accuracy_score(y_test, model.predict(X_test))
    train_f1 = f1_score(y_train, model.predict(X_train), zero_division=0)
    test_f1  = f1_score(y_test, model.predict(X_test), zero_division=0)

    rf_complexity_rows.append({
        "max_depth": depth if depth is not None else 20,  # plot "None" as an unbounded depth
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_F1": train_f1,
        "test_F1": test_f1
    })

rf_complexity_df = pd.DataFrame(rf_complexity_rows)

fig_rf_accuracy = plt.figure(figsize=(6, 4))
plt.plot(rf_complexity_df["max_depth"], rf_complexity_df["train_accuracy"], marker="o", label="Train accuracy")
plt.plot(rf_complexity_df["max_depth"], rf_complexity_df["test_accuracy"], marker="o", label="Test accuracy")
plt.xlabel("Max tree depth (20 = unbounded)")
plt.ylabel("Accuracy")
plt.title("Random Forest complexity curve")
plt.legend()
plt.grid()
plt.show()


fig_rf_f1 = plt.figure(figsize=(6, 4))
plt.plot(rf_complexity_df["max_depth"], rf_complexity_df["train_F1"], marker="o", label="Train F1")
plt.plot(rf_complexity_df["max_depth"], rf_complexity_df["test_F1"], marker="o", label="Test F1")
plt.xlabel("Max tree depth (20 = unbounded)")
plt.ylabel("F1")
plt.title("Random Forest complexity curve")
plt.legend()
plt.grid()
plt.show()

# %% 4.

reg_data = final_table[features + ["runoff_ratio"]].dropna()
X_reg = reg_data[features]
y_reg = reg_data["runoff_ratio"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.25, random_state=0
)

rf_reg = RandomForestRegressor(n_estimators=300, random_state=0)
rf_reg.fit(X_reg_train, y_reg_train)

y_reg_pred = rf_reg.predict(X_reg_test)
rf_reg_test_r2 = r2_score(y_reg_test, y_reg_pred)
print("Test R2:", rf_reg_test_r2)


# Observed vs predicted scatter
fig_rf_regression = plt.figure(figsize=(5.5, 5.5))
plt.scatter(y_reg_test, y_reg_pred, s=25, alpha=0.6, color="tab:blue")
lims = [min(y_reg_test.min(), y_reg_pred.min()), max(y_reg_test.max(), y_reg_pred.max())]
plt.plot(lims, lims, "k--", label="1:1 line")
plt.xlabel("Observed runoff ratio")
plt.ylabel("Predicted runoff ratio (Random Forest)")
plt.legend()
plt.show()







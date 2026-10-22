#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:02:56 2026

@author: Oski
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, f1_score
import statsmodels.formula.api as smf

final_table = pd.read_csv("/Users/Oski/Desktop/DaMod_computer_labs/week-02-data-wrangling/data/processed/camels_gb_1991_2020_analysis_ready.csv",
    dtype={"gauge_id": str},
)

#############################Part 1############################################
#Defining the labels 
final_table["flow_variability"] = final_table["Q95"] / final_table["Q5"]


threshold = 40
final_table["flashy"] = (final_table["flow_variability"] > threshold).astype(int)
final_table["flashy"].value_counts()


#Defining features
feature = ["pet_mean", "porosity_hypres", "dwood_perc_2015", "ewood_perc_2015", "elev_mean"]


#Splitting data into training and testing data 
model_data = final_table[feature + ["flashy"]].dropna()
x = model_data[feature]
y = model_data["flashy"]


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0, stratify=y
)
##standardizing data 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#fitting training and testing model + results 
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


#Fitting of the second model with changes 
feature = ["p_mean", "t_mean", "porosity_hypres", "soil_depth_pelletier" ,"dwood_perc_2015", "ewood_perc_2015", "elev_mean", "area"]


model_data = final_table[feature + ["flashy"]].dropna()
x = model_data[feature]
y = model_data["flashy"]


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


mlp = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="relu",
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
#############################Part 2############################################

hidden_sizes = [1, 2, 4, 8, 16, 24,60]

complexity_rows = []
for size in hidden_sizes:
    model = MLPClassifier(hidden_layer_sizes=(size,), activation="relu", max_iter=2000, random_state=0)
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

#plotting 
fig_mlp_accuracy = plt.figure(figsize=(6, 4))
plt.plot(complexity_df["hidden_size"], complexity_df["train_accuracy"], marker="o", label="Train accuracy")
plt.plot(complexity_df["hidden_size"], complexity_df["test_accuracy"], marker="o", label="Test accuracy")
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

################################Part 3
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

importances = pd.Series(rf_clf.feature_importances_, index=feature).sort_values()

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
        "max_depth": depth if depth is not None else 20, 
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

################################Part 4#######################################
 
regression_data = final_table[feature+ ["runoff_ratio"]].dropna()
X_regression = regression_data[feature]
y_regression = regression_data["runoff_ratio"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_regression, y_regression, test_size=0.25, random_state=0
)

rf_reg = RandomForestRegressor(n_estimators=300, random_state=0)
rf_reg.fit(X_reg_train, y_reg_train)

y_reg_pred = rf_reg.predict(X_reg_test)
rf_reg_test_r2 = r2_score(y_reg_test, y_reg_pred)
print("Test R2:", rf_reg_test_r2)


fig_rf_regression = plt.figure(figsize=(5.5, 5.5))
plt.scatter(y_reg_test, y_reg_pred, s=25, alpha=0.6, color="tab:blue")
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], "k--", label="1:1 line")
plt.xlabel("Observed runoff ratio")
plt.ylabel("Predicted runoff ratio (Random Forest)")
plt.title("Observed vs Predicted runoff ratio from the Random Forest model")
plt.show()



model2_data = X_reg_train.copy()
model2_data["runoff_ratio"] = y_reg_train

model_x_train = smf.ols(
    formula="runoff_ratio ~ " + " + ".join(X_reg_train.columns),
    data=model2_data
).fit()

print(model_x_train.summary())

# The formula API requires a DataFrame for prediction
X_reg2_test_df = X_reg_test.copy()
y_pred = model_x_train.predict(X_reg2_test_df)

r2_x_test = r2_score(y_reg_test, y_pred)
print("Test R²:", r2_x_test)

fig_ols_regression = plt.figure(figsize=(5.5, 5.5))
plt.scatter(y_reg_test, y_pred, s=25, alpha=0.6, color="tab:blue")
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], "k--", label="1:1 line")
plt.xlabel("Observed runoff ratio")
plt.ylabel("Predicted runoff ratio (OLS multiple regression)")
plt.title("Observed vs Predicted runoff ratio from multiple linear regression")
plt.show()
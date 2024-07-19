import numpy as np
from numpy import sqrt
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB

battery_df = pd.read_csv("Battery_RUL.csv")
feature = battery_df.drop(columns=["RUL", "Cycle_Index"], axis=1)
target = battery_df["RUL"]
##scaling data
scale_factor = StandardScaler()
feature_scaled = scale_factor.fit_transform(feature)
feature_df = pd.DataFrame(feature_scaled, columns=feature.columns)


def tuning_rf(
    estimator: int,
    criterion: str,
    test_size: float,
    max_depth: int,
    min_samples_split: int | float,
    min_samples_leaf: int | float,
    min_weight_fraction_leaf: float,
    max_features: int | float | str,
):
    x_train, x_test, y_train, y_test = train_test_split(
        feature, target, train_size=test_size
    )
    model = RandomForestRegressor(
        n_estimators=estimator,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        bootstrap= False, # use entire dataset,
        verbose= 1
    )
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    mse = mean_squared_error(y_test, predict)
    mae = mean_absolute_error(y_test, predict)
    rmse = sqrt(mse)
    score_val = np.mean(
        cross_val_score(model, x_train, y_train, cv=5, scoring="r2"),
    )
    # residuals = y_test - predict
    return (mse, mae, rmse, score_val)


def pipeline_tuning():
    val_dict = {
        "tunning_parameters": [],
        "mse": [],
        "mae": [],
        "rmse": [],
        "avg R^2 score": [],
    }
    criterion = [8,8,8,8,8,8,8,8,8,8,8,8]
    for i in criterion:
        metrics = tuning_rf(
            estimator=810,
            criterion="squared_error",
            max_depth=1000,
            test_size=0.2,
            min_samples_split=6,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=8,
        )
        val_dict["tunning_parameters"].append(i)
        val_dict["mse"].append(metrics[0])
        val_dict["mae"].append(metrics[1])
        val_dict["rmse"].append(metrics[2])
        val_dict["avg R^2 score"].append(metrics[3])
    val_df = pd.DataFrame(val_dict).sort_values(by="avg R^2 score", ascending=0)
    return val_df


pipeline_tuning()


# n = 810 -> 0.989
# criterion: either poisson (0.986) or squared error (0.985)
# max_depth: None actually gives lower R^2, 1000 gives 0.9888
# min_samples_split: going above 100 decrease R2 alot. 6 gives the best R2 0.98887
# min_samples_leaf: 1 gives the best R2
# min_weight_fraction_leaf: 0.0 gives the best and reduce as it is larger than 0.2
#max_features : 8 gives the best. 0.1 < float < 1.0 is similar to 1.0. higher than 8 reduce r2. log and sqrt2 shows lower r2 score than 8

#!TODO: use gridsearchcv and randomizedsearchcv
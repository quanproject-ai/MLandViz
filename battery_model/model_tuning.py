import numpy as np
from numpy import sqrt
import pandas as pd
from pandas import DataFrame
import data_preprocess as dp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    make_scorer,
    r2_score,
)
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB


####manual tunnig for Randomforestregressor
def get_performance_metrics_from_model(
    x_train,
    x_test,
    y_train,
    y_test,
    method,
) -> tuple:
    """
    Return the performance metric of a model
    Args:
        x_train : feature data set for training
        x_test : feature data set for testing
        y_train : target data set for training
        y_test : target data set for testing
        method  : ML models from sklearn

    Returns:
        tuple: tuple of the performance metrics in this order (mse,mae,rmse,score_val)
    """
    model = method
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    mse = mean_squared_error(y_test, predict)
    mae = mean_absolute_error(y_test, predict)
    rmse = sqrt(mse)
    score_val = np.mean(
        cross_val_score(model, x_train, y_train, cv=5, scoring="r2"),
    )
    return (mse, mae, rmse, score_val)


def manually_pipline_tuning(
    method, preset_params: dict, tuning_name: dict, tuning_params: list
) -> DataFrame:
    """Function tune model manually without using builtin sklearn functions. This function is meant to get an understand
    of how each model's parameters contribute to the performance metrics.

    Args:
        method: sklearn ML models
        preset_params (dict): preset params of ML models
        tuning_name (dict): the name of the tuning parameters
        tuning_params (list): the list of items applying for that model

    Returns:
        DataFrame: A df table shows each tuning parameters its associated performance metrics
    """
    x_train, x_test, y_train, y_test = dp.split_train_test_set(
        dp.feature, dp.target, size=0.2
    )
    key_tuning_name = next(iter(tuning_name))  # get the key of the tuning name
    val_dict = {
        key_tuning_name: [],
        "mse": [],
        "mae": [],
        "rmse": [],
        "avg R^2 score": [],
    }
    for value in tuning_params:
        tuning_name[key_tuning_name] = value
        preset_params.update(tuning_name)
        print("getting results for:", method(**preset_params))
        metrics = get_performance_metrics_from_model(
            x_train, x_test, y_train, y_test, method=method(**preset_params)
        )
        val_dict[key_tuning_name].append(value)
        val_dict["mse"].append(metrics[0])
        val_dict["mae"].append(metrics[1])
        val_dict["rmse"].append(metrics[2])
        val_dict["avg R^2 score"].append(metrics[3])
    val_df = pd.DataFrame(val_dict).sort_values(by="avg R^2 score", ascending=0)
    return val_df


###tuning using built-in methods from scikit learn
def grid_tuning(grid: dict, score: dict, model, refit_var: str) -> DataFrame:
    """Use gridsearchcv to tune a model with specific parameters and scoring metrics

    Args:
        grid (dict): parameters that need tuning in format dict(key:[value])
        score (dict): parameters that need tuning in format dict(key:sklearn scorer(type of error/score))
        model: sklearn model class
        refit_var (str): refit key mentioned in the keys in score dictionary

    Returns:
        DataFrame : DataFrame from GridSearchCV giving performance of each parameters 
    """
    x_train, x_test, y_train, y_test = dp.split_train_test_set(
        dp.feature, dp.target, 0.2
    )
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring=score,
        refit=refit_var,
        n_jobs=-1,
        cv=6,
        verbose=0,
    )
    grid_search.fit(x_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    df_result = pd.DataFrame(grid_search.cv_results_).sort_values(
        by="mean_test_R2", ascending=False
    )
    return df_result


# Note: RandomSearchGrid below give this param as the highest:
# Standard no parameters in RandomForest
# Best parameters: {'random_state': 100, 'n_estimators': 10, 'min_weight_fraction_leaf': 0.0, 'min_samples_split': 6, 'min_samples_leaf': 1, 'max_features': 6, 'max_depth': None, 'criterion': 'poisson'}
# Best score: 0.9820621506303263
# param_grid = {
#         "n_estimators": [10, 100, 810, 1000],w
#         "criterion": ["poisson", "squared_error", "absolute_error"],
#         "max_depth": [None, 1000, 100, 2000],
#         "min_samples_split": [2, 5, 6, 10, 100],
#         "min_samples_leaf": [1, 2, 4, 6],
#         "min_weight_fraction_leaf": [0.0, 0.1],
#         "max_features": [0.5, 1, 6, 8, 12, "log"],
#         "random_state": [None, 10, 100, 1000],
#     }


# Note: RandomSearchGrid below give this param as the highest (2nd run)
# Random forest params :  criterion="absolute_error", max_depth=None, random_state=None, min_weight_fraction_leaf=0.0
# Best parameters: {'n_estimators': 1000, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 8}
# Best score is 0.9855
# param_grid = {
#         "n_estimators": [100, 810, 1000],
#         "min_samples_split": [2, 5, 6],
#         "min_samples_leaf": [2, 4,6],
#         "max_features": [
#             0.5,
#             6,
#             8,
#         ],
#     }

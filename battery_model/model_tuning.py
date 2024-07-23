import numpy as np
from numpy import sqrt
import pandas as pd
import datapreprocess as dp
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

battery_df = pd.read_csv("Battery_RUL.csv")
feature = battery_df.drop(columns=["RUL", "Cycle_Index"], axis=1)
target = battery_df["RUL"]


####manual tunnig for Randomforestregressor
def get_performance_metrics_from_model(method, test_size: float) -> tuple:
    """
    Return the performance metric of a model
    Args:
        method (class): ML models from sklearn
        test_size (float): percent of test size as float

    Returns:
        tuple: tuple of the performance metrics in this order (mse,mae,rmse,score_val)
    """
    x_train, x_test, y_train, y_test = train_test_split(
        feature, target, train_size=test_size
    )
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


def pipeline_tuning(method, preset_params:dict, params:list):
    """Function tunes model one parameters at a time.

    Args:
        method (_type_): _description_
        preset_paremeters (dict): _description_

    Returns:
        _type_: _description_
    """
    val_dict = {
        "tunning_parameters": [],
        "mse": [],
        "mae": [],
        "rmse": [],
        "avg R^2 score": [],
    }
    for key,value in preset_params.items():
        if value is None:
            for i in params:
                preset_params.update({key:i})
                metrics = get_performance_metrics_from_model(
                    method=method(**preset_params), test_size=0.2
                )
                val_dict["tunning_parameters"].append(i)
                val_dict["mse"].append(metrics[0])
                val_dict["mae"].append(metrics[1])
                val_dict["rmse"].append(metrics[2])
                val_dict["avg R^2 score"].append(metrics[3])
    val_df = pd.DataFrame(val_dict).sort_values(by="avg R^2 score", ascending=0)
    return val_df

method_params = {
    'n_estimators':800,
    'criterion' :None,
}
tuning_params = ["squared_error", "absolute_error", "friedman_mse", "poisson"]


pipeline_tuning(method=RandomForestRegressor,preset_params=method_params,params= tuning_params)



# n = 810 -> 0.989
# criterion: either poisson (0.986) or squared error (0.985)
# max_depth: None actually gives lower R^2, 1000 gives 0.9888
# min_samples_split: going above 100 decrease R2 alot. 6 gives the best R2 0.98887
# min_samples_leaf: 1 gives the best R2
# min_weight_fraction_leaf: 0.0 gives the best and reduce as it is larger than 0.2
# max_features : 8 gives the best. 0.1 < float < 1.0 is similar to 1.0. higher than 8 reduce r2. log and sqrt2 shows lower r2 score than 8


###tuning using built-in methods from scikit learn
def grid_tuning_rf(test_size: float):
    x_train, x_test, y_train, y_test = train_test_split(
        feature, target, train_size=test_size
    )
    scoring_dict = {
        "MSE": make_scorer(mean_squared_error, greater_is_better=False),
        "R2": make_scorer(r2_score),
    }
    param_grid = {
        "min_samples_split": [2, 6],
    }
    rf = RandomForestRegressor(
        criterion="absolute_error",
        n_estimators=800,
        max_depth=None,
        random_state=None,
        min_weight_fraction_leaf=0.0,
        max_features=7,
        min_samples_leaf=1,
    )
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scoring_dict,
        refit="R2",
        n_jobs=-1,
        cv=6,
        verbose=1,
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
#         "n_estimators": [10, 100, 810, 1000],
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

import numpy as np
from numpy import sqrt
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

## a lot of ML Models
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    BaggingRegressor,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import (
    LassoLarsCV,
    SGDRegressor,
    LogisticRegression,
    LinearRegression,
)
from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    BaggingRegressor,
    VotingRegressor,
)
from sklearn.naive_bayes import GaussianNB

#### assuming that this is a raw dataset without prior sanitization
battery_df = pd.read_csv("Battery_RUL.csv")
feature = battery_df.drop(columns=["RUL", "Cycle_Index"], axis=1)
target = battery_df["RUL"]
##scaling data
scale_factor = StandardScaler()
feature_scaled = scale_factor.fit_transform(feature)
feature_df = pd.DataFrame(feature_scaled, columns=feature.columns)


def modeling_with_entire_dataset(algorithm, test_size: float) -> tuple:
    """Input a ML model and return performance metrics
    Metrics: mean absolute error, mean squared error, root mean squared error, residuals, predict array)

    Args:
        algorithm (class) :sci-kit learn ML
        test_size (float): test size range from 0.0 to 1.0

    Returns:
        tuple: tuple of performance metrics in order (mse, mae, rmse, score valuation, residuals and predict)
    """
    # train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        feature_df, target, test_size=test_size
    )
    # train model
    model = algorithm
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    mse = mean_squared_error(y_test, predict)
    mae = mean_absolute_error(y_test, predict)
    rmse = sqrt(mse)
    score_val = np.mean(
        cross_val_score(model, x_train, y_train, cv=5, scoring="r2"),
    )
    residuals = y_test - predict
    return (mse, mae, rmse, score_val, residuals, predict)




def apply_multiple_model(algorithm: list, test_size: float) -> DataFrame:
    """Input a list of algorithms and a return a summary table has all of performance metrics and time trained
    Also, there is a plot of residual vs. predicted value per algorithm

    Args:
        algorithm (list): A list of sci-kit learn ML algorithms
        test_size (float): test size range from 0.0 to 1.0

    Returns:
        DataFrame: a summary table has mean absolute error, mean squared error, root mean squared error, time
    """
    model_dict = {
        "Model Name": [],
        "mse": [],
        "mae": [],
        "rmse": [],
        "average score validation R^2 of 5": [],
        "traing time elapsed": [],
    }
    for alg in algorithm:
        start_time = time.time()
        try:
            alg_name = alg.__name__
        except AttributeError: #certain class of ML does not have __name__
            alg_name = alg
        model_dict["Model Name"].append(alg_name)
        (mse, mae, rmse, score_val, residuals, predict) = (
            modeling_with_entire_dataset(alg, test_size)
        )
        _plot_residual_analysis(alg_name, x=predict, y=residuals)
        model_dict["mse"].append(mse)
        model_dict["mae"].append(mae)
        model_dict["rmse"].append(rmse)
        model_dict["average score validation R^2 of 5"].append(score_val)
        end_time = time.time()
        model_dict["traing time elapsed"].append(end_time - start_time)
    report_df = pd.DataFrame(model_dict).sort_values(
        by="average score validation R^2 of 5", ascending=0
    )
    return report_df


def voting_regressors(x, y,test_size):
    """Generate 1 plot of predicted value vs. training samples, and 4 plots of each regressor algorithms used in this fucntion

    Args:
        x (DataFrame): feature df
        y (ndarray): target array

    Returns:
        DataFrame: A summary table includes performance metrics (mae, mse, rmse, avg r^2 and time)
    """
    clf1 = GradientBoostingRegressor()
    clf2 = RandomForestRegressor()
    clf3 = LinearRegression()
    clf1.fit(x, y)
    clf2.fit(x, y)
    clf3.fit(x, y)
    voting_clf = VotingRegressor(estimators=[("gb", clf1), ("rf", clf2), ("lr", clf3)])
    voting_clf.fit(x, y)
    x_test = x.sample(frac=test_size)
    pred_clf1 = clf1.predict(x_test)
    pred_clf2 = clf2.predict(x_test)
    pred_clf3 = clf3.predict(x_test)
    pred_clf4 = voting_clf.predict(x_test)
    _plot_voting_regressors(pred_clf1, pred_clf2, pred_clf3, pred_clf4)
    report_df = apply_multiple_model([clf1,clf2,clf3,voting_clf],test_size)
    return report_df

##Helper function##
def _plot_residual_analysis(algorithm, x, y):
    """Helper function to make code cleaner to plot residual analysis of an algorithm

    Args:
        algorithm (class): Algorithm of sci-kit learn
        x (ndarray): predicted value
        y (ndarray): residual value
    """
    plt.scatter(x, y)
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title(f"Residual plot of {algorithm}")
    plt.show()


def _plot_voting_regressors(model_1, model_2, model_3, vote_model):
    """Helper function to make code cleaner to plot voting regressors

    Args:
        model_1 (class): GradientBoostingRegressor
        model_2 (class): RandomForestRegressor
        model_3 (class): LinearRegression
        vote_model (class): VotingRegressor
    """
    plot = plt.figure()
    plt.plot(model_1, "gd", label="GradientBoostingRegressor")
    plt.plot(model_2, "b^", label="RandomForestRegressor")
    plt.plot(model_3, "ys", label="LinearRegression")
    plt.plot(vote_model, "r*", ms=10, label="VotingRegressor")

    plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    plt.ylabel("predicted value")
    plt.xlabel("test sample value")
    plt.legend(loc="best")
    plt.title("Regressor predictions and their average")
    plt.show()




# print(apply_multiple_model(
#     algorithm=[
#         LassoLarsCV(),
#         GaussianNB(),
#         DecisionTreeRegressor(),
#         ExtraTreeRegressor(),
#         KNeighborsRegressor(),
#         SGDRegressor(),
#         # LogisticRegression(),
#         # RandomForestRegressor(),
#         # AdaBoostRegressor(),
#         # GradientBoostingRegressor(),
#         # BaggingRegressor(),
#     ],
#     test_size=0.2,
# )

(voting_regressors(x=feature_df, y=target,test_size=0.1))

#!Todo: paremeters tuning of the best 3 models, data imbalance SMOTE, data processing (how to create fake data to account), feature construction
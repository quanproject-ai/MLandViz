# import numpy as np
import pandas as pd
from pandas import DataFrame, arrays
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
)
from imblearn.over_sampling import SMOTE, RandomOverSampler,ADASYN, BorderlineSMOTE, KMeansSMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler,ClusterCentroids

battery_df = pd.read_csv("Battery_RUL.csv")


def handle_na_in_df(df: DataFrame, how: str) -> DataFrame:
    """Handle NA values in df

    Args:
        df (DataFrame): data df
        how (str): options are ffill, bfill or numeric value

    Returns:
        DataFrame: df with na value filled with specific method
    """
    df = df.fillna(how).copy
    return df


def scale_df(df: DataFrame, method) -> DataFrame:
    """Apply sklearn scaling method into df of choice

    Args:
        df (DataFrame): feature df or target
        method (sklearn class): choices are MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

    Returns:
        DataFrame: scaled df
    """
    scale_factor = method
    feature_scale = scale_factor.fit_transform(df)
    df_scaled = pd.DataFrame(feature_scale, columns=df.columns)
    return df_scaled

def apply_oversample_techniques(method,x_features:DataFrame, y_target:arrays) -> tuple:
    """Initialize a SMOTE model for handling oversampling issue

    Args:
        method: options are SMOTE, RandomOverSampler,ADASYN, BorderlineSMOTE, KMeansSMOTE
        x_features: DataFrame of features
        y_target: arrays of target

    Returns:
       Tuple : A tuple contains 2 dataframes from feature and target
    """
    technique = method
    x_res, y_res = technique.fit_resample(x_features,y_target)
    return (x_res, y_res)

battery_df
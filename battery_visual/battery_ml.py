import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

SINGLETESTFILEPATH = (
    "G:\\All Coding Project\\machinelearning\\dataset\\timeseriescell\\00001.csv"
)
METADATAFILEPATH = "G:\\All Coding Project\\machinelearning\\dataset\\celldata.csv"
BATTERY_GROUP = {
    "1": ["B0025", "B0026", "B0027", "B0028"],
    "2b": ["B0029", "B0030", "B0031", "B0032"],
    "2c": ["B0033", "B0034", "B0036"],
    "2d": ["B0038", "B0039", "B0040"],
    "2e": ["B0041", "B0042", "B0043", "B0044"],
    "3": ["B0045", "B0046", "B0047", "B0048"],
    "4": ["B0049", "B0050", "B0051", "B0052"],
    "5": ["B0053", "B0054", "B0055", "B0056"],
    "6": ["B0005", "B0006", "B0007", "B0018"],
}
# https://www.kaggle.com/code/susanketsarkar/rul-prediction-using-variational-autoencoder-lstm
metadata_df = pd.read_csv(filepath_or_buffer=SINGLETESTFILEPATH)

train,test = train_test_split(metadata_df, test_size=0.5, random_state=10)
drop_cols = ['Current_load','Voltage_load','Temperature_measured']

train_Y = train['Temperature_measured'] #only has column temperature measure
train_X = train.drop(drop_cols, axis=1) #has voltage_measure, current_measure and time


test_Y = test['Temperature_measured']
test_X = test.drop(drop_cols, axis =1)


## RandomForestRegressor ##
# model = RandomForestRegressor(random_state=0)
# model.fit(train_X,train_Y)
# preds = model.predict(test_X)
# mae = mean_absolute_error(test_Y, preds)
# r2 = r2_score(test_Y,preds)
# print(f"MAE: {mae}")
# print(f"R2S: {r2}")
# feature_imp = model.feature_importances_
# indices = np.argsort(feature_imp)[::-1]
# for f in range(train_X.shape[1]):
#     print(f"{train_X.columns[indices[f]]}: {feature_imp[indices[f]]}")

## Support Vector Regression ##
# model = SVR()
# model.fit(train_X, train_Y)
# preds = model.predict(test_X)
# mae = mean_absolute_error(test_Y, preds)
# r2 = r2_score(test_Y,preds)
# print(f"MAE: {mae}")
# print(f"R2S: {r2}")

## Regression based on k-nearest neighbors ##
# model = KNeighborsRegressor()
# model.fit(train_X, train_Y)
# preds = model.predict(test_X)
# mae = mean_absolute_error(test_Y, preds)
# r2 = r2_score(test_Y,preds)
# print(f"MAE: {mae}")
# print(f"R2S: {r2}")

#!TODO: - loop through all of the files and separate among charge, discharge and impedance.
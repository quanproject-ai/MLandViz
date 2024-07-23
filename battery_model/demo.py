import model_selection as ms
import model_tuning as mt
import data_preprocess as dp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    make_scorer,
    r2_score,
)


##demo of manual tuning#
preset = {
    "criterion": "absolute_error",
}
variable = {"n_estimators": None}
variable_params = [1, 10]
print(
    mt.manually_pipline_tuning(
        method=RandomForestRegressor,
        preset_params=preset,
        tuning_name=variable,
        tuning_params=variable_params,
    )
)

##demo gridsearch tuning
scoring_dict = {
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
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
print(mt.grid_tuning(grid=param_grid, score=scoring_dict, model=rf, refit_var="R2"))

import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LassoLarsCV, SGDRegressor,LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

battery_df = pd.read_csv('Battery_RUL.csv')
# battery_df.info()

target = battery_df['RUL']
feature = battery_df.drop(['RUL', 'Cycle_Index'],axis=1)

## training set splitting ##
x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size= 0.2, random_state= 0)
# X_train1, X_train2, y_train1, y_train2 = train_test_split(feature, target, test_size= 0.8, random_state= 0)

## LassoLarsCV cross validaiton model ##
def lassolarcv(train_x, train_y, test_x, test_y):
    """Fit usig LassoLarsCV model

    Args:
        train_x (_type_): _description_
        train_y (_type_): _description_
        test_x (_type_): _description_
        test_y (_type_): _description_
    """
    model = LassoLarsCV(cv=5)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    print(f"R^2 score from LassoLarsCV: {score}")
    _metrics(model,train_x,train_y,test_x,test_y)
    _plot_residual_analysis(model,test_y, test_x)
    feature_index = np.argsort(model.feature_names_in_)
    for number in range(train_x.shape[1]):
        print(f"{train_x.columns[feature_index[number]]}: {model.coef_[feature_index[number]]}")

## SGDRegressor
def sgdregressor(train_x,train_y,test_x,test_y):
    model = SGDRegressor()
    model.fit(train_x,train_y)
    _metrics(model,train_x,train_y,test_x,test_y)
    _plot_residual_analysis(model,test_y, test_x)
    feature_index = np.argsort(model.feature_names_in_)
    for number in range(train_x.shape[1]):
        print(f"{train_x.columns[feature_index[number]]}: {model.coef_[feature_index[number]]}")


def voting(feature_db, target_db):
    clf1 = LogisticRegression ()
    clf2 = RandomForestClassifier()
    clf3 = GaussianNB()
    #cross-validation on all model
    for clf,label in zip([clf1,clf2,clf3],['LogisticRegression ','RandomForestClassifier','GaussianNB',]):
        scores = cross_val_score(clf,feature_db,target_db,cv=5)
        print(f"{label} : Acurracy = {np.mean(scores):.2f}")
    voting_clf = VotingClassifier(estimators=[('lr',clf1),('rf',clf2),('nb',clf3),],voting='soft')
    voting_scores = cross_val_score(voting_clf,feature_db,target_db,cv=5)
    print(f"Voting Classifier (Soft): Accuracy = {np.mean(voting_scores):.2f}")


################# HELPER FUNCTION ##############
def _metrics(model,train_x, train_y , test_x, test_y):
    """Generate sklearn metric packages for mae, mse ,rmse

    Args:
        model (_type_): _description_
        train_x (_type_): _description_
        train_y (_type_): _description_
        test_x (_type_): _description_
        test_y (_type_): _description_
    """
    score_val = cross_val_score(model, train_x, train_y, cv =5,scoring='r2')
    print(f"Cross validation score using R^2: {score_val}")
    mae = mean_absolute_error(test_y,model.predict(test_x))
    mse = mean_squared_error(test_y,model.predict(test_x)) #actual vs predicted value
    rmse = sqrt(mse) #predicted vs. observed value
    print(f"Mean absolute error: {mae}")
    print(f"Mean square error: {mse}")
    print(f"Root mean square error: {rmse}")

def _plot_residual_analysis(model,test_y, test_x):
    residuals = test_y - model.predict(test_x)
    plt.scatter(model.predict(test_x), residuals)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residual plot')
    plt.show()


###### TEST AREA #########
# sgdregressor(X_train1,y_train1,X_test1, y_test1)
lassolarcv(x_train,y_train, x_test, y_test)




# #https://www.kaggle.com/code/yeonseokcho/battery-remaining-life-prediction/notebook

#DOCS: https://scikit-learn.org/stable/modules/model_evaluation.html, https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lars.html
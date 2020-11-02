import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import *
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV


def logreg_prediction(X, y, ts, rs):
    """
    Logistic Regression Model
    
    Parameters
    ----------
    X : the dataframe to use in the logistic regression

    y : the target to reach in the logistic regression

    ts = test_size : the size of the test dataframe

    rs = random_state : controls the shuffling applied to the data before applying the split.

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts,random_state=rs)

    Reg_Log = LogisticRegression()
    Reg_Log.fit(X_train, y_train)
    y_pred = Reg_Log.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("score : ", Reg_Log.score(X_test,y_test))
    
    
def randforest_prediction(X, y, ts, rs, estimators=300,depth=20):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts,random_state=rs)
    
    random_forest = RandomForestClassifier(n_estimators=estimators,criterion="gini",max_depth=depth)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("score : ", random_forest.score(X_test,y_test))
    return random_forest.score(X_test,y_test)


def gridsearchCV_prediction(X,y,ts,rs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts,random_state=rs)
    
    Estimator = RandomForestClassifier(random_state=42)
    parameters = {
        'n_estimators': [100,150,200,250,300],
        'max_depth': np.arange(6,16,2),
        'min_samples_split': np.arange(10,30,5),
        'min_samples_leaf': np.arange(5,20,5),
        'criterion': ["gini"]
    }

    model2 = GridSearchCV

    # cross_val_score(estimator=Estimator, X=X_train, y=y_train, cv=5)
    gd_sr = GridSearchCV(estimator=Estimator,
                         param_grid=parameters,
                         cv=5,verbose=1,
                         n_jobs=-1)
    gd_sr.fit(X_train, y_train)
    y_pred = gd_sr.predict(X_test)
    print("Parameters chosen: ", gd_sr.best_params_)
    print(classification_report(y_test, y_pred))
    print("score: ",gd_sr.score(X_test, y_test))

    
# imports
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import *
from TaxiFareModel.encoders import *

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_distance = make_pipeline(DistanceTransformer(),StandardScaler())
        dist_cols = ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column = 'pickup_datetime'),OneHotEncoder(sparse = False))
        time_cols = ['pickup_datetime']
        preprocessing = ColumnTransformer([('time',pipe_time,time_cols),('distance',pipe_distance,dist_cols)])
        self.pipeline = Pipeline([('preprocessing',preprocessing),('regressor',LinearRegression())])

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X,self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return np.sqrt(((y_pred - y_test)**2).mean())


if __name__ == "__main__":
    df = get_data()
    df_cleaned = clean_data(df)
    # set X and y
    X = df.drop(columns = 'fare_amount')
    y = df[['fare_amount']]
    # hold out
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

    # train
    trainer = Trainer(X_train,y_train)
    trainer.run()
    rmse = trainer.evaluate(X_test,y_test)
    #if rmse['fare_amount'] > 0 :
        #df.to_csv(index = False)
    print(rmse)
    # evaluate

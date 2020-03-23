# This script trains the model on the latest dataset and predicts the next value
# Author: Neilay Khasnabish
# Instructions: Get daily data from John Hopkin's Univ
# https://github.com/CSSEGISandData/COVID-19


#  Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV


# Plotting graph
def plotGraph3(ref, mdl, varName):
    plt.figure()
    plt.plot(np.ravel(ref), 'g')
    plt.plot(mdl, 'r')
    plt.title(varName)
    plt.ylabel('Number of infected people')
    plt.xlabel('Data points')
    plt.show()


# Finding RMSE
def ErrorCalc(mdl, ref, tag):
    relError = np.abs(mdl - ref)/ np.abs(ref+1)
    MeanErrorV = np.mean(relError)
    print(tag + ': Mean Rel Error in %: ', MeanErrorV * 100)
    return MeanErrorV


# Since cumulative prediction
def AdjustingErrorsOutliers(tempPred, df) :
    tempPred = np.round(tempPred)
    tempPrev = df['day5'].to_numpy() # Next cumulative prediction must be more than or equal to previous
    for i in range(len(tempPred)):
        if tempPred[i] < tempPrev[i] : # Since cumulative prediction
            tempPred[i] = tempPrev[i]
    return tempPred


# Train model
def TrainMdl (trainIpData, trainOpData, PredictionData) :
    testSize = 0.1 # 90:10 ratio >> for final testing
    randomState = 42 # For train test split

    # Final validation
    X_train, X_test, y_train, y_test = train_test_split(trainIpData, trainOpData, test_size=testSize, random_state=randomState)

    # Another set of input
    TrainIP = X_train[['day1', 'day2', 'day3', 'day4', 'day5', 'tempVal', 'ageVal']]
    TrainOP = X_train['gammaFun']
    TestIP = X_test[['day1', 'day2', 'day3', 'day4', 'day5', 'tempVal', 'ageVal']]
    TestOP = X_test['gammaFun']

    print('Training starts ...')
    # Adaboost Regressor >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    treeDepth = 10 # Fixed
    mdl = DecisionTreeRegressor(max_depth=treeDepth) # This is fixed
    param_grid = {
    'n_estimators': [100, 250, 500],
    'learning_rate': [0.1, 0.01, 0.001]
                }
    regrMdl = AdaBoostRegressor(base_estimator=mdl)
    clf = RandomizedSearchCV(estimator = regrMdl, param_distributions = param_grid,
                                     n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    clf.fit(TrainIP, TrainOP)
    print('Model ready ...')

    # Calculating Error
    y_predictedTrain = clf.predict(TrainIP) # Predicting the gamma function
    y_predictedTrain = AdjustingErrorsOutliers(y_predictedTrain * TrainIP['day5'].to_numpy(), TrainIP)
    ErrorCalc(y_predictedTrain, y_train.to_numpy(), 'Train Data-set') # y_predictedTrain converted to numbers

    y_predictedTest = clf.predict(TestIP) # Predicting the gamma function
    y_predictedTest = AdjustingErrorsOutliers(y_predictedTest * TestIP['day5'].to_numpy(), TestIP)
    ErrorCalc(y_predictedTest, y_test.to_numpy(), 'Validation Data-set ') # y_predictedTest converted to numbers

    # Prediction
    finalPrediction = clf.predict(PredictionData)  # Predicting the gamma function
    tempPred = finalPrediction * PredictionData['day5'].to_numpy()
    y_predictedFinal = AdjustingErrorsOutliers(tempPred, PredictionData)


    return y_predictedFinal


# Main code starts
df = pd.read_csv('G:/COVID19_Data/Processed_Data/TrainTest.csv')
dfP = pd.read_csv('G:/COVID19_Data/Processed_Data/Predict.csv')
trainIpData = df[['day1', 'day2', 'day3', 'day4', 'day5', 'tempVal', 'ageVal', 'gammaFun']]
trainOpData = df['dayPredict']
PredictionData = dfP[['day1', 'day2', 'day3', 'day4', 'day5', 'tempVal', 'ageVal']]
predictions = TrainMdl (trainIpData, trainOpData, PredictionData)
dfP['NextPredictions'] = predictions
dfP['LatestNumberCases'] = dfP['day5']
dfP[['Country', 'LatestNumberCases', 'NextPredictions']].to_csv('G:/COVID19_Data/Processed_Data/CountryWisePredictions.csv')
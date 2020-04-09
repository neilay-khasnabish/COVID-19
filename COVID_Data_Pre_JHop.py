# Create and Preprocess Data
# This script can extract data from John Hopkins Dataset and make it ready for our application
# Author: Neilay Khasnabish
# Instructions: Get daily data from John Hopkin's Univ
# https://github.com/CSSEGISandData/COVID-19


# Lib
import pandas as pd
import numpy as np


# Settings
displayAll = 1
if displayAll == 1 :
    pd.set_option('display.max_columns', None)

# Adjust John Hopkins Dataset
worldCorona = pd.read_csv('G:/COVID19_Data/time_series_covid19_confirmed_global.csv')
worldCorona = worldCorona.fillna(0)
worldCorona = worldCorona.drop(['Province/State'], axis=1)
worldCorona['Country'] = worldCorona['Country/Region']
worldCorona = worldCorona.drop(['Country/Region', 'Lat', 'Long'], axis=1)
worldCorona = worldCorona.groupby(['Country']).sum()
worldCorona.to_csv('Hellos.csv')
worldCorona = pd.read_csv('Hellos.csv')


# Total countries in the world
totalCountries = pd.read_csv('G:/COVID19_Data/WorldCountryNames.csv')
print('Countries infected: ', np.shape(worldCorona)[0], '/', np.shape(totalCountries)[0])

# Countrywise temperature
countryTemp = pd.read_csv('G:/COVID19_Data/Temp.csv')

# Countrywise age
countryAge = pd.read_csv('G:/COVID19_Data/MedAge.csv')

# Merging all three
result1 = pd.merge(worldCorona, countryTemp, on='Country').reset_index(drop=True)
result = pd.merge(result1, countryAge, on='Country').reset_index(drop=True)
print('Final size of merged data (rows equal to number of countries to be processed): ', np.shape(result))
result.to_csv('hellow.csv')

# Creating dataframe for training
[rf, cf] = np.shape(result)
print('Row', rf, '| Col: ', cf)
df=[]
for i in range(rf): # It scans through the entire row
    iCol = 6 # Start index
    while iCol <= cf-4 :
        dayPredict = result.iloc[i, iCol+1]
        day5 = result.iloc[i, iCol]
        day4 = result.iloc[i, iCol-1]
        day3 = result.iloc[i, iCol-2]
        day2 = result.iloc[i, iCol-3]
        day1 = result.iloc[i, iCol-4]
        diff1 = day5 - day4
        diff2 = day4 - day3
        diff3 = day3 - day2
        diff4  = day2 - day1
        iCol = iCol + 1
        ageVal = result.iloc[i, cf - 1]
        tempVal = result.iloc[i, cf - 2]
        dividen = day5 + 1
        gammaFun = dayPredict / dividen
        data = {'day1': [day1], 'day2': [day2], 'day3': [day3], 'day4': [day4], 'day5': [day5], 'tempVal': [tempVal], 'ageVal': [ageVal],
                'dayPredict': [dayPredict], 'gammaFun': [gammaFun], 'diff1': [diff1], 'diff2': [diff2], 'diff3': [diff3], 'diff4': [diff4]}
        df2 = pd.DataFrame(data)
        df.append(df2)

df = pd.concat(df).reset_index(drop=True)
df = df.fillna(0)
df.to_csv('G:/COVID19_Data/Processed_Data/TrainTest.csv')


# Preparing real-time prediction data
dfP=[]
for i in range(rf): # It scans through the entire row
    day5 = result.iloc[i, cf - 3]
    day4 = result.iloc[i, cf - 4]
    day3 = result.iloc[i, cf - 5]
    day2 = result.iloc[i, cf - 6]
    day1 = result.iloc[i, cf - 7]
    diff1 = day5 - day4
    diff2 = day4 - day3
    diff3 = day3 - day2
    diff4 = day2 - day1
    ageVal = result.iloc[i, cf - 1]
    tempVal = result.iloc[i, cf - 2]
    countryName = result.iloc[i, 0]
    data = {'day1': [day1], 'day2': [day2], 'day3': [day3], 'day4': [day4], 'day5': [day5], 'tempVal': [tempVal], 'ageVal': [ageVal], 'Country': [countryName],
            'diff1': [diff1], 'diff2': [diff2], 'diff3': [diff3], 'diff4': [diff4]}
    df2 = pd.DataFrame(data)
    dfP.append(df2)

dfP = pd.concat(dfP).reset_index(drop=True)
dfP.to_csv('G:/COVID19_Data/Processed_Data/Predict.csv')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plot
from matplotlib import style
import seaborn as sns
from sklearn import svm
from sklearn import preprocessing
import math
import sys
print(80*'=')
print('Loading the Data')
print(80*'-')
'''data=pd.read_csv('./us_data_combined.csv')
print(80*'=')
print('Filtering the Road Condition Data:')
road_data=data[['Description','Start_Time', 'End_Time','City','State', 'Zipcode','Distance(mi)',
        'Side','Weather_Condition','Sunrise_Sunset','Amenity', 'Bump', 'Crossing','Give_Way','Junction',
        'No_Exit','Railway', 'Roundabout', 'Station', 'Stop','Traffic_Calming',
        'Traffic_Signal', 'Turning_Loop','Severity']]
#save the road condtion data to the file
road_data.to_csv('./road_condition_2019_draft.csv')'''
road_data=pd.read_csv('./road_condition_2019_draft.csv')
# convert Severity column to numeric.
#road_data['Severity']=road_data['Severity'].apply(lambda x:0 if x=='Low' else 1)
print('Sample data:','\n',road_data.head(10))
print(80*'=')
print('Count of records is :',len(road_data.index))
print(80*'=')
print('Columns:',road_data.columns)
print(80*'=')
print('Analyzeing the Data')
print(80*'-')
print('1- Plot the Severity High/Low')
sns.countplot(x='Severity', data=road_data)
plot.show()
print(80*'-')
print('2- plot correlation matrix heatmap between the features ')
feature_cols = road_data[['Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop']]
corr = feature_cols.corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plot.show()
print(80*'-')
print('Lets do more analysis for the data using different features')
print(80*'-')
print('3- Plot the Severity High/Low per Day time ')
sns.countplot(x='Sunrise_Sunset',hue='Severity',data=road_data)
plot.show()
print(80*'=')
print('4- Plot the Severity if there is a Junction ')
sns.countplot(x='Junction', hue='Severity',data=road_data)
plot.show()
print(80*'=')
print('5- Plot the Severity if there is a Bump ')
sns.countplot(x='Bump', hue='Severity',data=road_data)
plot.show()
print(80*'=')
print('6- Plot the Severity if there is a No Exit ')
sns.countplot(x='No_Exit', hue='Severity',data=road_data)
plot.show()
print(80*'=')

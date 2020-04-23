#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
import missingno as msno

# 1- Severity

df = pd.read_csv('./us_data_combined.csv')

print(df.columns)
labels = ['High','Low']
print(labels)
severity_count = df.groupby("Severity")["Severity"].count()
print(severity_count)
explode = (0, 0)
plt.pie(severity_count, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
plt.show()

# 2- temperature
temp= df["Temperature(F)"].dropna()
print(temp)
plt.hist(temp, color='blue', edgecolor='black',bins=int(180 / 5))
plt.show()

# 3- Sunrise_Sunset

print(df['Sunrise_Sunset'].head(10))
sunrise_sunset_count = df.groupby("Sunrise_Sunset")["Sunrise_Sunset"].count()
severity_count = df.groupby("Severity")["Severity"].count()
plt.bar(severity_count, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)


'''us_data

# In[6]:


plt.figure(figsize=(9, 8))
sns.distplot(us_data["Temperature(F)"], color="r", bins=50, hist_kws={'alpha': 0.4})
plt.show()

# In[7]:


plt.figure(figsize=(9, 8))
sns.distplot(us_data["Wind_Chill(F)"], color="b", bins=50, hist_kws={'alpha': 0.4})
plt.show()
# In[9]:


plt.figure(figsize=(9, 8))
sns.distplot(us_data["Humidity(%)"], color="y", bins=100, hist_kws={'alpha': 0.4})
plt.show()

# In[10]:


plt.figure(figsize=(9, 8))
sns.distplot(us_data["Wind_Speed(mph)"], color="b", bins=100, hist_kws={'alpha': 0.4})
plt.show()

# In[13]:


plt.figure(figsize=(9, 8))
sns.distplot(us_data["Pressure(in)"], color="b", bins=50, hist_kws={'alpha': 0.4})
plt.show()
# In[ ]:


# In[ ]:'''






# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import xport
import time


# In[3]:


df = pd.read_sas('LLCP2017.XPT',format='xport')
# with open('/root/usr/share/Project for ML/v1.0/LLCP2017.XPT', 'rb') as f:
#     for row in xport.Reader(f):
#         print (row)


# In[13]:


df.head()


# ## Filter the dataset by keeping useful features
# 
# Each of the features is a particular question in the survey, by checking with the [codebook](https://www.cdc.gov/brfss/annual_data/2017/pdf/codebook17_llcp-v2-508.pdf) we will know the corresponding meaning of any feature. In this step, we filter out irrelevant features.

# In[5]:


race = df['_IMPRACE']
age = df['_AGEG5YR']
sex = df['SEX']
height = df['HTM4'] # in cm 91-244
weight = df['WEIGHT2'] # in Pounds 
bmi = df['_BMI5CAT'] # 4 bmi levels: 1 underweight, 2 normal, 3 overweight, 4 obese
smoke_status = df['_SMOKER3'] # 1 Everyday smoker, 2 Someday smoker, 3 Former smoker, 4 Non-smoker, 9 nil
general_health = df['GENHLTH'] # 1 excelent, 2 very good, 3 good, 4 fair, 5 poor, 7 don't know, 9 refused, blank nil
sleep_hour = df['SLEPTIM1'] # 1-24 # hours, 77 don't know, 99 refused, blank nil
has_heart_disease = df['CVDINFR4'] # cardiovascular problems. 1 yes, 2 no, 7 don't know, 9 refused, blank nil
has_cancer = df['CNCRDIFF'] # number of types of cancer 1 had. 1 2 3. 7 don't know, 9 refuesd, blank no cancer
mental_health = df['_MENT14D'] # number of days mental no good. 1 no, 2 1-13 days, 3 14+ days, 9 nil
binge_drinker = df['_RFBING5'] # 1 No, 2 yes, 9 nil
has_diabetes = df['DIABETE3'] # 1 Yes, 2 yes female during pregnancy, 3 no, 4 no, 7 9 blank


# In[10]:


brfss = pd.concat([race, age, sex, height, weight, smoke_status, general_health, sleep_hour, has_heart_disease, has_cancer, 
                  mental_health, binge_drinker, has_diabetes, bmi], axis=1)
brfss.columns = ['race', 'age', 'sex', 'height', 'weight', 'smoke_status', 'general_health', 'sleep_hour', 'has_heart_disease', 
                 'has_cancer', 'mental_health', 'binge_drinker', 'has_diabetes', 'bmi_label']


# In[39]:


brfss.to_csv('brfss_short.csv', index=False)


# In[40]:


df = pd.read_csv('brfss_short.csv')


# ## Clean the data

# In[70]:


race_replace = {1:'White', 2:'Black', 3:'Asian', 4:'American Indian/Alaskan Native', 5:'Hispanic', 6:'Ohter'}
age_replace = {1:'18-24', 2:'25-29', 3:'30-34', 4:'35-39', 
               5:'40-44', 6:'45-49', 7:'50-54', 8:'55-59', 9:'60-64', 10:'65-69', 11:'70-74', 12:'75-79', 13:'80+', 14:np.nan}
sex_replace = {1:'male', 2:'female'}
smoke_replace = {1:'Everyday smoker', 2:'Someday smoker', 3:'Former smoker', 4:'Non-smoker', 9:np.nan}
health_replace = {1:'excelent', 2:'very good', 3:'good', 4:'fair', 5:'poor', 9:np.nan}
heart_replace = {1:'yes', 2:'no', 9:np.nan}


# In[71]:


race = race.replace(race_replace)
age = age.replace(age_replace)
sex = sex.replace(sex_replace)
smoke_status = smoke_status.replace(smoke_replace)
general_health = general_health.replace(health_replace)
has_heart_disease = has_heart_disease.replace(heart_replace)


# In[72]:


brfss_clean = pd.concat([race, age, sex, height, weight, bmi, smoke_status, general_health, sleep_hour, has_heart_disease, has_cancer, 
                  mental_health, binge_drinker, has_diabetes], axis=1)
brfss_clean.columns = ['race', 'age', 'sex', 'height', 'weight', 'bmi', 'smoke_status', 'general_health', 'sleep_hour', 'has_heart_disease', 
                 'has_cancer', 'mental_health', 'binge_drinker', 'has_diabetes']
brfss_clean.to_csv('brfss_clean.csv', index=False)


# In[73]:


brfss_clean


# In[11]:


brfss.head()


# In[19]:


brfss = brfss.fillna(value = {'has_cancer': 0})


# In[23]:


brfss = brfss.dropna()


# In[24]:


brfss.info()


# In[25]:


brfss.to_csv('brfss_13cols.csv', index=False)


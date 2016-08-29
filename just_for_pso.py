
# coding: utf-8

# In[1]:

#!pip install pyswarm


# In[2]:

import numpy as np
#from sklearn.metrics import jaccard_similarity_score


# In[3]:

from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")


# In[4]:

import pandas as pd


# In[5]:

df = pd.read_csv("https://raw.githubusercontent.com/iamnewneo/Context-Aware-Recommender-System/master/data/comoda_data.csv")


# In[6]:

df = df.replace(-1.0, np.nan)
df = df.dropna(how="any")


# In[7]:

## CONSTANTS ASSUMPTION
sigma = 1
threshold = 0.1
columns = df.columns.tolist()
columns.remove("rating")
columns.append("rating")
df = df[columns]
X = df[columns[2:-1]]
y = df[columns[-1:]]
df = df.reset_index(drop=True)


# In[8]:

columns = df.columns


# In[9]:

columns = columns[2:-1]


# In[10]:

df_new = df.drop(['city','physical','director','movieCountry','movieLanguage','movieYear','genre1', 'genre2',
       'genre3', 'actor1', 'actor2', 'actor3','budget'], axis=1,)


# In[11]:

from collections import defaultdict
new_size_of_feature = defaultdict(dict)
for column in df_new.columns:
    new_size_of_feature[column]['length'] = len(df_new[column].unique())
    new_size_of_feature[column]['names'] = df_new[column].unique()


# In[12]:

sum_of_features = 0
for k,v in new_size_of_feature.items():
    sum_of_features += v['length']


# In[13]:

## Get Dummy Variables
for column in df_new.columns[2:-1]:
    df_col = pd.get_dummies(df_new[column], prefix=column)
    df_new = pd.concat([df_new, df_col], axis=1)


# In[14]:

df_new = df_new.drop(['age', 'sex', 'country', 'time', 'daytype',
       'season', 'location', 'weather', 'social', 'endEmo', 'dominantEmo',
       'mood', 'decision', 'interaction'], axis=1)


# In[15]:

columns = df_new.columns.tolist()
columns.remove("rating")
columns.append("rating")
df_new = df_new[columns]


# In[16]:

entriesToRemove = ('userID','itemID','city','physical','director','movieCountry','movieLanguage','movieYear','genre1', 'genre2',
       'genre3', 'actor1', 'actor2', 'actor3','budget','rating')


# In[17]:

for k in entriesToRemove:
    new_size_of_feature.pop(k, None)


# In[18]:

index_limit=[]
for x in df.columns:
    index_limit.append(new_size_of_feature[x])


# In[19]:

index_limit = index_limit[2:-12]


# In[20]:

final_index_limit= []
for i in range(len(index_limit)):
    x = index_limit[i]
    try:
        final_index_limit.append(x['length'])
    except:
        pass


# In[21]:

final_index_limit, sum(final_index_limit)


# In[22]:

from pyswarm import pso
import time
from random import *


# In[23]:

start_time = time.time()
#user_context = df_new.iloc[0][2:-1].tolist()
user_rating = df_new.rating.tolist()[:30]
swarmsize=200
maxiter=100
def banana(x):
    lhs = 0.0
    square_error = 0.0
    for i in range(30):
        user_context = df_new.iloc[i][2:-1].tolist()
        lhs = np.dot(x,user_context)
#         for weight, context in zip(x, user_context):
#             lhs += weight * context
        square_error += (user_rating[i] - lhs)**2
    #print(square_error)
    return square_error/2

def con(x):
    con_index = []
    final_index_limit = [21, 2, 4, 4, 3, 4, 3, 5, 7, 7, 7, 3, 2, 2]
    start_index=0
    con_index = []
    for index in final_index_limit:
        temp = []
        temp.append(sum(x[start_index:start_index+index]) - 1)
        temp = temp*index
        con_index=con_index + temp
        start_index = index
    return con_index

lb = [0]*74
ub = [1]*74

#xopt, fopt = pso(banana,lb, ub,f_ieqcons=con,swarmsize=swarmsize,maxiter=maxiter)
print("--- %s seconds ---" % (time.time() - start_time))


# In[24]:

# print("----------------------Difference Between Ratings---------------------")
# for i in range(10):
#     rating = 0
#     count = 0
#     original_rating = df_new.iloc[i][-1]
#     predicted_rating = np.dot(xopt, df_new.iloc[i][2:-1].tolist())
#     print(predicted_rating, original_rating)
#     #print(str(float(original_rating - original_rating)))


# In[25]:

# from sklearn.metrics import mean_squared_error
# original_rating = [df_new.iloc[i][-1] for i in range(1000)]
# predicted_rating = [np.dot(xopt, df_new.iloc[i][2:-1].tolist()) for i in range(1000)]
# print("----------------------  RMSE VALUE IS  ---------------------")
# mean_squared_error(original_rating,predicted_rating)


# In[26]:

from sklearn.metrics import mean_squared_error
def calculate_mse(weights):
    original_rating = [df_new.iloc[i][-1] for i in range(1000)]
    predicted_rating = [np.dot(weights, df_new.iloc[i][2:-1].tolist()) for i in range(1000)]
    #print("----------------------  RMSE VALUE IS  ---------------------")
    return mean_squared_error(original_rating,predicted_rating)


# In[ ]:

def run_pso_multiple_times():
    swarm_size = 100
    indi_results = []
    for max_iter in [100,150,200,300,400,500]:
        print("Current number of Iteration", str(max_iter))
        start_time = time.time()
        xopt, fopt = pso(banana,lb, ub,f_ieqcons=con,swarmsize=swarm_size,maxiter=max_iter)
        rmse = calculate_mse(xopt)
        finish_time = time.time() - start_time
        pso_param = [swarm_size, max_iter, finish_time, rmse]
        indi_results.append(pso_param + xopt.tolist())
    indi_results = np.array(indi_results)
    np.savetxt("pso.csv", indi_results, delimiter=",")


# In[ ]:

run_pso_multiple_times()


# In[ ]:




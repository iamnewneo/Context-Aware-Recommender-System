
# coding: utf-8

# In[5]:

import numpy as np
from sklearn.metrics import jaccard_similarity_score


# In[6]:

from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")


# In[7]:

import pandas as pd


# In[8]:

df = pd.read_csv("data/comoda_data.csv")


# In[9]:

df = df.replace(-1.0, np.nan)
df = df.dropna(how="any")


# ## Clean Data Here

# In[10]:

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
#df


# In[11]:

class neighbors:
    def __init__(self, X, y):
        self.model = KNeighborsClassifier(n_neighbors=10)
        self.model.fit(X,y)
    def get_neighbors_data(self, user, itemID):
        #all_neighbors_indices = self.model.kneighbors(user[2:-1])[1][0]
        #print(all_neighbors_indices)
        #neighbors_data = df.iloc[all_neighbors_indices]
        neighbors_data = df[(df.itemID == itemID)]
        #print(neighbors_data)
        self.good_neighbors = pd.DataFrame()
        #print(neighbors_data.shape)
        threshold = 0.3
        for index, row in neighbors_data.iterrows():
            similarity = jaccard_similarity_score(row[2:-1], user[2:-1])
            #print(similarity)
            if similarity > threshold:
                self.good_neighbors=self.good_neighbors.append(row, ignore_index=True)
                #print(row)
                #print(good_neighbors)
                #print("Inside Threshold")
            #print(self.good_neighbors)
        #print(self.good_neighbors)
        return self.good_neighbors


# In[12]:

class similarity_class:
    def __init__(self):
        pass
    def jaccard_similarity(self, x, y):
        return jaccard_similarity_score(x,y)


# In[13]:

class neighborhood_contribution:
    def __init__(self):
        pass
    def neighbor_rating(self, neighbor, itemID, sigma, threshold):
        self.ratings_sum=0
        self.similarity_sum=0
        #print(type(user))
        #print(user)
        #print(neighbor[2:10])
        #print(df.itemID)
        ratings = df[(df.userID == neighbor.userID) & (df.itemID == itemID)]
        #print(ratings.shape)
        for index, user_rating in ratings.iterrows():
            similarity = jaccard_similarity_score(neighbor[2:-1],user_rating[2:-1])
            #print(similarity)
            if similarity > threshold:
                self.ratings_sum += user_rating.rating * similarity
                self.similarity_sum += similarity
                #print(self.similarity_sum)
            #print(self.ratings_sum)
            #print(neighbor[-10:],user_rating[-10:])
        #print("Rating sum")
        #print(self.ratings_sum)
        #print("Similarity Sum")
        #print(self.similarity_sum)
        try:
            rating = self.ratings_sum / self.similarity_sum
            #rating = ratings.rating.mean()
        except:
            rating = 0
        #print(rating)
        return rating
    def neighbor_average(self,neighbor,sigma,threshold):
        ratings_of_neighbor = df[df.userID == neighbor.userID]#.rating.mean()
        rating_sum=0;
        count=0;
        for index, row in ratings_of_neighbor.iterrows():
            similarity = jaccard_similarity_score(row[2:-1], neighbor[2:-1])
            if similarity > threshold:
                rating_sum+=row.rating
                count+=1
        average_rating_in_given_context = rating_sum/count
        #print(average_rating_in_given_context)
        return average_rating_in_given_context
    def baseline_rating(self, itemID):
        average_item_rating = df[df.itemID == itemID].rating.mean()
        return average_item_rating


# In[14]:

def numerator(neighborhood_contribution_rating, similarity_of_neighbors):
    numerator_sum=0
    #print(neighborhood_contribution_rating)
    #print(similarity_of_neighbors)
    for neighbor_rating, similarity in zip(neighborhood_contribution_rating,similarity_of_neighbors):
        numerator_sum+= neighbor_rating*similarity
    return numerator_sum


# In[15]:

def predict_rating(user,itemID):
    similarity_function = similarity_class()
    user_neighbors = neighbors(X,y)
    user_neighborhood_contribution = neighborhood_contribution()
    neighbors_data = user_neighbors.get_neighbors_data(user,itemID)
    #print(neighbors_data.shape)
    #print(neighbors_data)
    #similarity_of_neighbors = [similarity_function.jaccard_similarity(neighbor[2:-1],user[2:-1]) for neighbor in neighbors_data]
    similarity_of_neighbors = []
    neighborhood_contribution_rating = []
    #print(neighbors_data.shape)
    for index, neighbor in neighbors_data.iterrows():
        similarity_of_neighbors.append(similarity_function.jaccard_similarity(neighbor[2:-1],user[2:-1]))
        neighborhood_contribution_rating.append(user_neighborhood_contribution.neighbor_rating(neighbor,itemID,1,0.1) - user_neighborhood_contribution.neighbor_average(neighbor,1,0.1))
#     neighborhood_contribution_rating = [user_neighborhood_contribution.neighbor_rating(neighbor,item) - user_neighborhood_contribution.neighbor_average(neighbor)
#                                         for neighbor in neighbors_data]
    numerator_value = numerator(neighborhood_contribution_rating, similarity_of_neighbors)
    baseline_rating = user_neighborhood_contribution.baseline_rating(itemID)
    #print(numerator_value)
    #print(similarity_of_neighbors)
    #print(baseline_rating)
    final_rating = baseline_rating + numerator_value/ sum(similarity_of_neighbors)
    return final_rating


# In[16]:

# if __name__ == "main":
#     userId = 20
#     iteId = 30
#     rating = predict_rating(20,30)
#     print(rating)


# In[17]:

predict_rating(df.iloc[37], 47)
#df.iloc[5]"


# In[18]:

#df[df.itemID == 47]


# In[19]:

#df.iloc[37]


# In[20]:

from sklearn.metrics import mean_squared_error


# In[21]:

#y_true = [df.iloc[i].rating for i in range(700,800)]


# In[22]:

#y_pred = [predict_rating(df.iloc[i], df.iloc[i].itemID) for i in range(700,800)]


# In[23]:

#y_pred


# In[24]:

#mean_squared_error(y_true, y_pred)


# ## Pyswarm Implementation

# In[ ]:




# In[25]:

len(df.rating.unique())


# In[26]:

columns = df.columns


# In[27]:

columns = columns[2:-1]


# In[28]:

columns


# In[29]:

#from collections import defaultdict
# size_of_feature = defaultdict(dict)
# for column in columns:
#     size_of_feature[column]['length'] = len(df[column].unique())
#     size_of_feature[column]['names'] = df[column].unique()
    


# In[30]:

#size_of_feature


# In[31]:

#size_of_feature['actor1']


# In[32]:

df.actor1.max()


# In[33]:

df.actor1.describe()


# In[34]:

# #from collections import defaultdict
# feature_dict = defaultdict(dict)
# for k,v in size_of_feature.items():
#     for i,j in v.items():
#         for name in v['names']:
#             feature_dict[k][name] = sum(df[k] == name)
#             #print(k,name)


# In[35]:

#feature_dict


# In[36]:

#df[df.columns[2:]].corr(method='kendall', min_periods=1)


# In[37]:

df.columns


# In[38]:

df_new = df.drop(['city','physical','director','movieCountry','movieLanguage','movieYear','genre1', 'genre2',
       'genre3', 'actor1', 'actor2', 'actor3','budget'], axis=1,)


# In[39]:

#df_new.describe()


# In[40]:

from collections import defaultdict
new_size_of_feature = defaultdict(dict)
for column in df_new.columns:
    new_size_of_feature[column]['length'] = len(df_new[column].unique())
    new_size_of_feature[column]['names'] = df_new[column].unique()


# In[41]:

sum_of_features = 0
for k,v in new_size_of_feature.items():
    sum_of_features += v['length']


# In[42]:

sum_of_features


# In[43]:

df_new.columns


# In[44]:

df_age = pd.get_dummies(df.age, prefix='age')


# In[45]:

#df_age.head()
df_new.head()


# In[46]:

for column in df_new.columns[2:-1]:
    df_col = pd.get_dummies(df_new[column], prefix=column)
    df_new = pd.concat([df_new, df_col], axis=1)


# In[47]:

df_new = df_new.drop(['age', 'sex', 'country', 'time', 'daytype',
       'season', 'location', 'weather', 'social', 'endEmo', 'dominantEmo',
       'mood', 'decision', 'interaction'], axis=1)


# In[48]:

df_new.shape


# In[49]:

df_new.head()


# In[50]:

columns = df_new.columns.tolist()
columns.remove("rating")
columns.append("rating")
df_new = df_new[columns]


# In[51]:

df_new.head()


# In[52]:

#df_new.iloc[0][2:-1].tolist()


# In[ ]:




# In[ ]:




# In[53]:


#user_rating = df_new.rating.tolist()[:100]


# In[ ]:




# In[ ]:




# In[54]:

entriesToRemove = ('userID','itemID','city','physical','director','movieCountry','movieLanguage','movieYear','genre1', 'genre2',
       'genre3', 'actor1', 'actor2', 'actor3','budget','rating')


# In[55]:

for k in entriesToRemove:
    new_size_of_feature.pop(k, None)


# In[56]:

for k,v in new_size_of_feature.items():
    print(v['length'])


# In[57]:

new_size_of_feature


# In[58]:

df.columns


# In[59]:

index_limit=[]
for x in df.columns:
    index_limit.append(new_size_of_feature[x])


# In[60]:

index_limit = index_limit[2:-12]


# In[61]:

index_limit


# In[62]:

final_index_limit= []
for i in range(len(index_limit)):
    x = index_limit[i]
    try:
        final_index_limit.append(x['length'])
    except:
        pass


# In[63]:

final_index_limit, sum(final_index_limit)


# ## Make the contstraint list

# In[64]:

from random import *
randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]
con_index = []
x = randBinList(181)
for index in final_index_limit:
    temp = []
    temp.append(sum(x[:index]) - 1)
    temp = temp*index
    con_index=con_index + temp


# In[65]:

#con_index
df_new.columns


# # Pyswarm Implementation

# In[66]:

# from pyswarm import pso
# import time
# from random import *


# In[67]:

# start_time = time.time()
# #user_context = df_new.iloc[0][2:-1].tolist()
# user_rating = df_new.rating.tolist()[:20]
# def banana(x):
#     lhs = 0.0
#     square_error = 0.0
#     for i in range(20):
#         user_context = df_new.iloc[i][2:-1].tolist()
#         lhs = np.dot(x,user_context)
# #         for weight, context in zip(x, user_context):
# #             lhs += weight * context
#         square_error += (user_rating[i] - lhs)**2
#     print(square_error)
#     return square_error/2

# def con(x):
#     con_index = []
#     final_index_limit = [21, 2, 4, 4, 3, 4, 3, 5, 7, 7, 7, 3, 2, 2]
#     start_index=0
#     con_index = []
#     for index in final_index_limit:
#         temp = []
#         temp.append(sum(x[start_index:start_index+index]) - 1)
#         temp = temp*index
#         con_index=con_index + temp
#         start_index = index
#     return con_index

# lb = [0]*74
# ub = [1]*74

# xopt, fopt = pso(banana,lb, ub,f_ieqcons=con,swarmsize=740,maxiter=1000)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[68]:

# Rating
# fopt


# In[69]:

# print("----------------------Difference Between Ratings---------------------")
# for i in range(10):
#     rating = 0
#     count = 0
#     original_rating = df_new.iloc[i][-1]
#     predicted_rating = np.dot(xopt, df_new.iloc[i][2:-1].tolist())
#     print(predicted_rating, original_rating)
#     print(str(float(original_rating - original_rating)))
        


# In[70]:

# original_rating = [df_new.iloc[i][-1] for i in range(1000)]
# predicted_rating = [np.dot(xopt, df_new.iloc[i][2:-1].tolist()) for i in range(1000)]
# print("----------------------  RMSE VALUE IS  ---------------------")
# mean_squared_error(original_rating,predicted_rating)


# In[71]:

#np.dot(x,y)


# In[72]:

# x = [1]*78
# final_index_limit = [21, 2, 4, 4, 3, 4, 3, 5, 7, 7, 7, 3, 2, 2]
# start_index=0
# con_index = []
# for index in final_index_limit:
#     temp = []
#     temp.append(sum(x[start_index:start_index+index]) - 1)
#     temp = temp*index
#     con_index=con_index + temp
#     start_index = index
# return con_index


# ### Firefly Optimization Implementation

# In[73]:

#user_rating = df_new.rating.tolist()[:20]


# In[77]:

from metaheuristic_algorithms.function_wrappers.abstract_wrapper import AbstractWrapper
class MinimizeWeightsSumSquareError(AbstractWrapper):

    def maximum_decision_variable_values(self):
        return [1]*74

    def minimum_decision_variable_values(self):
        return [0]*74

    def objective_function_value(self, decision_variable_values):
        user_rating = df_new.rating.tolist()[:20]
        lhs = 0.0
        square_error = 0.0
        for i in range(20):
            user_context = df_new.iloc[i][2:-1].tolist()
            lhs = np.dot(decision_variable_values,user_context)
            square_error += (user_rating[i] - lhs)**2
        return square_error/2

    def initial_decision_variable_value_estimates(self):
        return [0.1]*74


# In[79]:

import time
from metaheuristic_algorithms.firefly_algorithm import FireflyAlgorithm
start_time = time.time()
#from metaheuristic_algorithms.function_wrappers.rosenbrook_function_wrapper import RosenbrookFunctionWrapper
#from metaheuristic_algorithms.function_wrappers.nonsmooth_multipeak_function_wrapper import NonsmoothMultipeakFunctionWrapper
minimize_weights_function_wrapper = MinimizeWeightsSumSquareError()

number_of_variables = 74
objective = "minimization"

firefly_algo = FireflyAlgorithm(minimize_weights_function_wrapper, number_of_variables, objective)


number_of_fireflies = 740
maximun_generation = 1000
randomization_parameter_alpha = 0.2
absorption_coefficient_gamma = 0.4

result = firefly_algo.search(number_of_fireflies = number_of_fireflies, 
                               maximun_generation = maximun_generation, 
                               randomization_parameter_alpha = randomization_parameter_alpha, 
                               absorption_coefficient_gamma = absorption_coefficient_gamma)
print(result["best_decision_variable_values"])
print("--- %s seconds ---" % (time.time() - start_time))
print(result["best_objective_function_value"]) 


# In[ ]:

with open("result.txt", "w") as myfile:
    for key in sorted(result):
        myfile.write(key + "," + ",".join(d[key]) + "\n")


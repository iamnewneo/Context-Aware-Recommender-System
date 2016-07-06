
# coding: utf-8

# In[80]:

import numpy as np
from sklearn.metrics import jaccard_similarity_score


# In[81]:

from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")


# In[82]:

import pandas as pd


# In[83]:

df = pd.read_csv("data/comoda_data.csv")


# In[84]:

df = df.replace(-1.0, np.nan)
df = df.dropna(how="any")


# ## Clean Data Here

# In[85]:

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


# In[86]:

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


# In[87]:

class similarity_class:
    def __init__(self):
        pass
    def jaccard_similarity(self, x, y):
        return jaccard_similarity_score(x,y)


# In[88]:

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


# In[89]:

def numerator(neighborhood_contribution_rating, similarity_of_neighbors):
    numerator_sum=0
    #print(neighborhood_contribution_rating)
    #print(similarity_of_neighbors)
    for neighbor_rating, similarity in zip(neighborhood_contribution_rating,similarity_of_neighbors):
        numerator_sum+= neighbor_rating*similarity
    return numerator_sum


# In[90]:

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


# In[91]:

# if __name__ == "main":
#     userId = 20
#     iteId = 30
#     rating = predict_rating(20,30)
#     print(rating)


# In[92]:

predict_rating(df.iloc[37], 47)
#df.iloc[5]"


# In[93]:

#df[df.itemID == 47]


# In[94]:

#df.iloc[37]


# In[95]:

from sklearn.metrics import mean_squared_error


# In[96]:

#y_true = [df.iloc[i].rating for i in range(700,800)]


# In[97]:

#y_pred = [predict_rating(df.iloc[i], df.iloc[i].itemID) for i in range(700,800)]


# In[98]:

#y_pred


# In[99]:

#mean_squared_error(y_true, y_pred)


# ## Pyswarm Implementation

# In[ ]:




# In[100]:

len(df.rating.unique())


# In[101]:

columns = df.columns


# In[102]:

columns = columns[2:-1]


# In[103]:

columns


# In[104]:

#from collections import defaultdict
# size_of_feature = defaultdict(dict)
# for column in columns:
#     size_of_feature[column]['length'] = len(df[column].unique())
#     size_of_feature[column]['names'] = df[column].unique()
    


# In[105]:

#size_of_feature


# In[106]:

#size_of_feature['actor1']


# In[107]:

df.actor1.max()


# In[108]:

df.actor1.describe()


# In[109]:

# #from collections import defaultdict
# feature_dict = defaultdict(dict)
# for k,v in size_of_feature.items():
#     for i,j in v.items():
#         for name in v['names']:
#             feature_dict[k][name] = sum(df[k] == name)
#             #print(k,name)


# In[110]:

#feature_dict


# In[111]:

#df[df.columns[2:]].corr(method='kendall', min_periods=1)


# In[112]:

df.columns


# In[113]:

df_new = df.drop(['director','genre1', 'genre2',
       'genre3', 'actor1', 'actor2', 'actor3','budget'], axis=1,)


# In[114]:

#df_new.describe()


# In[115]:

from collections import defaultdict
new_size_of_feature = defaultdict(dict)
for column in df_new.columns:
    new_size_of_feature[column]['length'] = len(df_new[column].unique())
    new_size_of_feature[column]['names'] = df_new[column].unique()


# In[116]:

sum_of_features = 0
for k,v in new_size_of_feature.items():
    sum_of_features += v['length']


# In[117]:

sum_of_features


# In[118]:

df_new.columns


# In[119]:

df_age = pd.get_dummies(df.age, prefix='age')


# In[120]:

#df_age.head()
df_new.head()


# In[121]:

for column in df_new.columns[2:-1]:
    df_col = pd.get_dummies(df_new[column], prefix=column)
    df_new = pd.concat([df_new, df_col], axis=1)


# In[122]:

df_new = df_new.drop(['age', 'sex', 'city', 'country', 'time', 'daytype',
       'season', 'location', 'weather', 'social', 'endEmo', 'dominantEmo',
       'mood', 'physical', 'decision', 'interaction', 'movieCountry',
       'movieLanguage', 'movieYear'], axis=1)


# In[123]:

df_new.shape


# In[124]:

df_new.head()


# In[125]:

columns = df_new.columns.tolist()
columns.remove("rating")
columns.append("rating")
df_new = df_new[columns]


# In[126]:

df_new.head()


# In[127]:

#df_new.iloc[0][2:-1].tolist()


# In[ ]:




# In[ ]:




# In[128]:


#user_rating = df_new.rating.tolist()[:100]


# In[ ]:




# In[ ]:




# In[129]:

entriesToRemove = ('userID','itemID','city','physical','director','movieCountry','movieLanguage','movieYear','genre1', 'genre2',
       'genre3', 'actor1', 'actor2', 'actor3','budget','rating')


# In[130]:

for k in entriesToRemove:
    new_size_of_feature.pop(k, None)


# In[131]:

for k,v in new_size_of_feature.items():
    print(v['length'])


# In[132]:

new_size_of_feature


# In[133]:

df.columns


# In[134]:

index_limit=[]
for x in df.columns:
    index_limit.append(new_size_of_feature[x])


# In[135]:

index_limit = index_limit[2:-12]


# In[136]:

index_limit


# In[137]:

final_index_limit= []
for i in range(len(index_limit)):
    x = index_limit[i]
    try:
        final_index_limit.append(x['length'])
    except:
        pass


# In[144]:

final_index_limit, sum(final_index_limit)


# ## Make the contstraint list

# In[139]:

from random import *
randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]
con_index = []
x = randBinList(181)
for index in final_index_limit:
    temp = []
    temp.append(sum(x[:index]) - 1)
    temp = temp*index
    con_index=con_index + temp


# In[140]:

#con_index


# # Pyswarm Implementation

# In[141]:

from pyswarm import pso
import time
from random import *


# In[142]:

start_time = time.time()
#user_context = df_new.iloc[0][2:-1].tolist()
user_rating = df_new.rating.tolist()[:50]
def banana(x):
    lhs = 0.0
    square_error = 0.0
    for i in range(50):
        user_context = df_new.iloc[i][2:-1].tolist()
        for weight, context in zip(x, user_context):
            lhs += weight * context
        square_error += (user_rating[i] - lhs)**2
    #equation = (user_rating**2 - equation**2)/user_rating**2
    return square_error/2

def con(x):
    #randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]
    con_index = []
    #x = randBinList(178)
    final_index_limit = [21, 2, 4, 4, 3, 4, 3, 5, 7, 7, 7, 3, 2, 2]
    for index in final_index_limit:
        temp = []
        temp.append(1 - sum(x[:index]))
        temp = temp*index
        con_index=con_index + temp
    return con_index

lb = [0]*74
ub = [1]*74

xopt, fopt = pso(banana,lb, ub,f_ieqcons=con,swarmsize=740,maxiter=1000)
print("--- %s seconds ---" % (time.time() - start_time)


# In[145]:

##Rating


# In[ ]:

print("----------------------Difference Between Ratings---------------------")
for i in range(100):
    rating = 0
    count = 0
    original_rating = df_new.iloc[i][1]
    for weight, context in zip(xopt,df_new.iloc[i][2:-1].tolist()):
        rating += context*weight
    print(str(original_rating - rating))
        


# In[146]:




# In[ ]:




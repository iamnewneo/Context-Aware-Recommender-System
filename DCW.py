
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.metrics import jaccard_similarity_score


# In[2]:

from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")


# In[3]:

import pandas as pd


# In[4]:

df = pd.read_csv("data/comoda_data.csv")


# In[5]:

df = df.replace(-1.0, np.nan)
df = df.dropna(how="any")


# ## Clean Data Here

# In[6]:

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


# In[7]:

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


# In[8]:

class similarity_class:
    def __init__(self):
        pass
    def jaccard_similarity(self, x, y):
        return jaccard_similarity_score(x,y)


# In[9]:

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


# In[10]:

def numerator(neighborhood_contribution_rating, similarity_of_neighbors):
    numerator_sum=0
    #print(neighborhood_contribution_rating)
    #print(similarity_of_neighbors)
    for neighbor_rating, similarity in zip(neighborhood_contribution_rating,similarity_of_neighbors):
        numerator_sum+= neighbor_rating*similarity
    return numerator_sum


# In[11]:

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


# In[12]:

# if __name__ == "main":
#     userId = 20
#     iteId = 30
#     rating = predict_rating(20,30)
#     print(rating)


# In[13]:

predict_rating(df.iloc[37], 47)
#df.iloc[5]"


# In[14]:

#df[df.itemID == 47]


# In[15]:

#df.iloc[37]


# In[16]:

from sklearn.metrics import mean_squared_error


# In[17]:

#y_true = [df.iloc[i].rating for i in range(700,800)]


# In[18]:

#y_pred = [predict_rating(df.iloc[i], df.iloc[i].itemID) for i in range(700,800)]


# In[19]:

#y_pred


# In[20]:

#mean_squared_error(y_true, y_pred)


# ## Pyswarm Implementation

# In[21]:

# from pyswarm import pso

# def banana(x):
#     x1 = x[0]
#     x2 = x[1]
#     return x1**4 - 2*x2*x1**2 + x2**2 + x1**2 - 2*x1 + 5

# def con(x):
#     x1 = x[0]
#     x2 = x[1]
#     return [-(x1 + 0.25)**2 + 0.75*x2]

# lb = [-3, -1]
# ub = [2, 6]

# xopt, fopt = pso(banana, lb, ub, f_ieqcons=con)


# In[22]:

from pyswarm import pso


# In[23]:

len(df.rating.unique())


# In[24]:

columns = df.columns


# In[25]:

columns = columns[2:-1]


# In[26]:

columns


# In[27]:

#from collections import defaultdict
# size_of_feature = defaultdict(dict)
# for column in columns:
#     size_of_feature[column]['length'] = len(df[column].unique())
#     size_of_feature[column]['names'] = df[column].unique()
    


# In[28]:

#size_of_feature


# In[29]:

#size_of_feature['actor1']


# In[30]:

df.actor1.max()


# In[31]:

df.actor1.describe()


# In[32]:

# #from collections import defaultdict
# feature_dict = defaultdict(dict)
# for k,v in size_of_feature.items():
#     for i,j in v.items():
#         for name in v['names']:
#             feature_dict[k][name] = sum(df[k] == name)
#             #print(k,name)


# In[33]:

#feature_dict


# In[34]:

#df[df.columns[2:]].corr(method='kendall', min_periods=1)


# In[35]:

df.columns


# In[36]:

df_new = df.drop(['director','genre1', 'genre2',
       'genre3', 'actor1', 'actor2', 'actor3','budget'], axis=1,)


# In[37]:

#df_new.describe()


# In[38]:

from collections import defaultdict
new_size_of_feature = defaultdict(dict)
for column in df_new.columns:
    new_size_of_feature[column]['length'] = len(df_new[column].unique())
    new_size_of_feature[column]['names'] = df_new[column].unique()


# In[39]:

sum_of_features = 0
for k,v in new_size_of_feature.items():
    sum_of_features += v['length']


# In[40]:

sum_of_features


# In[41]:

df_new.columns


# In[42]:

df_age = pd.get_dummies(df.age, prefix='age')


# In[43]:

#df_age.head()
df_new.head()


# In[44]:

for column in df_new.columns[2:-1]:
    df_col = pd.get_dummies(df_new[column], prefix=column)
    df_new = pd.concat([df_new, df_col], axis=1)


# In[45]:

df_new = df_new.drop(['age', 'sex', 'city', 'country', 'time', 'daytype',
       'season', 'location', 'weather', 'social', 'endEmo', 'dominantEmo',
       'mood', 'physical', 'decision', 'interaction', 'movieCountry',
       'movieLanguage', 'movieYear'], axis=1)


# In[46]:

df_new.shape


# In[47]:

df_new.head()


# In[48]:

columns = df_new.columns.tolist()
columns.remove("rating")
columns.append("rating")
df_new = df_new[columns]


# In[49]:

df_new.head()


# In[50]:

#df_new.iloc[0][2:-1].tolist()


# In[53]:

from pyswarm import pso
import time


# In[ ]:

start_time = time.time()
#user_context = df_new.iloc[0][2:-1].tolist()
user_rating = df_new.rating.tolist()[:10]
def banana(x):
    lhs = 0.0
    square_error = 0.0
    for i in range(10):
        user_context = df_new.iloc[i][2:-1].tolist()
        for weight, context in zip(x, user_context):
            lhs += weight * context
        square_error += (user_rating[i] - lhs)**2
    #equation = (user_rating**2 - equation**2)/user_rating**2
    return square_error/2

def con(x):
    sum_of_weights = sum(x)
    return list([sum_of_weights - 20])

lb = [0]*178
ub = [1]*178

xopt, fopt = pso(banana,lb, ub,f_ieqcons=con,swarmsize=300,maxiter=1000)
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:

#xopt
#user_rating = df_new.rating.tolist()[:100]


# In[ ]:

fopt


# In[ ]:

rating = 0
for weight, context in zip(xopt,df_new.iloc[0][2:-1].tolist()):
    rating += context*weight
#     if context == 1:
#         print(context, weight)
print(rating)


# In[ ]:




# In[ ]:




# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math

def euclidean_distance(point1, point2):
    # Calculate the sum of the squared differences between the coordinates
    squared_diffs = [(point1[i] - point2[i]) ** 2 for i in range(len(point1))]
    sum_squared_diffs = sum(squared_diffs)
    # Take the square root of the sum to get the Euclidean distance
    distance = math.sqrt(sum_squared_diffs)
    return distance

point1 = (1, 2, 3)
point2 = (4, 5, 6)
distance = euclidean_distance(point1, point2)
print(distance)


# In[2]:


import numpy as np
def cos_sim(a, b):
 """Takes 2 vectors a, b and returns the cosine similarity according
 to the definition of the dot product
 """
 dot_product = np.dot(a, b)
 norm_a = np.linalg.norm(a)
 norm_b = np.linalg.norm(b)
 return dot_product / (norm_a * norm_b)

# the counts we computed above
sentence_m = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
sentence_h = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0])
sentence_w = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1])

# We should expect sentence_m and sentence_h to be more similar
print(cos_sim(sentence_m, sentence_h)) # 0.5
print(cos_sim(sentence_m, sentence_w)) # 0.25


# In[3]:


from math import sqrt

#create function to calculate Manhattan distance
def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

#define vectors
A = [2, 4, 4, 6]
B = [5, 5, 7, 8]

#calculate Manhattan distance between vectors
manhattan(A, B)


# In[4]:


import math

def euclidean_distance(point1, point2):
    # Calculate the sum of the squared differences between the coordinates
    squared_diffs = [(point1[i] - point2[i]) ** 2 for i in range(len(point1))]
    sum_squared_diffs = sum(squared_diffs)
    # Take the square root of the sum to get the Euclidean distance
    distance = math.sqrt(sum_squared_diffs)
    return distance

point1 = (1, 2, 3)
point2 = (4, 5, 6)
distance = euclidean_distance(point1, point2)
print(distance)


# In[ ]:





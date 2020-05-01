#!/usr/bin/env python
# coding: utf-8

# In[93]:


import numpy as np
import matplotlib.pyplot as plt
billy= {'Day1': 100,'Day2': 108,'Day3':112,'Day4':115,'Day5':150,
          'Day6':178, 'Day7': 143, 'Day8': 132, 'Day9':190, 'Day10': 235,
          'Day11':253, 'Day12': 298, 'Day13': 328, 'Day14':390, 'Day15': 257,
          'Day16':288, 'Day17': 393, 'Day18': 425, 'Day19':458, 'Day20': 450,
          'Day21':473, 'Day22': 333, 'Day23': 452, 'Day24':490, 'Day25': 495,
          'Day26':488, 'Day27': 543, 'Day28': 532, 'Day28':590, 'Day30': 605,'Day31':600}
billy.values


# In[94]:


key=0
score= [] 
for key in billy.keys() : 
    score.append (billy[key]) 


# In[95]:


print(str(billy))


# In[96]:


print(str(score))


# In[97]:


import numpy as np
day= np.array([ 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
print(day)


# In[98]:


import matplotlib.pyplot as plt
x = np.linspace(0,1,500)
y = np.linspace(0, 1, 5)
x=score
y=day
plt.ylim(1,30) 
plt.xlim(1,700)

plt.plot(x,y)
plt.xlabel('x - axis') 
 
plt.ylabel('y - axis')

plt.show


# In[100]:


med=np.median(score)
print("median=")
print(med)


# In[102]:


mi=np.min(score)
print("minimum")
print(mi)


# In[103]:


ma=np.max(score)
print("maximun")
print(ma)


# In[ ]:





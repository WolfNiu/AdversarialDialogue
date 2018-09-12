
# coding: utf-8

# In[2]:


"""
A class that handles perturbation by a certain strategy
"""
from pattern3 import en


# In[ ]:


class Strategy(object):
    def __init__(self,
                 strategy_name,
                 vocab, # avoid introducing tokens that is OOV
                )


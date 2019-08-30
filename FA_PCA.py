
# coding: utf-8

# In[21]:


import pandas as pd
from sklearn.decomposition import FactorAnalysis, PCA


# In[2]:


df = pd.read_csv('brfss_13cols.csv')


# In[65]:


df.head(5)
features = df.columns


# ## sklearn FactorAnalysis

# In[56]:


factor = FactorAnalysis().fit(df)
factors = pd.DataFrame(factor.components_)


# In[66]:


pd.DataFrame(factor.components_, columns = features)


# In[57]:


factors.idxmax(axis = 1)


# ### PCA

# In[62]:


pca = PCA().fit(df)
pcs = pd.DataFrame(pca.components_)


# In[67]:


pcs


# In[63]:


pcs.idxmax(axis = 1)



# coding: utf-8

# In[1]:


import pandas as pd
import pickle


def get_data_for_top_countries(data=arabica, n_top_countries=4):
    '''
    Takes in an integer n_top_countries indicating cutoff for countries to consider (top defined by # observations for that country).
    Output is a dataframe of observations for the top n countries only.
    '''
    countrycounts = pd.DataFrame([data['Country.of.Origin'].value_counts()]).transpose()
    countrycounts['cum_percent_total'] = countrycounts['Country.of.Origin'].cumsum()* 100 / countrycounts['Country.of.Origin'].sum()
    top_country_list = countrycounts.index[0:n_top_countries]
    
    data_subset = data[data['Country.of.Origin'].isin(top_country_list)]
    return data_subset




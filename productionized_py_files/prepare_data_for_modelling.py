
# coding: utf-8

# In[1]:


from prepare_data_fns import get_data_for_top_countries
import pandas as pd
import pickle


# In[ ]:


#define in and out filenames
csv_in = 'arabica_data_cleaned.csv'
pickle_out = 'to_model_top_4_classes.pickle'


# In[2]:


#import data from csv
arabica = pd.read_csv(csv_in, header=0, index_col=0, error_bad_lines=False)



data_subset = get_data_for_top_countries(arabica, 4)


# limiting columns to model to more understandable scores: the goal of the model is to be 
    # able to predict country of origin based on apparent characteristics, so "cupper points"
    # and similar features are too obscure to be useful. Also, total points is a linear combo of other features.
columns_to_model = ['Country.of.Origin','Aroma',
                    'Flavor', 'Aftertaste', 'Acidity', 
                    'Body', 'Balance', 'Sweetness', 
                    'Uniformity', 'Clean.Cup']


# In[6]:


to_model = data_subset[columns_to_model]


# In[7]:


#discard anomalous data with values of '0' in all columns:
to_model = to_model[to_model.Aroma!=0]


# In[8]:


#save cleaned dataframe as pickle file
with open(pickle_out, 'wb') as f:
    pickle.dump(to_model, f)


# Where-Has-Your-Coffee-Bean
In this project, I classify coffee beans' country of origin based on coffee tasting ratings using Scikit-Learn in Python. I used hyperopt to select my model and hyperparameters, ultimately choosing a support vector machine classifer. I managed data and administration with PostgreSQL.

A presentation of my findings can be found [here](https://docs.google.com/presentation/d/1Gxz0_6hMghDZ80TVuH1bbvwLoDDOI1uuEN9pEa0iN9A/edit?usp=sharing).

### Data
The coffee tasting rating data was produced by The Coffee Quality Institute. I obtained my dataset from [this repository](https://github.com/jldbc/coffee-quality-database). It contains over 1300 tastings, with a variety of features. I selected the following features for analysis:
* Acidity
* Aftertaste
* Aroma
* Body
* Flavor
* Sweetness

### Challenges
While 35 countries are represented in this dataset, the majority of them have very few observations. Because of this, I reduced the scope of my analysis to the top 4 countries, with 132-236 observations each. This is still an impractically small data set, so I focused on hyperparameter optimization to eke out the best performance I could.

### Techniques
To perform advanced hyperparameter search, I harnessed the power of *tree-structured parzen estimators*, implemented in the [Hyperopt package](https://github.com/hyperopt/hyperopt). For more information on how TPEs work, check out my [presentation introducing the technique](https://docs.google.com/presentation/d/1gjq_LDwkFDz_iJ8w9h-Rmy9Wj48ErvojnLzWaXLXpmc/edit?usp=sharing).

Using hyperopt, I compared the following models (considering a range of hyperparameters for each):
* Multinomial Logistic Regression
* K Nearest Neighbors
* Support Vector Machine Classifier
* Random Forest Classifier
* Neural Net

I used *permutation importance* to rank the features by their contribution to classification accuracy. For this technique, each feature is individually randomized and each value of the feature is reassigned to random observations. This preserves the overall distribution of the feature, while disassociating it from it's relationship with the target variable. Then the machine learning model is run over the data again, and the resulting loss in accuracy gives an idea of how important that feature was in classifying the data to begin with. Note that this does not yield absolute measures of importance, but rather relative importance, compared to each other feature.

### Findings
I found that a Support Vector Machine Classifier with a radial basis function kernel outperformed all other models. This makes some sense given the small number of observations considered, as SVMs tend to perform pretty well on wide data. Ultimately, I was able to achieve 56% accuracy over the 4 classes on the test set. As expected, given the small dataset, I was best able to classify coffee from the best-represented countries in the dataset.

Using permutation importance, I found that Balance and Flavor contributed the most information to the classification.

### Repository guide
*initial_exploration.ipynb* - jupyter notebook documenting my data exploration process

*modeling_hyperopt.ipynb* - jupyter notebook documenting the modeling process and findings, including visualizations

*make_schema_a.sql* - simple schema used to import data into PostgreSQL

*productionized_py_files* - a folder of modular python scripts, for replicating the process outlined in the jupyter notebooks
* modeling_with_hyperopt.py - modeling process script, including hyperopt model selection
* modeling_fns.py - functions used in modeling process

* prepare_data_for_modelling.py - data cleaning and preparation script
* prepare_data_fns.py - functions to clean and select the appropriate data subset

* to_model_top_4_classes.pickle - dataset created from data preparation process, for use in modeling

*coffee_flask_app* - I also created a simple flask app to showcase my model. It allows users to put in their own coffee ratings and then generates a guess of which of the 4 countries their coffee most likely came from.

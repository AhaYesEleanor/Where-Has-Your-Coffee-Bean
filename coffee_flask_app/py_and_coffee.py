import pickle
import pandas as pd
import numpy as np

# read in the model
mc_model = pickle.load(open("rfc_model.pickle","rb"))

# create a function to take in user-entered amounts and apply the model
def coffee_country(amounts_float, model=mc_model):
    
    # put everything in terms of tablespoons
    # flour, milk, sugar, butter, eggs, baking powder, vanilla, salt
    #multipliers = [16, 16, 16, 16, 3, .33, .33, .33]
    
    # sum up the total values to get the total number of tablespoons in the batter
    #total = np.dot(multipliers, amounts_float)

    # note the proportion of flour and sugar
    #flour_cups_prop = multipliers[0] * amounts_float[0] * 100.0 / total
    #sugar_cups_prop = multipliers[2] * amounts_float[2] * 100.0 / total

    # inputs into the model
    uniformity = 9.85
    clean_cup = 9.822
    cupper_points = 7.455
    total_cup_points = 81.954
    moisture = 0.092268
    cat1_defects = 0.435792
    cat2_defects = 4.165301
    
    
    
    input_df = [[amounts_float[0], 
                amounts_float[1], 
                amounts_float[2], 
                amounts_float[3], 
                amounts_float[4], 
                amounts_float[5],
                uniformity,
                clean_cup,
                amounts_float[6],
                cupper_points,
                total_cup_points,
                moisture,
                cat1_defects,
                cat2_defects 
                ]]

    # make a prediction
    prediction = mc_model.predict(input_df)[0]
    probs = mc_model.predict_proba(input_df)[0] 
    
    countries = mc_model.classes_
    class_weights = dict(zip(countries, probs))
    top_prob = class_weights[prediction]

    # create a return message
    out_message = "Out of Brazil, Columbia, Guatemala and Mexico, your coffee is most likely from {}".format(prediction) + ", with {0:.0f}% probability.".format(top_prob * 100) 



    return out_message

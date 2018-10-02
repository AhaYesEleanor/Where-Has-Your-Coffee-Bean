from flask import Flask, request, render_template
from py_and_coffee import coffee_country

# create a flask object
app = Flask(__name__)

# creates an association between the / page and the entry_page function (defaults to GET)
@app.route('/')
def entry_page():
    return render_template('index.html')

# creates an association between the /predict_recipe page and the render_message function
# (includes POST requests which allow users to enter in data via form)
@app.route('/coffee_origin/', methods=['GET', 'POST'])
def render_message():

    # user-entered ingredients
    ingredients = ['aroma', 'flavor', 'aftertaste', 'acidity', 'body', 'balance', 'sweetness']

    # error messages to ensure correct units of measure
    messages = ["The aroma must be between 0 and 10",
                "The flavor must be between 0 and 10",
                "The aftertaste must be between 0 and 10",
                "The acidity must be between 0 and 10",
                "The body must be between 0 and 10",
                "The balance must be between 0 and 10",
                "The sweetness must be between 0 and 10"]

    # hold all amounts as floats
    amounts = []

    # takes user input and ensures it can be turned into a floats
    for i, ing in enumerate(ingredients):
        user_input = request.form[ing]
        try:
            float_ingredient = float(user_input)
        except:
            return render_template('index.html', message=messages[i])
        amounts.append(float_ingredient)

    # show user final message
    final_message = coffee_country(amounts)
    return render_template('index.html', message=final_message)

if __name__ == '__main__':
    app.run(debug=True)
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
from flask import Flask
import os
app = Flask(__name__)
# model = pickle.load(open("model.pkl", "rb"))


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity = float(request.form['volatile_acidity'])
        citric_acid = float(request.form['citric_acid'])
        residual_sugar = float(request.form['residual_sugar'])
        chlorides = float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
        density = float(request.form['density'])
        pH = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])

        val = np.array([fixed_acidity,  volatile_acidity, citric_acid, residual_sugar, chlorides,
                       free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol])
        # val = np.array([0.247788,	0.397260,	0.00,	0.068493,	0.106845,
        #                0.149254,	0.098940,	0.567548,	0.606299,	0.137725,	0.153846])

        final_features = [np.array(val)]
        model_path = os.path.join('models', 'wine3.sav')
        model = pickle.load(open(model_path, 'rb'))
        res = model.predict(final_features)

        return render_template('index.html', prediction_text=res)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)

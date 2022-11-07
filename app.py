from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from waitress import serve

'''
Tar utgangspunkt i app.py fra house_price_app.zip som ligger i moduler, Uke 10 - implementasjon og tilpasser den.
'''
app = Flask(__name__)

model_dict = pickle.load(open('model.pkl', 'rb'))

model = model_dict['model']

@app.route('/')
def index():
    return render_template('./index.html')


@app.route('/predict', methods=['POST'])
def predict():

    features = dict(request.form)

    # Forventede kolonner fra templaten
    needed_cols = ['År','Måned', 'Dato', 'Ukedag', 'Klokkeslett', 'Lufttemperatur',
                   'Lufttrykk', 'Globalstraling', 'Solskinstid']

    def to_numeric(key, value, needed_cols = needed_cols):
        if key not in needed_cols:
            return value
        try:
            return float(value)
        except: 
            return nan
    
    features = {key: to_numeric(key, value) for key, value in features.items()}

    # Lager en dataframe med featurene
    features_df = pd.DataFrame(features, index=[0]).loc[:, needed_cols]
    
    # Legger til en ekstra feature engineering om det er felles ferie eller ikke.
    features_df['Fellesferie'] = features_df['Måned'].apply(lambda x : x == 7)
    features_df['Fellesferie'] = features_df['Fellesferie'].replace([True, False], [1, 0])

    # Encoder Ukedag kolonnen slik at maskinlæringsmodellen tar inn 0 og 1 istedenfor 1,2,3,4,5,6,7
    #Valgte å ikke bruke denne featuren da den ikke forbedret resultatene fra modellen.
    #features_df = pd.get_dummies(features_df, columns = ['Ukedag'])

    # Bruker modellen til å predikere på dataframen
    prediction = model.predict(features_df)
    prediction = int(prediction)
    prediction = np.clip(prediction, 0, np.inf)

    return render_template('./index.html', 
                            prediction_text='Predicted number of cycles {}'.format(prediction))

if __name__=='__main__':
    serve(app, host='0.0.0.0', port=8080)











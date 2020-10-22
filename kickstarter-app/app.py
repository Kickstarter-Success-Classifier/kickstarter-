from flask import Flask, render_template, request, jsonify, make_response
from typing import Tuple
import numpy as np
import pandas as pd
import joblib
from kickstarter_ import ImputerDF

MODEL = joblib.load('kickstarter-app/pipeline.joblib')

CATEGORIES = ['Product Design', 'Documentary', 'Music', 'Tabletop Games', 'Shorts',
              'Food', 'Video Games', 'Film & Video', 'Fiction', 'Fashion', 'Art',
              'Nonfiction', 'Theater', 'Rock', "Children's Books", 'Apparel',
              'Technology', 'Indie Rock', 'Photography', 'Webseries', 'Apps',
              'Publishing', 'Narrative Film', 'Comics', 'Country & Folk', 'Web',
              'Crafts', 'Hip-Hop', 'Design', 'Hardware', 'Pop', 'Painting',
              'Public Art', 'Illustration', 'Accessories', 'Games', 'Mixed Media',
              'Restaurants', 'Comic Books', 'Software', 'Classical Music',
              'Art Books', 'Dance', 'Animation', 'Gadgets', 'Drinks', 'Comedy',
              'Performance Art', 'Playing Cards', 'World Music']

MAIN_CATEGORIES = ['Film & Video', 'Music', 'Publishing', 'Games', 'Technology', 'Art',
                   'Design', 'Food', 'Fashion', 'Theater', 'Comics', 'Photography',
                   'Crafts', 'Journalism', 'Dance']

CURRENCIES = ['USA', 'UK', 'Europe', 'Canada', 'Australia']

def create_app():
  """Create and configure Flask application and load model"""
  app = Flask(__name__)
  
  @app.route('/')
  def root():
    return render_template("base.html", main_categories=MAIN_CATEGORIES,
                           categories=CATEGORIES, currencies=CURRENCIES)

  @app.route('/predict', methods=['GET', 'POST'])
  def predict():
    if request.method == 'POST':
      prediction = get_prediction(request.form.to_dict())
      return make_response(jsonify({'prediction': int(prediction)}),
                           200)#('SUCCESS!' if prediction else 'FAILURE',
              #200, {'Cotent-Type': 'text'})
    else:
      return "204 - No content"

  return app

def get_prediction(input: dict) -> str:
  input['name'] = len(input['name'])
  df = pd.DataFrame({key:[value] for key, value in input.items()})
  print(MODEL.predict(df))
  print(df)
  return MODEL.predict(df)[0] 
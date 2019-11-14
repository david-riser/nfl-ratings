"""
Train a logistic regression for prediction of NFL weekly odds.
This script also does inference, when supplied a trained model.
"""

# Python lib 
import os
from datetime import date 

# Third party 
import argparse 
import numpy as np 
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 


def validate_args(args):
    if not args.train and args.predict and not args.model:
        print(("Cannot proceed, no model to use for prediction."
               "Please pass a model with keyword --model or try"
               "training one from scratch with --train."))
        exit()

def load_clean_dataset():
    """ Load and clean our dataset downloaded from 
    the download_dataset.py file. """
    data_dir = '../data/'
    data = pd.read_csv(data_dir + 'historical_data.csv')
    data = data[np.logical_not(data['score1'].isna())]
    data = data[np.logical_not(data['score2'].isna())]

    data = data[data['season'] > 1959]
    data = data[data['score1'] != data['score2']]
    data['date'] = pd.to_datetime(data['date'])    

    add_elo_features(data)
    add_targets(data)
    return data

def load_weekly_preds():
    data_dir = '../data/'
    data = pd.read_csv(data_dir + 'historical_data.csv')
    data['date'] = pd.to_datetime(data['date'])
    add_elo_features(data)
    add_targets(data)
    
    today = pd.Timestamp(date.today())
    next_week = today + pd.Timedelta(1, unit='w')
    weekly_preds = data[np.logical_and(data['date'] >= today,
                                       data['date'] < next_week)]
    return weekly_preds

def add_elo_features(data):
    """ Based on the elo rating for the two teams, 
        create new features.  Here, elo1 is assumed
        to be the home team.
        
        The data is modified inplace.
    """
    data['elo_sum'] = data['elo1_pre'] + data['elo2_pre']
    data['elo_diff'] = data['elo1_pre'] - data['elo2_pre']
    data['elo_asym'] = data['elo_diff'] / data['elo_sum']

def add_targets(data):
    """ Add targets based on the game outcome. """
    
    # Did the home team win? Note, ties are removed.
    data['outcome'] = data['score1'] > data['score2']
    data['outcome'] = data['outcome'].astype(np.int)
    
    data['point_sum'] = data['score1'] + data['score2']
    data['point_diff'] = data['score1'] - data['score2']
    data['point_asym'] = data['point_diff'] / data['point_sum']

def train_model(data, features, target):
    """ Training logistic regression model. """

    x_train, x_test, y_train, y_test = train_test_split(
        data[features].values, data[target].values)

    model = LogisticRegression(solver='lbfgs')
    model.fit(x_train, y_train.ravel())

    # Metric report 
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    score_train = accuracy_score(y_train, y_pred_train)
    score_test = accuracy_score(y_test, y_pred_test)

    print(("Training Accuracy {0:6.4f},"
            " Testing Accuracy {1:6.4f}".format(
                score_train, score_test
            )))
    
    return model 

def predict(data, model, features):
    """ Predictions added to dataframe inplace. """
    x = data[features].values
    return model.predict_proba(x)[:,1]

def print_predictions(data):
    """ Summarize weekly predictions. """

    output_template = ("{0} {1:4s}({2:4d}) {3:4.2f}%,"
                       "{4:4s}({5:4d}) {6:4.2f}%")
    
    # Yes this is slow, but there are few
    # games so it doesn't matter at all. 
    for index, game in data.iterrows():
        print(output_template.format(
            game['date'],
            game['team1'],
            int(game['elo1_pre']),
            100 * game['preds'],
            game['team2'],
            int(game['elo2_pre']),
            100 * (1 - game['preds'])))
        
if __name__ == "__main__":

    # Basic application setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='run in training mode')
    parser.add_argument('--predict', action='store_true',
                        help='run in prediction mode')
    parser.add_argument('--model', type=str, required=False,
                        help='trained model file')
    parser.add_argument('--save_name', type=str, default='model.pkl',
                        help='save name for output model')

    # Get and validate the configuration 
    args = parser.parse_args()
    validate_args(args)

    # Training options 
    features = ['elo_sum', 'elo_diff', 'elo_asym', 'elo_prob1']
    target = ['outcome']

    # Run training and prediction, if required. 
    if args.train:
        print('Running in training mode.')
        data = load_clean_dataset() 
        model = train_model(data, features, target)
        dump(model, args.save_name)
        
    if args.predict:
        print('Running in prediction mode.')
        data = load_weekly_preds()

        if not args.train:
            model = load(args.model)

        preds = predict(data, model, features)
        data['preds'] = preds
        print_predictions(data)

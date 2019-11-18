""" Glicko rating system implementation. """

from collections import namedtuple 

import numpy as np
import pandas as pd

GlickoRating = namedtuple('GlickoRating', 'value deviation')
CONST_Q = 0.0057564
CONST_C = 25.0
CONST_START_STD = 350
CONST_START_MEAN = 1500

def predict(team1, team2):
    """ Predict the probability that team1 beats team2. """
    g2 = g_term(team2.deviation)
    return 1.0 / (1.0 + 10**(-1 * g2 * (team1.value - team2.value) / 400.0))
    
def g_term(deviation):
    """ G term calculation given team rating deviation.  Can't think of
    a better name for this function. 
    
    See: https://en.wikipedia.org/wiki/Glicko_rating_system """
    return 1.0 / np.sqrt(1 + 3 * CONST_Q**2 * deviation**2 / np.pi**2)

def d_term(team1, team2):
    """ D term calculated for one game. The value is already squared. """
    prob = predict(team1, team2)
    denom = CONST_Q**2 * g_term(team2.deviation)**2 * prob * (1 - prob)
    return 1.0 / denom

def update_rating(team1, team2, outcome):
    """ Return the updated rating for the outcome.  
    outcome = 0, team 1 lost
    outcome = 0.5, draw
    outcome = 1, team 1 won 
    """
    prefactor = CONST_Q / (1 / team1.deviation**2 + 1 / d_term(team1, team2))
    return team1.value + prefactor * g_term(team2.deviation) * (outcome - predict(team1, team2))

def expand_deviation(deviation, time_off):
    """ After a time_off in weeks, we increase our uncertainty about
    the player rating. """
    return min(CONST_START_STD, np.sqrt(deviation**2 + CONST_C**2 * time_off))

def shrink_deviation(team1, team2):
    """ Reduce our uncertainty in the std of team 1. """
    return 1.0 / np.sqrt(1 / team1.deviation**2 + 1 / d_term(team1, team2))

def simulate_game(team1, team2, outcome):

    print('Test case: team1 = ({},{}), team2 = ({},{})'.format(
        team1.value, team1.deviation, team2.value, team2.deviation
    ))
    print('P(1 > 2) = {0:6.4f}'.format(predict(team1, team2)))

    # Team 1 Wins
    team1_new_value = update_rating(team1, team2, outcome)
    team1_new_deviation = shrink_deviation(team1, team2)
    team1 = GlickoRating(value=team1_new_value, deviation=team1_new_deviation)

    rev_outcome = 0 if outcome == 1 else 1
    if outcome == 0.5:
        rev_outcome = outcome
        
    team2_new_value = update_rating(team2, team1, rev_outcome)
    team2_new_deviation = shrink_deviation(team2, team1)
    team2 = GlickoRating(value=team2_new_value, deviation=team2_new_deviation)
    
    print('Simulated outcome: ', outcome)
    print('\tUpdated ratings: team1 = ({},{})'.format(team1.value, team1.deviation))
    print('\tUpdated ratings: team2 = ({},{})'.format(team2.value, team2.deviation))    

class Glicko:

    def __init__(self, init_mean=1500, init_std=350, c=34.0):
        self.init_mean = init_mean
        self.init_std = init_std
        self.c = c
        self.ratings = {}        
            
if __name__ == "__main__":

    #team1 = GlickoRating(value=1500, deviation=CONST_START_STD)
    #team2 = GlickoRating(value=1500, deviation=CONST_START_STD)

    #simulate_game(team1, team2, 1)
    #simulate_game(team1, team2, 0)

    from model import load_clean_dataset
    data = load_clean_dataset()

    # Active teams since 1960.
    teams = set(data['team1'].unique()) | set(data['team2'].unique())
    ratings = {}
    last_season = {} 
    for team in teams:
        ratings[team] = GlickoRating(value=CONST_START_MEAN, deviation=CONST_START_STD)
        last_season[team] = 1960
        
    # Run the games forward in time. 
    glicko_prob = np.zeros(len(data))
    mu1 = np.zeros(len(data))
    mu2 = np.zeros(len(data))
    sig1 = np.zeros(len(data))
    sig2 = np.zeros(len(data))
    for i in range(len(data)):
        
        # Game details 
        team1 = data.iloc[i]['team1']
        team2 = data.iloc[i]['team2']
        score1 = data.iloc[i]['score1']
        score2 = data.iloc[i]['score2']

        # Expand
        time_off1 = 35 * (data.iloc[i]['season'] - last_season[team1]) + 1
        time_off2 = 35 * (data.iloc[i]['season'] - last_season[team2]) + 1
        last_season[team1] = data.iloc[i]['season']
        last_season[team2] = data.iloc[i]['season']
        ratings[team1] = GlickoRating(value=ratings[team1].value,
                                      deviation=expand_deviation(ratings[team1].deviation, time_off1))
        ratings[team2] = GlickoRating(value=ratings[team2].value,
                                      deviation=expand_deviation(ratings[team2].deviation, time_off2))

        
        # Game outcome 
        outcome = 0.5
        if score1 > score2:
            outcome = 1
        elif score2 > score1:
            outcome = 0

        rev_outcome = 0.5
        if outcome == 0:
            rev_outcome = 1
        elif outcome == 1:
            rev_outcome = 0
            
        prob1 = predict(ratings[team1], ratings[team2])
        glicko_prob[i] = prob1
        new_rating1 = update_rating(ratings[team1], ratings[team2], outcome)
        new_rating2 = update_rating(ratings[team2], ratings[team1], rev_outcome)
        new_dev1 = shrink_deviation(ratings[team1], ratings[team2])
        new_dev2 = shrink_deviation(ratings[team2], ratings[team1])

        ratings[team1] = GlickoRating(value=new_rating1, deviation=new_dev1)
        ratings[team2] = GlickoRating(value=new_rating2, deviation=new_dev2)

        mu1[i] = ratings[team1].value 
        sig1[i] = ratings[team1].deviation
        mu2[i] = ratings[team2].value 
        sig2[i] = ratings[team2].deviation

    data['glicko_rating1'] = mu1
    data['glicko_rating2'] = mu2
    data['glicko_dev1'] = sig1
    data['glicko_dev2'] = sig2
    data['glicko_prob1'] = glicko_prob

    data.to_csv('data/glicko.csv', index=False)

""" Glicko rating system implementation. """

import os 
from collections import namedtuple 

import numpy as np
import pandas as pd

GlickoRating = namedtuple('GlickoRating', 'value deviation')
CONST_Q = 0.0057564

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

def expand_deviation(deviation, time_off, init_std=350, c=34.5):
    """ After a time_off in weeks, we increase our uncertainty about
    the player rating. """
    return min(init_std, np.sqrt(deviation**2 + c**2 * time_off))

def shrink_deviation(team1, team2):
    """ Reduce our uncertainty in the std of team 1. """
    return 1.0 / np.sqrt(1 / team1.deviation**2 + 1 / d_term(team1, team2))

def glicko(teams1, teams2, outcomes, seasons, init_mean=1500,
           init_std=350, c=34.5):
    """ Run a Glicko model for these teams and 
    return the results in np.ndarray. """

    # Get active teams and initialize 
    teams = set(teams1) | set(teams2)
    ratings = {}
    last_season = {}
    first_season = min(seasons)
    for team in teams:
        ratings[team] = GlickoRating(value=init_mean, deviation=init_std)
        last_season[team] = first_season 
        
    # Run the games forward in time. 
    glicko_prob = np.zeros(len(teams1))
    mu1 = np.zeros(len(teams1))
    mu2 = np.zeros(len(teams1))
    sig1 = np.zeros(len(teams1))
    sig2 = np.zeros(len(teams1))

    for i in range(len(teams1)):
        
        # Game details 
        team1 = teams1[i]
        team2 = teams2[i]
        outcome = outcomes[i]
        season = seasons[i]
        
        # Expand
        time_off1 = 35 * (season - last_season[team1]) + 1
        time_off2 = 35 * (season - last_season[team2]) + 1
        last_season[team1] = season
        last_season[team2] = season
        ratings[team1] = GlickoRating(value=ratings[team1].value,
                                      deviation=expand_deviation(ratings[team1].deviation, time_off1,
                                                                 init_std=init_std, c=c))
        ratings[team2] = GlickoRating(value=ratings[team2].value,
                                      deviation=expand_deviation(ratings[team2].deviation, time_off2,
                                                                 init_std=init_std, c=c))

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
        
    return mu1, mu2, sig1, sig2, glicko_prob

def brier_score(pred, outcome):
    return np.sum(25 - 100 * (outcome - pred)**2)

if __name__ == "__main__":

    from model import load_clean_dataset
    from sklearn.metrics import accuracy_score
    data = load_clean_dataset()
    valid_idx = np.where(data['season'] > 2015)[0]

    cs = np.random.uniform(0.1, 400, size=100)
    for current_c in cs:
        val1, val2, dev1, dev2, prob = glicko(
            data['team1'].values, data['team2'].values,
            data['outcome'].values, data['season'].values,
            init_mean=1500, init_std=350, c=current_c)
        
        data['glicko_prob'] = prob 
        
        print(current_c,
              accuracy_score(data['outcome'], data['glicko_prob'].round()),
              accuracy_score(data.iloc[valid_idx]['outcome'], data.iloc[valid_idx]['glicko_prob'].round()),
              brier_score(data['outcome'], data['glicko_prob']),
              brier_score(data.iloc[valid_idx]['outcome'], data.iloc[valid_idx]['glicko_prob']))
    

""" Glicko rating system implementation. 


To Do: 
------
1) Right now model.py cannot make predictions 
for weekly because we don't have the glicko 
predictions for the next games.  This data is 
dropped here by doing score1.dropna()...

"""

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
           init_std=350, c=34.5, iterations=1):
    """ Run a Glicko model for these teams and 
    return the results in np.ndarray. """

    # Get active teams and initialize 
    teams = set(teams1) | set(teams2)
    ratings = {}
    last_season = {}
    first_season = min(seasons)

    for current_iter in range(iterations):
        for team in teams:
            ratings[team] = ratings.get(team, GlickoRating(
                value=init_mean, deviation=init_std))

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
        
    return mu1, mu2, sig1, sig2, glicko_prob, ratings

def brier_score(pred, outcome):
    return np.sum(25 - 100 * (outcome - pred)**2)

def binary_cross_entropy(pred, outcome):
    """ BCE Score (often used as loss) """
    class0_idx = np.where(outcome == 0)[0]
    class1_idx = np.where(outcome == 1)[0]
    return - np.sum(np.log(1 - pred[class0_idx])) - np.sum(np.log(pred[class1_idx]))
    
if __name__ == "__main__":

    from model import load_clean_dataset, load_weekly_preds
    from sklearn.metrics import accuracy_score
    data = load_clean_dataset(start_year=2014)
    valid_idx = np.where(data['season'] > 2015)[0]

    """
    This analysis shows that a low value of C is better, 
    preseving ratings over off-seasons.  It also shows that 
    a few passes of the algorithm are better.
    C = 5.0, iterations = 5 will be used. 

    cs = np.random.uniform(0.1, 40, size=50)
    iterations = np.random.randint(1,20,size=50)
    for current_c, current_iter in zip(cs, iterations):
        val1, val2, dev1, dev2, prob = glicko(
            data['team1'].values, data['team2'].values,
            data['outcome'].values, data['season'].values,
            init_mean=1500, init_std=350, c=current_c, iterations=current_iter)
        
        data['glicko_prob'] = prob 

        fmt = '{0:6.4f} {1:4d} {2:6.4f} {3:6.4f} {4:6.4f}'
        print(fmt.format(
            current_c, current_iter,
            accuracy_score(data.iloc[valid_idx]['outcome'], data.iloc[valid_idx]['glicko_prob'].round()),
            brier_score(data.iloc[valid_idx]['outcome'], data.iloc[valid_idx]['glicko_prob']),
            binary_cross_entropy(data.iloc[valid_idx]['glicko_prob'].values,
                                 data.iloc[valid_idx]['outcome'].values)
        ))
     """ 

    val1, val2, dev1, dev2, prob, ratings = glicko(
        data['team1'].values, data['team2'].values,
        data['outcome'].values, data['season'].values,
        init_mean=1500, init_std=350, c=5.0, iterations=5)

    data['glicko_prob'] = prob 

    # Save for training of ML model in next step.
    save_dir = os.path.normpath(
        os.path.dirname(os.path.abspath(__file__)) + '/../data/glicko.csv')
    data.to_csv(save_dir, index=False)

    # Save for prediction.
    weekly = load_weekly_preds()
    gprob = np.zeros(len(weekly))

    # Predict for this week
    for i, game in weekly.iterrows():
        gprob[i] = predict(ratings[game['team1']], ratings[game['team2']])

    weekly['glicko_prob'] = gprob
    save_dir = os.path.normpath(
        os.path.dirname(os.path.abspath(__file__)) + '/../data/glicko_weekly.csv')
    weekly.to_csv(save_dir, index=False)
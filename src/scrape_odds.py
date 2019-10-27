"""

Scrape historical NFL odds for predicting the
outcome of future games based on old odds.

"""

from collections import namedtuple
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup



GameOdds = namedtuple('GameOdds', 'home_team away_team spread date id_code')

def info_is_bad(info):
    return (info == '') or ('XX' in info) or ('PK' in info)

def get_spread(home_info, away_info):
    home_bad = info_is_bad(home_info)
    away_bad = info_is_bad(home_info)

    if home_bad or away_bad:
        return None
    else:
        if 'u' in home_info:
            return float(away_info.split()[0])
        elif 'u' in away_info:
            return float(home_info.split()[0])

def create_df(odds):
    df = pd.DataFrame.from_records(
        odds,
        columns=GameOdds._fields
    )
    return df


def process_table_entry(table_entry):

    # The hyperlink contains some of the information we want.
    link_tokens = table_entry['href'].split('/')
    teams = link_tokens[5]
    date_info = link_tokens[7]

    # Find the home and away team
    home_team = teams.split('-@-')[-1].split('.')[0]
    away_team = teams.split('-@-')[0]

    # Find the date and ID
    date = date_info.split('#')[0]
    id_code = date_info.split('#')[-1]

    # The br tag contains other info
    children = list(table_entry.children)
    away_info = children[2].replace('½', '.5').strip()
    home_info = children[4].replace('½', '.5').strip()
    spread = get_spread(home_info, away_info)

    return GameOdds(
        home_team=home_team,
        away_team=away_team,
        date=date,
        id_code=id_code,
        spread=spread
    )

def get_table(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    return soup.find_all('td', {'class': 'viCellBg1 cellTextNorm cellBorderL1 center_text nowrap'})

if __name__ == '__main__':

    print('Starting scrape.')
    base_url = 'https://www.vegasinsider.com/nfl/odds/las-vegas'

    odds = []
    for page_ext in ['', '/2/']:
        url = base_url + page_ext
        print(f'Scraping {url}.')

        # Get the page and remove all of the table data
        # in list format.
        for td in get_table(url):

            # Each row has an a tag inside of it
            table_entry = td.find('a', {'class':'cellTextNorm'})
            if table_entry is not None:
                odd = process_table_entry(table_entry)
                odds.append(odd)

    df = create_df(odds)
    df.to_csv(f'../data/{date.today()}-latest_odds.csv', index=False)
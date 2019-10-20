#
#    David Riser
#    Sept. 21, 2019
#    download_dataset.py
#
#    The blog FiveThirtyEight has made historical as well as recent
#    data from NFL games available.  In this script, the latest files
#    are downloaded to the data/ directory in the top folder of this
#    project.
#

import os
from urllib import request

if __name__ == '__main__':

    # Build the correct directory structure
    # for the project if it doesn't exist.
    data_dir = os.path.normpath(
        os.path.dirname(os.path.abspath(__file__)) + '/../data'
    )
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    print(f'Downloading NFL datasets into {data_dir}...')

    # Data provided by FiveThirtyEight, https://fivethirtyeight.com/
    history_url = 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv'
    latest_url = 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo_latest.csv'

    # Download each file, if the directory structure is changed, these lines will
    # need to change as well.
    request.urlretrieve(history_url, f'{data_dir}/historical_data.csv')
    request.urlretrieve(latest_url, f'{data_dir}/latest_data.csv')
    print('Done downloading data!')

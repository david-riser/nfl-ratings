import dash
import dash_table
import os 
import pandas as pd

project_dir = os.path.dirname(os.path.abspath(__file__)) + '/../'
df = pd.read_csv(os.path.join(project_dir, 'data/glicko.csv'))

display_cols = ['team1', 'team2', 'season', 'date', 'elo_prob1', 'glicko_prob',
                'outcome']

app = dash.Dash(__name__)

app.layout = dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in display_cols],
    data=df.to_dict("rows"),
)

if __name__ == '__main__':
    app.run_server(debug=True)

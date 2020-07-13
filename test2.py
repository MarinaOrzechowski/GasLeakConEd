import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

import os

from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import manifold
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, meta_tags=[
    {"content": "width=device-width, initial-scale=1.0"}
], external_stylesheets=external_stylesheets)


mapbox_access_token = 'pk.eyJ1IjoibWlzaGtpY2UiLCJhIjoiY2s5MG94bWRoMDQxdjNmcHI1aWI1YnFkYyJ9.eFsHqEMYY7qxa0Pb9USCtQ'
mapbox_style = "mapbox://styles/mishkice/ckbjhq6w50hlc1io4cnqg7svc"

# Load data
dir_path = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(
    dir_path + '\data\processed\important_(used_in_app)\Merged_asc_fdny_data.csv')
centers_df = pd.read_csv(
    dir_path + '\data\processed\important_(used_in_app)\geoid_with_centers.csv')

print(df['incident_year'].unique())

# bins
BINS = [
    "0-0.0015",
    "0.00151-0.003",
    "0.0031-0.0045",
    "0.00451-0.006",
    "0.0061-0.0075",
    "0.00751-0.009",
    "0.0091-10"
]

# colors
DEFAULT_COLORSCALE = [
    "#eff3ff",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#084594"
]

DEFAULT_OPACITY = 0.8


colors = {
    'background': '#F8F8F8',
    'text': '#000000',
    'text2': '#000000',
    'border': '#000000',
    'chart': ['#27496d', '#00909e', '#4d4c7d']
}

# function to assign colors to markers by boroughs


'''def find_colorscale_by_boro(df):
    color_by_boro = ['#6a2c70' if row['boro_name'] == 'manhattan' else '#b83b5e' if row['boro_name'] == 'brooklyn' else '#f08a5d' if row['boro_name'] ==
                     'queens' else '#f9ed69' if row['boro_name'] == 'staten island' else '#3ec1d3' for index, row in df.iterrows()]
    return color_by_boro
colorscale_by_boro = ['#6a2c70', '#b83b5e', '#f08a5d', '#f9ed69', '#3ec1d3']'''


# page layout
app.layout = html.Div(
    html.Div([

        # row with the header
        html.Div([
            html.H1(
                children='NYC Gas Leaks Information',
                style={
                    'textAlign': 'center',
                    'color': colors['text'],
                    'paddingTop':'1%',
                    'paddingBottom': '3%'
                })
        ], className='row'),
        # row 1 with the dropdowns
        html.Div([
            # dropdown to choose neighborhoods
            html.Div([
                dcc.Dropdown(
                    id='dropdownNta',
                    options=[
                        {
                            'label': i, 'value': i
                        } for i in centers_df['nta'].unique()
                    ],
                    multi=True,
                    placeholder='Choose neighborhoods')
            ],
                className='six columns',
                style={'display': 'inline-block'}),

            # dropdown to choose attributes
            html.Div([
                dcc.Dropdown(
                    id="dropdownAttr",
                    options=[
                        {
                            'label': i, 'value': i
                        } for i in df.columns[3:] if i != 'median_houshold_income'
                    ],
                    value=['gas_leaks_per_person',
                           'median_age', 'lonely_housholder_over65%'],
                    multi=True,
                    placeholder="Select attributes",
                    style={'display': 'inline-block', 'width': '100%'},
                )
            ],
                className='six columns',
                style={'display': 'inline-block'}),

        ],
            className='row'),

        html.Div([

            # range slider to choose years
            html.Div([
                dcc.RangeSlider(
                    id="timeline",
                    # min=df['incident_year'].min(),
                    # max=df['incident_year'].max(),
                    # step=1,
                    # marks={year: str(year)
                    #       for year in df['incident_year'].unique()},
                    #value=[max, max]
                    min=2013,
                    max=2018,
                    step=1,
                    marks={2013: '2013', 2014: '2014', 2015: '2015',
                           2016: '2016', 2017: '2017', 2018: '2018'},
                    value=[2018, 2018]


                )
            ],
                className='six columns',
                style={'float': 'left'}
            ),

            # dropdown to choose type of graph
            html.Div([
                dcc.Dropdown(
                    id="dropdownGraph",
                    options=[
                        {
                            "label": "Scatter matrix (pairwise comparison)",
                            "value": "scatter",
                        },
                        {
                            "label": "PCA",
                            "value": "pca",
                        },
                        {
                            "label": "ISOMAP",
                            "value": "isomap",
                        }
                    ],
                    value='scatter',
                    multi=False,
                    placeholder="Select type of graph",
                    style={'width': '100%'},
                )
            ],
                className='six columns',
                style={'float': 'right'}
            )

        ],
            className='row'),


    ],
        style={'backgroundColor': colors['background']})
)


if __name__ == '__main__':
   # app.config['suppress_callback_exceptions'] = True
    app.run_server(debug=True)

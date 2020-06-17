import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

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

df_gas_leaks = pd.read_csv(
    "C:/Users/mskac/machineLearning/GasLeakConEd/data/data_with_centroids.csv")

# trees neighborhoods bins
BINS = [
    "0-0.0015",
    "0.00151-0.003",
    "0.0031-0.0045",
    "0.00451-0.006",
    "0.0061-0.0075",
    "0.00751-0.009",
    "0.0091-10"
]

# trees neighborhoods colors
DEFAULT_COLORSCALE = [
    "#f0f9e8",
    "#ccebc5",
    "#a8ddb5",
    "#7bccc4",
    "#4eb3d3",
    "#2b8cbe",
    "#08589e"
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


def find_colorscale_by_boro(df):
    color_by_boro = ['#6a2c70' if row['borough'] == 'manhattan' else '#b83b5e' if row['borough'] == 'brooklyn' else '#f08a5d' if row['borough'] ==
                     'queens' else '#f9ed69' if row['borough'] == 'staten island' else '#3ec1d3' for index, row in df.iterrows()]
    return color_by_boro


colorscale_by_boro = ['#6a2c70', '#b83b5e', '#f08a5d', '#f9ed69', '#3ec1d3']


# page layout
app.layout = html.Div(
    html.Div(style={'backgroundColor': colors['background']}, children=[

        html.Div([
            html.H1(
                children='Gas Leak Information',
                style={
                    'textAlign': 'center',
                    'color': colors['text'],
                    'paddingTop':'1%',
                    'paddingBottom': '3%'
                })
        ], className='row'),

        html.Div(
            [
                html.Div([
                    html.P(children="Choose attribute:")
                ], style={'display': 'inline-block', 'paddingRight': 18}),

                html.Div([
                    dcc.RadioItems(
                        id='choiceAttr',
                        options=[
                            {'label': 'Gas Leaks per Person',
                             'value': 'leaks'},
                        ],
                        # value='boroughs',
                        labelStyle={
                            'color': colors['text'], 'backgroundColor': colors['background'],
                            'display': 'inline-block',
                            'paddingRight': 10}
                    )
                ], style={'display': 'inline-block'})

            ],
            className='row',
            style={'marginTop': 0, 'marginLeft': '2%', 'width': '40%',
                   'color': colors['text']}
        ),
        ######################################
        html.Div([
            html.Div([
                dcc.Graph(
                    id='mapGraph',
                    figure=dict(

                        layout=dict(
                            mapbox=dict(
                                layers=[],
                                accesstoken=mapbox_access_token,
                                style=mapbox_style,
                                center=dict(
                                    lat=40.7342,
                                    lon=-73.91251
                                ),
                                pitch=0,
                                zoom=9,
                            ),
                            autosize=False,
                        ),
                    ),
                )], className='six columns', style={'width': '40%', 'paddingLeft': '2%', 'paddingBottom': 10, 'marginBottom': 10}),  # left half ends here

            html.Div([
                html.Div([
                    dcc.Graph(
                        id='scatter_matrix'
                    )

                ], className='row'),

            ], className='six columns', style={'width': '50%', 'paddingLeft': '3%', 'paddingRight': '1%', 'marginTop': 0, 'paddingTop': 0})  # right half ends here

        ], className='row', style={'width': '100%'}),  # big row ends here
    ]))

# callbacks

######################################################################################################################
# map callback
######################################################################################################################


@ app.callback(
    Output("mapGraph", "figure"),
    [Input("choiceAttr", "value")],
    [State("mapGraph", "figure")],
)
def display_map(choiceAttr, figure):
    annotations = [
        dict(
            showarrow=False,
            align="right",
            text="gas leaks per person",
            font=dict(color="#000000"),
            bgcolor=colors['background'],
            x=0.95,
            y=0.95,
        )
    ]

    bins = BINS
    colorscale = DEFAULT_COLORSCALE
    latitude = df_gas_leaks["centerLat"]
    longitude = df_gas_leaks["centerLong"]
    hover_text = df_gas_leaks["hover"]
    base = "https://raw.githubusercontent.com/MarinaOrzechowski/GasLeakConEd/master/data/geolayers/gasleaks_per_person/"

    cm = dict(zip(bins, colorscale))
    data = [
        dict(
            lat=latitude,
            lon=longitude,
            text=hover_text,
            type="scattermapbox",
            hoverinfo="text",
            marker=dict(size=5, color="black", opacity=0),
        )
    ]

    for i, bin in enumerate(reversed(bins)):
        color = cm[bin]
        annotations.append(
            dict(
                arrowcolor=color,
                text=bin,
                x=0.95,
                y=0.85 - (i / 20),
                ax=-60,
                ay=0,
                arrowwidth=5,
                arrowhead=0,
                bgcolor="#F8F8F8",
                font=dict(color='#000000'),
            )
        )

    if "layout" in figure:
        lat = figure["layout"]["mapbox"]["center"]["lat"]
        lon = figure["layout"]["mapbox"]["center"]["lon"]
        zoom = figure["layout"]["mapbox"]["zoom"]
    else:
        lat = (40.7342,)
        lon = (-73.91251,)
        zoom = 10

    layout = dict(
        mapbox=dict(
            layers=[],
            accesstoken=mapbox_access_token,
            style=mapbox_style,
            center=dict(lat=lat, lon=lon),
            zoom=zoom,
        ),
        height=900,
        transition={'duration': 500},
        hovermode="closest",
        margin=dict(r=0, l=0, t=0, b=0),
        annotations=annotations,
        dragmode="lasso"
    )

    for bin in bins:
        geo_layer = dict(
            sourcetype="geojson",
            source=base + bin + ".geojson",
            type="fill",
            color=cm[bin],
            opacity=DEFAULT_OPACITY,
            # CHANGE THIS
            fill=dict(outlinecolor="#afafaf"),
        )
        layout["mapbox"]["layers"].append(geo_layer)

    fig = dict(data=data, layout=layout)

    return fig


######################################################################################################################
# scattermatrix callback
######################################################################################################################


if __name__ == '__main__':

    app.run_server(debug=True)

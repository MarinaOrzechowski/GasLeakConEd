import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from itertools import combinations

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

df = pd.read_csv(
    r'C:\Users\mskac\machineLearning\GasLeakConEd\data\data_with_centroids_updated.csv')

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


def find_colorscale_by_boro(df):
    color_by_boro = ['#6a2c70' if row['boro_name'] == 'manhattan' else '#b83b5e' if row['boro_name'] == 'brooklyn' else '#f08a5d' if row['boro_name'] ==
                     'queens' else '#f9ed69' if row['boro_name'] == 'staten island' else '#3ec1d3' for index, row in df.iterrows()]
    return color_by_boro


colorscale_by_boro = ['#6a2c70', '#b83b5e', '#f08a5d', '#f9ed69', '#3ec1d3']


# page layout
app.layout = html.Div(
    html.Div([

        # row with the header
        html.Div([
            html.H1(
                children='NYC Trees & Properties ',
                style={
                    'textAlign': 'center',
                    'color': colors['text'],
                    'paddingTop':'1%',
                    'paddingBottom': '3%'
                })
        ], className='row'),

        # row with the dropdowns
        html.Div([
            # dropdown to choose neighborhoods
            html.Div([
                dcc.Dropdown(
                    id='dropdownNta',
                    options=[
                        {
                            'label': i, 'value': i
                        } for i in df['ntaname'].unique()
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
                            "label": "Poverty Rate",
                            "value": "povertyrate",
                        },
                        {
                            "label": "Unemployment Rate",
                            "value": "unemplrate",
                        },
                        {
                            "label": "Number of crimes per person",
                            "value": "crimes_person",
                        },
                        {
                            "label": "Number of reported to FDNY gas leask per person",
                            "value": "gas_leaks_person",
                        },
                    ],
                    value=['gas_leaks_person',
                           'crimes_person', 'povertyrate'],
                    multi=True,
                    placeholder="Select attributes",
                    style={'display': 'inline-block'},
                )
            ],
                className='six columns',
                style={'display': 'inline-block'})
        ],
            className='row'),

        # row with a map and a matrix
        html.Div([
            # map
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
                                zoom=11,
                            ),
                            autosize=False,
                        ),
                    ),
                )
            ],
                className='six columns',
                style={'display': 'inline-block'}),

            # matrix
            html.Div([
                dcc.Graph(
                    id='scatter_matrix'
                )
            ],
                className='six columns',
                style={'display': 'inline-block'})
        ],
            className='row'),

        # row with parallel coordinates
        html.Div([],
                 className='row'),

    ],
        style={'backgroundColor': colors['background']})
)


# callbacks

######################################################################################################################
# map callback
######################################################################################################################


@ app.callback(
    Output("mapGraph", "figure"),
    [Input("dropdownNta", "value")],
    [State("mapGraph", "figure")],
)
def display_map(choiceMap, figure):
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
    latitude = df["centerLat"]
    longitude = df["centerLong"]
    hover_text = df["hover"]
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


@app.callback(
    Output('scatter_matrix', 'figure'),
    [
        Input('mapGraph', 'selectedData'),
        Input('dropdownNta', 'value'),
        Input('dropdownAttr', 'value')
    ])
def display_selected_data(selectedAreaMap, selectedAreaDropdown, selectedAttr):

    df_selected = df
    title_part = ' census tracks'
    key = 'geoid'

    font_ann = dict(
        size=10,
        color=colors['text']
    )

    if selectedAreaMap is not None:
        points = selectedAreaMap["points"]
        area_names = [str(point["text"].split("<br>")[2])
                      for point in points]
        df_selected = df_selected[df_selected[key].isin(area_names)]

    index_vals = df_selected['boro_name'].astype('category').cat.codes
    coef_list = []

    # find pearson coeff and p_value for each pair of attributes
    pairs = combinations(selectedAttr, 2)
    flag = True
    for pair in pairs:
        if len(df_selected[pair[0]]) >= 2 and len(df_selected[pair[1]]) >= 2:
            coef_list.append(
                pearsonr(df_selected[pair[0]], df_selected[pair[1]]))
        else:
            flag = False
    if flag:
        ann = [
            dict(
                x=1,
                y=1,
                xref="x2",
                yref="y1",
                font=font_ann,
                text="PCC: " +
                str(round(coef_list[0][0], 2)) + "<br>p: " +
                ('{:0.1e}'.format(coef_list[0][1])),
                showarrow=False,

            ),
            dict(
                x=1,
                y=1,
                xref="x1",
                yref="y2",
                font=font_ann,
                text="PCC: " +
                str(round(coef_list[0][0], 2)) + "<br>p: " +
                ('{:0.1e}'.format(coef_list[0][1])),
                showarrow=False,
            ),
            dict(
                x=1,
                y=1,
                xref="x3",
                yref="y1",
                font=font_ann,
                text="PCC: " +
                str(round(coef_list[1][0], 2)) + "<br>p: " +
                ('{:0.1e}'.format(coef_list[1][1])),
                showarrow=False,
            ),
            dict(
                x=1,
                y=1,
                xref="x1",
                yref="y3",
                font=font_ann,
                text="PCC: " +
                str(round(coef_list[1][0], 2)) + "<br>p: " +
                ('{:0.1e}'.format(coef_list[1][1])),
                showarrow=False,
            ),
            dict(
                x=1,
                y=1,
                xref="x3",
                yref="y2",
                font=font_ann,
                text="PCC: " +
                str(round(coef_list[2][0], 2)) + "<br>p: " +
                ('{:0.1e}'.format(coef_list[2][1])),
                showarrow=False,
            ),
            dict(
                x=1,
                y=1,
                xref="x2",
                yref="y3",
                font=font_ann,
                text="PCC: " +
                str(round(coef_list[2][0], 2)) + "<br>p: " +
                ('{:0.1e}'.format(coef_list[2][1])),
                showarrow=False,
            ),

        ]
    else:
        ann = []

    axisd = dict(showline=True,
                 zeroline=False,
                 gridcolor='#cecece',
                 showticklabels=True)

    # here we build a scatter matrix, and add annotations for each subgraph
    layout = go.Layout(
        dragmode='select',

        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        autosize=False,
        hovermode='closest',
        font=dict(color=colors['text2'], size=12),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        xaxis1=dict(axisd),
        xaxis2=dict(axisd),
        xaxis3=dict(axisd),
        xaxis4=dict(axisd),
        yaxis1=dict(axisd),
        yaxis2=dict(axisd),
        yaxis3=dict(axisd),
        yaxis4=dict(axisd),
        annotations=ann)

    fig = go.Figure(data=go.Splom(
        dimensions=[dict(label=selectedAttr[0],
                         values=df_selected[selectedAttr[0]]),
                    dict(label=selectedAttr[1],
                         values=df_selected[selectedAttr[1]]),
                    dict(label=selectedAttr[2],
                         values=df_selected[selectedAttr[2]]),
                    ],
        text=df_selected['boro_name'] + ', ' + df_selected['ntaname'],
        hoverinfo="x+y+text",
        # showlegend=True,
        marker=dict(color=index_vals,
                    showscale=False,  # colors encode categorical variables
                    line_color='black', line_width=0.4),
        diagonal=dict(visible=True)
    ), layout=layout
    )

    return fig


if __name__ == '__main__':

    app.run_server(debug=True)

import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State


import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, meta_tags=[
    {"content": "width=device-width, initial-scale=1.0"}
], external_stylesheets=external_stylesheets)


mapbox_access_token = 'pk.eyJ1IjoibWlzaGtpY2UiLCJhIjoiY2s5MG94bWRoMDQxdjNmcHI1aWI1YnFkYyJ9.eFsHqEMYY7qxa0Pb9USCtQ'
mapbox_style = "mapbox://styles/mishkice/ckbjhq6w50hlc1io4cnqg7svc"

# Load data
base = "https://raw.githubusercontent.com/MarinaOrzechowski/GasLeakConEd/timeline_branch/data/geolayers/"
base2 = "https://raw.githubusercontent.com/MarinaOrzechowski/GasLeakConEd/timeline_branch/data/ct_geolayers/"
df = pd.read_csv(
    'https://raw.githubusercontent.com/MarinaOrzechowski/GasLeakConEd/timeline_branch/data/processed/important_(used_in_app)/Merged_asc_fdny_data.csv')
df.rename(columns={
          'housholders_grandparents_responsible_for_grandchildren%': '%housh. grandp resp for grandch'}, inplace=True)
df = df.dropna()
df = df.drop(['occupied_housing_units%'], axis=1)

columns = df.columns
for column in columns:
    df[column] = pd.to_numeric(df[column])


centers_df = pd.read_csv(
    'https://raw.githubusercontent.com/MarinaOrzechowski/GasLeakConEd/timeline_branch/data/processed/important_(used_in_app)/geoid_with_centers.csv')
months_df = pd.read_csv(
    'https://raw.githubusercontent.com/MarinaOrzechowski/GasLeakConEd/timeline_branch/data/processed/important_(used_in_app)/Merged_asc_fdny_data_months.csv')
property_use_df = pd.read_csv(
    'https://raw.githubusercontent.com/MarinaOrzechowski/GasLeakConEd/timeline_branch/data/processed/important_(used_in_app)/FULL_fdny_2013_2018.csv')
property_use_df['ntaname'] = property_use_df['ntaname'].str.lower()
df_all_years = pd.read_csv(
    'https://raw.githubusercontent.com/MarinaOrzechowski/GasLeakConEd/timeline_branch/data/processed/important_(used_in_app)/Merged_asc_fdny_data_all_years.csv')
df_all_years = df_all_years.dropna()
df_all_years = df_all_years.drop(['occupied_housing_units%'], axis=1)
df_all_years.rename(columns={
    'housholders_grandparents_responsible_for_grandchildren%': '%housh. grandp resp for grandch'}, inplace=True)

# dictionary where keys are ntas and values are geoids in each of the ntas
nta_geoid_dict = {}
for index, row in centers_df.iterrows():
    if row['nta'] not in nta_geoid_dict:
        nta_geoid_dict[row['nta']] = [row['geoid']]
    else:
        nta_geoid_dict[row['nta']].append(row['geoid'])


# bins
BINS = [
    "0-0.0015",
    "0.00151-0.003",
    "0.0031-0.0045",
    "0.00451-0.006",
    "0.0061-0.0075",
    "0.00751-0.009",
    "0.0091-10",
    "park_cemetery"
]

# colors
DEFAULT_COLORSCALE = [
    "#eff3ff",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#084594",
    "#cbffcb"
]

DEFAULT_OPACITY = 0.8


colors = {
    'background': '#F5F5F5',
    'background2': '#d3d3d3',
    'text': '#000000',
    'text2': '#000000',
    'border': '#000000',
    'chart': ['#27496d', '#00909e', '#4d4c7d']
}

# function to assign colors to markers by boroughs


'''def find_colorscale_by_boro(df):
    color_by_boro = ['#6a2c70' if row['boro'] == 'manhattan' else '#b83b5e' if row['boro'] == 'brooklyn' else '#f08a5d' if row['boro'] ==
                     'queens' else '#f9ed69' if row['boro'] == 'staten island' else '#3ec1d3' for index, row in df.iterrows()]
    return color_by_boro'''
colorscale_by_boro = ['#e41a1c',
                      '#377eb8',
                      '#4daf4a',
                      '#984ea3',
                      '#ff7f00']


# page layout
app.layout = html.Div(
    html.Div([

        # Hidden div inside the app that stores the selected areas on the map and passes it into
        # the map callback so those areas are colored
        html.Div(id='color_selection', style={'display': 'none'}),

        # row with the header
        html.Div([
            html.H1(
                children='NYC Gas Leaks Information',
                 className='eleven columns'),
            html.Button('?', id='InfoBtn', n_clicks=0, style={}),

        ], style={
            'textAlign': 'center',
            'color': colors['text'],
            'paddingTop':'1%',
            'paddingBottom': '1%'
        }, className='row'),

        # row with a hidden info box
        html.Div([
            dcc.Textarea(
                id='InfoTxt',
                disabled=True,
                value="Description of the visualization here",
                style={'width': '100%'}
            ),
        ], className='row'),

        # row 1 with the dropdowns
        html.Div([
            # dropdown to choose neighborhoods
            html.Div([
                dcc.Dropdown(
                    id='dropdownNta',
                    options=[
                        {'label': i, 'value': i} for i in np.append('all', centers_df['nta'].unique())
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
                        } for i in df.columns[3:] if (i != 'median_houshold_income') & (i != 'gas_leaks_per_person')
                    ],
                    value=['avg_houshold_size',
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

            # range slider to choose year
            html.Div([
                dcc.Slider(
                    id="timeline",
                    min=2013,
                    max=2019,
                    step=1,
                    marks={2013: '2013', 2014: '2014', 2015: '2015',
                           2016: '2016', 2017: '2017', 2018: '2018 (Jan-Jun)', 2019: 'all'},
                    value=2018,
                    included=False
                )
            ],
                className='six columns',
                style={'float': 'left'}
            ),

            # toggle to hide outliers
            html.Div([
                daq.BooleanSwitch(
                    id='outliers_toggle',
                    on=True,
                    label='Hide outliers, set limit of gas_leaks/person to',
                    labelPosition='right',
                    color='#2a9df4'
                )
            ],
                className='three columns',
                style={'float': 'left'}
            ),

            # input field for upper limit on gas_leaks_per_person
            html.Div([
                dcc.Input(
                    id='limit_input_field',
                    type='number',
                    value=0.04,
                    min=0,
                    max=1,
                    step=0.01
                )
            ],
                className='one column',
                style={'float': 'left', 'margin': 0, 'padding': 0}
            )

        ],
            className='row'),

        # row with parallel coordinates
        html.Div([
            dcc.Graph(
                id='para_coor'
            )
        ],
            className='row'),
        # row with a map, a timeline by month, and a property use barchart
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
                                zoom=10,
                            ),
                            autosize=False,
                        ),
                    ),
                )
            ],
                className='six columns',
                style={'display': 'inline-block'}),

            # timeline of gas leaks per person
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='timeline_by_month'
                    )], className='row'),
                html.Div([
                    dcc.Graph(
                        id='property_use_barchart'
                    )], className='row')
            ],
                className='six columns',
                style={'display': 'inline-block'})
        ],
            className='row'),
        # row with the scatterplots
        html.Div([
            dcc.Graph(
                id='scatter_matrix'
            )
        ],
            className='row'),


        # row with pearson coeff heatmap
        html.Div([
            dcc.Graph(
                id='pearson_heatmap'
            )
        ],
            className='row')



    ],
        style={'backgroundColor': colors['background']})
)


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

######################################################################################################################
# InfoTxt callback
######################################################################################################################


@app.callback(
    Output('InfoTxt', 'style'),
    [
        Input('InfoBtn', 'n_clicks')
    ])
def activate_input(clicks):
    if clicks % 2 == 0:
        return {'display': 'none'}
    else:
        return {'display': 'inline', 'width': '100%'}

######################################################################################################################
# hidden_map_selection callback
######################################################################################################################


@app.callback(
    Output('color_selection', 'children'),
    [
        Input('mapGraph', 'selectedData'),
        Input('dropdownNta', 'value')
    ])
def color_selection_on_map(selectedAreaMap, selectedAreaDropdown):

    selected = []
    if selectedAreaDropdown is not None:
        if len(selectedAreaDropdown) == 0:
            pass
        elif 'all' not in selectedAreaDropdown:
            for nta in selectedAreaDropdown:
                selected.extend(nta_geoid_dict[nta])

    elif selectedAreaMap is not None:
        points = selectedAreaMap["points"]
        selected = np.unique([str(point["text"].split("<br>")[2])
                              for point in points])

    return selected

######################################################################################################################
# limit_input_field callback
######################################################################################################################


@app.callback(
    Output('limit_input_field', 'style'),
    [
        Input('outliers_toggle', 'on')
    ])
def activate_input(is_on):
    if is_on:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

######################################################################################################################
# timeline_by_month callback
######################################################################################################################


@app.callback(
    Output('timeline_by_month', 'figure'),
    [
        Input('mapGraph', 'selectedData'),
        Input('dropdownNta', 'value')
    ])
def display_selected_data(selectedAreaMap, selectedAreaDropdown):

    df_selected = months_df.merge(centers_df, on='geoid')
    df_selected = df_selected[df_selected['nta'].str[:6]
                              != 'park-c']
    key = 'geoid'

    if selectedAreaDropdown is not None:
        if len(selectedAreaDropdown) == 0:
            pass
        elif 'all' not in selectedAreaDropdown:
            df_selected = df_selected[df_selected['nta'].isin(
                selectedAreaDropdown)]
    elif selectedAreaMap is not None:
        points = selectedAreaMap["points"]
        area_names = [str(point["text"].split("<br>")[2])
                      for point in points]
        df_selected = df_selected[df_selected[key].isin(area_names)]

    df_selected = df_selected.groupby(['incident_year', 'incident_month', 'geoid']).agg(
        {'gas_leaks_per_person': 'mean'}).reset_index()

    # some of the values are inf, as we divide by population, and population is 0 in some areas (like parks)
    df_selected = df_selected[df_selected['gas_leaks_per_person'] < 1]

    fig = go.Figure()
    months = [month for month in range(1, 13)]
    months_str = ['Jan', 'Feb', 'Mar', 'Apr', "May",
                  'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # add monthly gas_leaks per person for each year
    for year in range(2013, 2018):

        df_year = df_selected[df_selected['incident_year'] == year].groupby(['incident_month']).agg(
            {'gas_leaks_per_person': 'mean'}).reset_index()

        # some areas had no gas leaks during some months, so fill it with zeros
        for i in range(12):
            if i+1 not in df_year['incident_month']:
                df_year = df_year.append(
                    {'incident_month': i+1, 'gas_leaks_per_person': 0}, ignore_index=True)

        gas_leaks = [df_year.iloc[i]['gas_leaks_per_person']
                     for i in range(12)]

        fig.add_trace(go.Scatter(x=months_str, y=gas_leaks,
                                 line=dict(width=0.5),
                                 mode='lines+markers',
                                 name=str(year)))

    # add monthly gas_leaks_per_person consolidated for all years 2013-2017 - trend. (2018 doesn't have information about all months)
    temp_df = df_selected.groupby(['incident_month']).agg(
        {'gas_leaks_per_person': 'mean'}).reset_index()

    # some areas had no gas leaks during some months, so fill it with zeros
    for i in range(12):
        if i+1 not in temp_df['incident_month']:
            temp_df = temp_df.append(
                {'incident_month': i+1, 'gas_leaks_per_person': 0}, ignore_index=True)

    gas_leaks = [temp_df.iloc[i]['gas_leaks_per_person'] for i in range(12)]

    fig.add_trace(go.Scatter(x=months_str, y=gas_leaks,
                             line=dict(color='black', width=2),
                             mode='lines+markers',
                             name='2013-2017'))
    fig.update_layout(xaxis_title='Month',
                      yaxis_title='Gas Leaks per Person',
                      plot_bgcolor=colors['background'],
                      paper_bgcolor=colors['background'],
                      title={
                          'text': "<b># Gas Leaks per Person (Monthly, for a Given Area)</b>",
                          'x': 0.5,
                          'xanchor': 'center'}
                      )
    return fig


'''
@ app.callback(
    Output("data_frame", "data"),
    [Input("timeline", "value")],
)
def choose_years(choice_years):
    df = df[(df.incident_date_time.str[6:10] >= choice_years[0]) &
            (df.incident_date_time.str[6:10] <= choice_years[1])]
    return df'''
######################################################################################################################
# property use barchart callback
######################################################################################################################


@app.callback(
    Output('property_use_barchart', 'figure'),
    [
        Input('mapGraph', 'selectedData'),
        Input('dropdownNta', 'value'),
        Input('timeline', 'value')
    ])
def display_selected_data(selectedAreaMap, selectedAreaDropdown, selectedYear):
    if selectedYear == 2018:
        title_part = '2018 (Jan-Jun)'
    elif selectedYear == 2019:
        title_part = '2013-2018'
    else:
        title_part = str(selectedYear)

    df_selected = property_use_df
    if selectedYear != 2019:
        df_selected = df_selected[df_selected['incident_date_time'].str[6:10] == str(
            selectedYear)]

    key = 'geoid'

    if selectedAreaDropdown is not None:
        if len(selectedAreaDropdown) == 0:
            pass
        elif 'all' not in selectedAreaDropdown:
            df_selected = df_selected[df_selected['ntaname'].isin(
                selectedAreaDropdown)]
    elif selectedAreaMap is not None:
        points = selectedAreaMap["points"]
        area_names = [str(point["text"].split("<br>")[2])
                      for point in points]
        df_selected = df_selected[df_selected[key].isin(area_names)]

    df_selected['count'] = 1
    df_selected = df_selected.groupby(['property_use_desc']).agg(
        {'count': 'count'}).reset_index().sort_values(by='count', ascending=False)
    total = df_selected['count'].sum()
    df_selected['percent'] = df_selected['count'] / total*100
    df_selected = df_selected.append({'property_use_desc': 'other', 'count': total-df_selected['count'][:10].sum(
    ), 'percent': 100 - df_selected['percent'][:10].sum()}, ignore_index=True)
    # sort data again so 'other' row takes correct place
    df_selected = df_selected.sort_values('percent', ascending=False)

    piechart = go.Figure(data=[go.Pie(
        labels=df_selected[:11]['property_use_desc'],
        values=df_selected[:11]['count'],
    )],
        layout=go.Layout(
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['background'],
            font={
                'color': colors['text2'],
                'size': 12
            },
            title={
                'text': "<b>Use of Properties where Gas Leaks Happened, " + title_part+"</b>",
                'x': 0.5,
                'xanchor': 'center'}))
    piechart.update_traces(hoverinfo='label+value', textinfo='text+percent', opacity=0.9,
                           marker=dict(colors=px.colors.qualitative.Prism, line=dict(color='#000000', width=1)))

    return piechart
######################################################################################################################
# map callback
######################################################################################################################


@ app.callback(
    Output("mapGraph", "figure"),
    [Input("timeline", "value"),
     Input("dropdownNta", "value"),
     Input('color_selection', 'children')],
    [State("mapGraph", "figure")],
)
def display_map(year, choiceMap, color_selected, figure):

    if year != 2019:
        df_selected = df[df.incident_year == year]
    else:
        df_selected = df_all_years

    df_selected = df_selected.merge(centers_df, on='geoid')
    df_selected['hover'] = df_selected['hover']+'<br>#Gas leaks per person: ' + \
        df_selected['gas_leaks_per_person'].round(6).astype(str)+'<br>Avg. built year: ' + \
        df_selected['avg_year_built'].round(5).astype(str)

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
    latitude = df_selected["centerLat"]
    longitude = df_selected["centerLong"]
    hover_text = df_selected["hover"]

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
        if year == 2019:
            year_str = 'all'
        else:
            year_str = str(year)
        geo_layer = dict(
            sourcetype="geojson",
            source=base + year_str+'_' + bin + ".geojson",
            type="fill",

            color=cm[bin],
            opacity=DEFAULT_OPACITY,
            # CHANGE THIS
            fill=dict(outlinecolor="#afafaf"),
        )
        layout["mapbox"]["layers"].append(geo_layer)

    for geoid in color_selected:
        geo_layer = dict(
            sourcetype="geojson",
            source=base2 + str(geoid) + ".geojson",
            type="fill",

            color='#FF0000',
            opacity=0.4,
            # CHANGE THIS
            fill=dict(outlinecolor="#afafaf"),
        )
        layout["mapbox"]["layers"].append(geo_layer)

    fig = dict(data=data, layout=layout)

    return fig

######################################################################################################################
# callbacks to reset dropdown/map selection
######################################################################################################################


@app.callback(
    Output('dropdownNta', 'value'),
    [
        Input('mapGraph', 'selectedData')
    ])
def reset_dropdown_selected(selectedAreaMap):
    if selectedAreaMap is not None:
        return None


'''@app.callback(
    Output('mapGraph', 'selectedData'),
    [
        Input('dropdownNta', 'value')
    ])
def reset_map_selected(selectedDropdown):
    if selectedDropdown is not None:
        return None
'''
######################################################################################################################
# parallel coordinates | scattermatrix | pearson coeff heatmap callbacks
######################################################################################################################


@app.callback(
    [
        Output('scatter_matrix', 'figure'),
        Output('para_coor', 'figure'),
        Output('pearson_heatmap', 'figure')
    ],
    [Input("timeline", "value"),
        Input('mapGraph', 'selectedData'),
        Input('dropdownNta', 'value'),
        Input('dropdownAttr', 'value'),
        Input('outliers_toggle', 'on'),
        Input('limit_input_field', 'value')
     ])
def display_selected_data(year, selectedAreaMap, selectedAreaDropdown, selectedAttr, hideOutliers, limit):

    if year == 2018:
        title_part = '2018 (Jan-Jun)'
    elif year == 2019:
        title_part = '2013-2018'
    else:
        title_part = str(year)

    num_of_attributes = len(selectedAttr)

    if year != 2019:
        df_selected = df[(df.incident_year == year)]
    else:
        df_selected = df_all_years

    if hideOutliers:
        df_selected = df_selected[df_selected['gas_leaks_per_person'] < limit]

    df_selected = df_selected.merge(centers_df, on='geoid')
    df_selected = df_selected[df_selected['nta'].str[:6]
                              != 'park-c']
    key = 'geoid'

    if selectedAreaDropdown is not None:
        if len(selectedAreaDropdown) == 0:
            pass
        elif 'all' not in selectedAreaDropdown:
            df_selected = df_selected[df_selected['nta'].isin(
                selectedAreaDropdown)]
    elif selectedAreaMap is not None:
        points = selectedAreaMap["points"]
        area_names = [str(point["text"].split("<br>")[2])
                      for point in points]
        df_selected = df_selected[df_selected[key].isin(area_names)]

    #########################################################################################################
    # parallel coordinates
    arr = [str(r) for r in df.columns[3:] if (
        r != 'median_houshold_income') & (r != 'occupied_housing_units%')]
    dim = [dict(range=[df_selected[attr].min(), df_selected[attr].max()],
                label=attr.replace('_', ' '), values=df_selected[attr]) for attr in arr]

    para = go.Figure(data=go.Parcoords(line=dict(color=df_selected['gas_leaks_per_person'],
                                                 colorscale=px.colors.sequential.Blues,
                                                 showscale=True
                                                 ), meta=dict(colorbar=dict(
                                                     title="gas leaks/person"
                                                 ),),
                                       dimensions=dim,
                                       labelangle=10))

    para.update_layout(height=500,
                       plot_bgcolor=colors['background'],
                       paper_bgcolor=colors['background2'],

                       )

    #########################################################################################################
    # scatterplots
    fig = make_subplots(rows=len(selectedAttr), cols=1, subplot_titles=[
        'Gas Leaks per Person VS ' + attr.replace('_', ' ').capitalize() for attr in selectedAttr
    ])

    show_legend = True
    for i in range(len(selectedAttr)):

        for ind, b in enumerate(df_selected['boro'].unique()):
            temp = df_selected[df_selected['boro'] == b]
            if i > 0:
                show_legend = False
            fig.add_trace(
                go.Scatter(x=temp['gas_leaks_per_person'],
                           y=temp[selectedAttr[i]],
                           mode='markers',
                           marker_color=f"rgba{(*hex_to_rgb(colorscale_by_boro[ind]), 0.6)}",
                           showlegend=show_legend,
                           name=b,
                           text=temp['hover']),

                row=i+1, col=1
            )

    fig.update_traces(mode='markers', marker_line_width=0.2, marker_size=3.5)
    fig.update_layout(font=dict(color=colors['text2'], size=12),
                      plot_bgcolor=colors['background'],
                      paper_bgcolor=colors['background'],
                      height=900,

                      title={
        'text': "<b>Comparison of Gas Leak#/person to Other Attributes, " + title_part + '</b>',
        'x': 0.5,
        'xanchor': 'center'})

    #######################################################################################################
    # pearson coeff heatmap
    l = {}
    df_heatmap = df_selected.drop(
        ['Unnamed: 0_x', 'geoid', 'Unnamed: 0_y', 'centerLong', 'centerLat'], axis=1)
    if year != 2019:
        df_heatmap = df_heatmap.drop(
            ['incident_year'], axis=1)

    pearsoncorr = df_heatmap.corr(method='pearson')
    heatmap = go.Figure(data=go.Heatmap(
        z=[pearsoncorr[i] for i in pearsoncorr.columns],
        x=[row.replace('_', ' ').capitalize() for row in pearsoncorr.columns],
        y=[row.replace('_', ' ').capitalize() for row in pearsoncorr.columns],
        colorscale='PRGn'
    ))
    heatmap.update_layout(
        title={
            'text': '<b>Pearson correlation coefficient, '+title_part+'</b>',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        height=700,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        autosize=False)
    return fig, para, heatmap


if __name__ == '__main__':
   # app.config['suppress_callback_exceptions'] = True
    app.run_server(debug=True)

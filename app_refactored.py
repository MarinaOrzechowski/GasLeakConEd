import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import re


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

columns_original = df.columns

centers_df = pd.read_csv(
    'https://raw.githubusercontent.com/MarinaOrzechowski/GasLeakConEd/timeline_branch/data/processed/important_(used_in_app)/geoid_with_centers.csv')
df = df.merge(centers_df, on='geoid')
df['hover'] = df['hover']+'<br>#Gas leaks per person: ' + \
    df['gas_leaks_per_person'].round(6).astype(str)+'<br>Avg. built year: ' + \
    df['avg_year_built'].round(5).astype(str)
months_df = pd.read_csv(
    'https://raw.githubusercontent.com/MarinaOrzechowski/GasLeakConEd/timeline_branch/data/processed/important_(used_in_app)/Merged_asc_fdny_data_months.csv')

months_centers_df = months_df.merge(centers_df, on='geoid')
property_use_df = pd.read_csv(
    'https://raw.githubusercontent.com/MarinaOrzechowski/GasLeakConEd/timeline_branch/data/processed/important_(used_in_app)/FULL_fdny_2013_2018.csv')
property_use_df['ntaname'] = property_use_df['ntaname'].str.lower()
df_all_years = pd.read_csv(
    'https://raw.githubusercontent.com/MarinaOrzechowski/GasLeakConEd/timeline_branch/data/processed/important_(used_in_app)/Merged_asc_fdny_data_all_years.csv')
df_all_years = df_all_years.dropna()
df_all_years = df_all_years.drop(['occupied_housing_units%'], axis=1)
df_all_years.rename(columns={
    'housholders_grandparents_responsible_for_grandchildren%': '%housh. grandp resp for grandch'}, inplace=True)
df_all_years = df_all_years.merge(centers_df, on='geoid')
df_all_years['hover'] = df_all_years['hover']+'<br>#Gas leaks per person: ' + \
    df_all_years['gas_leaks_per_person'].round(6).astype(str)+'<br>Avg. built year: ' + \
    df_all_years['avg_year_built'].round(5).astype(str)

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

colorscale_by_boro = ['#e41a1c',
                      '#377eb8',
                      '#4daf4a',
                      '#984ea3',
                      '#ff7f00']


# page layout
app.layout = html.Div(
    html.Div([

        # Hidden divs which store filtered data
        html.Div(id='filtered_year_data', style={'display': 'none'}),
        html.Div(id='outliers_filtered_data', style={'display': 'none'}),
        html.Div(id='location_filtered_data', style={'display': 'none'}),
        html.Div(id='color_selection_map', style={'display': 'none'}),
        html.Div(id='par_coord_range', style={'display': 'none'}),

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
                    id="dropdown_attr",
                    options=[
                        {
                            'label': i, 'value': i
                        } for i in columns_original[3:] if (i != 'median_houshold_income') & (i != 'gas_leaks_per_person')
                    ],
                    value=['avg_houshold_size'],
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
                    id='limit_outliers_field',
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


        # row with a map, a timeline by month, and a Pearson correlation heatmap
        html.Div([
            # map
            html.Div([
                dcc.Graph(
                    id='map_graph',
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
                        id='pearson_heatmap'
                    )], className='row'),
                html.Div([
                    dcc.Graph(
                        id='timeline_by_month'
                    )], className='row')

            ],
                className='six columns',
                style={'display': 'inline-block'})
        ],
            className='row'),


        # row with parallel coord
        html.Div([
            dcc.Graph(
                id='para_coor'
            )
        ],
            className='row'),
        # row with the scatterplots
        html.Div([
            dcc.Graph(
                id='scatter_matrix'
            )
        ],
            className='row'),

        # row with pie chart (property use)
        html.Div([
            dcc.Graph(
                id='property_use_barchart'
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


########################################
# InfoTxt callback

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

########################################
# filter data by selected year callback


@app.callback(
    Output('filtered_year_data', 'children'),
    [
        Input('timeline', 'value')
    ])
def filter_by_year(year):
    return df[df.incident_year == year].to_json() if year != 2019 else df_all_years.to_json()

########################################
# visibility of outliers limit  callback


@app.callback(
    Output('limit_outliers_field', 'style'),
    [
        Input('outliers_toggle', 'on')
    ])
def activate_input(is_on):
    if is_on:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

########################################
# filter data by outlier limit callback


@app.callback(
    Output('outliers_filtered_data', 'children'),
    [
        Input('outliers_toggle', 'on'),
        Input('limit_outliers_field', 'value'),
        Input('filtered_year_data', 'children')
    ])
def filter_by_outlier(is_on, limit, data_json):
    if not is_on:
        return data_json
    else:
        data = pd.read_json(data_json)
        return data[data['gas_leaks_per_person'] < limit].to_json()


########################################
# filter data by selected locations callback

@app.callback(
    Output('location_filtered_data', 'children'),
    [
        Input('map_graph', 'selectedData'),
        Input('dropdownNta', 'value'),
        Input('outliers_filtered_data', 'children')
    ])
def filter_by_location(selected_area_map, selected_area_dropdown, filtered_data_json):
    filtered_data = pd.read_json(filtered_data_json)
    df_selected = filtered_data[filtered_data['nta'].str[:6] != 'park-c']

    if selected_area_dropdown is not None:
        if len(selected_area_dropdown) == 0:
            pass
        elif 'all' not in selected_area_dropdown:
            df_selected = df_selected[df_selected['nta'].isin(
                selected_area_dropdown)]
    elif selected_area_map is not None:
        points = selected_area_map["points"]
        area_names = [str(point["text"].split("<br>")[2])
                      for point in points]
        df_selected = df_selected[df_selected['geoid'].isin(area_names)]

    return df_selected.to_json()


####################################################################
# filter data by selected locations on par. coord. and scatter plots

@app.callback(
    Output('color_selection_map', 'children'),
    [
        Input('map_graph', 'selectedData'),
        Input('dropdownNta', 'value'),
        Input('scatter_matrix', 'selectedData'),
        Input('par_coord_range', 'children')
    ])
def color_selection_on_map(selectedAreaMap, selectedAreaDropdown, scatter_selection, constraintrange_parcoord):

    selected = []

    if selectedAreaDropdown:
        if len(selectedAreaDropdown) == 0:
            pass
        elif 'all' not in selectedAreaDropdown:
            for nta in selectedAreaDropdown:
                selected.extend(nta_geoid_dict[nta])

    elif selectedAreaMap:
        points = selectedAreaMap["points"]
        selected = np.unique([str(point["text"].split("<br>")[2])
                              for point in points])

    if constraintrange_parcoord:
        if len(constraintrange_parcoord) == 0:
            pass
        else:
            selected = constraintrange_parcoord
    elif scatter_selection:
        points = scatter_selection["points"]
        selected = np.unique([str(point["text"].split("<br>")[2])
                              for point in points])

    return selected

##########################################################
# if something on map is selected, the dropdown gets reset


@app.callback(
    Output('dropdownNta', 'value'),
    [
        Input('map_graph', 'selectedData'),
        Input('scatter_matrix', 'selectedData'),
        Input('par_coord_range', 'children')
    ])
def reset_dropdown_selected(selectedAreaMap, scatter, parcoord):
    if selectedAreaMap or scatter or parcoord:
        return None

######################################################################################################################
# get selected range from parallel coordinates
######################################################################################################################


@app.callback(
    Output('par_coord_range', 'children'),
    [Input('para_coor', 'restyleData'),
     Input('para_coor', 'figure'),
     Input('location_filtered_data', 'children')]
)
def get_selected_parcoord(restyleData, figure, data_selected):
    # if selected_scatter is not None:
    #     return None
    ranges = []
    all_geoids = []

    dff = pd.read_json(data_selected)
    if restyleData and 'constraintrange' in figure['data'][0]['dimensions'][0].keys():
        label = figure['data'][0]['dimensions'][0]['label']
        # list of lists
        ranges = figure['data'][0]['dimensions'][0]['constraintrange']

        all_geoids = []
        # select geoids with gas_leaks in the selected intervals
        if isinstance(ranges[0], list):
            for range in ranges:
                selected_dff = dff[dff['gas_leaks_per_person'].between(
                    range[0], range[1], inclusive=True)]
                geoids = selected_dff['geoid']
                all_geoids.extend(geoids)
        else:
            selected_dff = dff[dff['gas_leaks_per_person'].between(
                ranges[0], ranges[1], inclusive=True)]
            geoids = selected_dff['geoid']
            all_geoids.extend(geoids)

    return all_geoids

#################################################
# update map depending on year and selected areas


@app.callback(

    Output("map_graph", "figure"),
    [
        Input("timeline", "value"),
        Input("filtered_year_data", "children"),
        Input('color_selection_map', 'children')],
    [
        State("map_graph", "figure")
    ]
)
def display_map(year, data_json, data_to_color, figure):

    data_ = pd.read_json(data_json)
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
    latitude = data_["centerLat"]
    longitude = data_["centerLong"]
    hover_text = data_["hover"]

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

    for geoid in data_to_color:
        geo_layer = dict(
            sourcetype="geojson",
            source=base2 + str(geoid) + ".geojson",
            type="fill",

            color='#F74DFF',
            opacity=0.4,
            # CHANGE THIS
            fill=dict(outlinecolor="#afafaf"),
        )
        layout["mapbox"]["layers"].append(geo_layer)

    fig = dict(data=data, layout=layout)

    return fig

##########################################
# parallel coordinates


@app.callback(

    Output('para_coor', 'figure'),
    [
        Input("location_filtered_data", "children"),
        Input('scatter_matrix', 'selectedData')
    ])
def build_parallel_coord(filtered_data_json, selected_scatter):

    filtered_data = pd.read_json(filtered_data_json)
    if selected_scatter and len(selected_scatter) > 0:
        points = selected_scatter["points"]
        area_names = [str(point["text"].split("<br>")[2])
                      for point in points]
        filtered_data = filtered_data[filtered_data['geoid'].isin(area_names)]

    arr = [str(r) for r in columns_original[3:] if (
        r != 'median_houshold_income') & (r != 'occupied_housing_units%')]

    dim = [dict(range=[filtered_data[attr].min(), filtered_data[attr].max()],
                label=attr.replace('_', ' '), values=filtered_data[attr]) for attr in arr]

    para = go.Figure(data=go.Parcoords(line=dict(color=filtered_data['gas_leaks_per_person'],
                                                 colorscale=px.colors.sequential.tempo,
                                                 showscale=True
                                                 ), meta=dict(colorbar=dict(
                                                     title="gas leaks/person"
                                                 ),),
                                       dimensions=dim,
                                       labelangle=10))

    para.update_layout(
        height=500,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background2'],

    )
    return para

#########################################################################################################
# scatterplots


@app.callback(
    Output('scatter_matrix', 'figure'),
    [
        Input("location_filtered_data", "children"),
        Input('par_coord_range', 'children'),
        Input('dropdown_attr', 'value')
    ])
def build_parallel_coord(filtered_data_json, selected_par, selected_attr):

    fig = make_subplots(rows=len(selected_attr), cols=1, subplot_titles=[
        'Gas Leaks per Person VS ' + attr.replace('_', ' ').capitalize() for attr in selected_attr])

    filtered_data = pd.read_json(filtered_data_json)
    if selected_par and len(selected_par) > 0:
        filtered_data = filtered_data[filtered_data['geoid'].isin(
            selected_par)]

    show_legend = True
    for i in range(len(selected_attr)):

        for ind, b in enumerate(filtered_data['boro'].unique()):
            temp = filtered_data[filtered_data['boro'] == b]
            if i > 0:
                show_legend = False
            fig.add_trace(
                go.Scatter(x=temp['gas_leaks_per_person'],
                           y=temp[selected_attr[i]],
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
                      dragmode='select',
                      title={
        # + title_part + '</b>',
        'text': "<b>Comparison of Gas Leak#/person to Other Attributes, ",
        'x': 0.5,
        'xanchor': 'center'})

    return fig

#######################################################################################################
# pearson coeff heatmap


@app.callback(
    Output('pearson_heatmap', 'figure'),
    [
        Input('map_graph', 'selectedData'),
        Input('dropdownNta', 'value'),
        Input('par_coord_range', 'children'),
        Input('scatter_matrix', 'selectedData'),
        Input('outliers_toggle', 'on'),
        Input('limit_outliers_field', 'value')
    ])
def display_selected_data(selectedAreaMap, selectedAreaDropdown, constraintrange_parcoord, scatter_selection, hideOutliers, limit):

    df_selected = df
    df_selected_all = df_all_years

    if hideOutliers:
        df_selected = df_selected[df_selected['gas_leaks_per_person'] < limit]
        df_selected_all = df_selected_all[df_selected_all['gas_leaks_per_person'] < limit]

    df_selected = df_selected[df_selected['nta'].str[:6]
                              != 'park-c']

    df_selected_all = df_selected_all[df_selected_all['nta'].str[:6]
                                      != 'park-c']
    key = 'geoid'

    if selectedAreaDropdown is not None:
        if len(selectedAreaDropdown) == 0:
            pass
        elif 'all' not in selectedAreaDropdown:
            df_selected = df_selected[df_selected['nta'].isin(
                selectedAreaDropdown)]
            df_selected_all = df_selected_all[df_selected_all['nta'].isin(
                selectedAreaDropdown)]
    elif selectedAreaMap is not None:
        points = selectedAreaMap["points"]
        area_names = [str(point["text"].split("<br>")[2])
                      for point in points]
        df_selected = df_selected[df_selected[key].isin(area_names)]
        df_selected_all = df_selected_all[df_selected_all[key].isin(
            area_names)]

    if constraintrange_parcoord:
        if len(constraintrange_parcoord) == 0:
            pass
        else:
            df_selected = df_selected[df_selected[key].isin(
                constraintrange_parcoord)]
            df_selected_all = df_selected_all[df_selected_all[key].isin(
                constraintrange_parcoord)]
    elif scatter_selection:
        points = scatter_selection["points"]
        area_names = np.unique([str(point["text"].split("<br>")[2])
                                for point in points])
        df_selected = df_selected[df_selected[key].isin(area_names)]
        df_selected_all = df_selected_all[df_selected_all[key].isin(
            area_names)]

    df_pearson = df_selected.drop(
        ['Unnamed: 0_x', 'geoid', 'Unnamed: 0_y', 'centerLong', 'centerLat'], axis=1)
    df_pearson_all = df_selected_all.drop(
        ['Unnamed: 0_x', 'geoid', 'Unnamed: 0_y', 'centerLong', 'centerLat'], axis=1)

    pearsoncorr_all = df_pearson_all.corr(method='pearson')
    pearson_gas_leaks_all = pearsoncorr_all['gas_leaks_per_person']

    attributes = [col.replace('_', ' ').capitalize()
                  for col in pearsoncorr_all.columns]

    matrix = [[] for _ in range(len(attributes)-1)]
    years = [year for year in range(2013, 2019)]

    for year in years:
        df_pearson_year = df_pearson[df_pearson['incident_year'] == year]
        df_pearson_year = df_pearson_year.drop(columns={'incident_year'})
        pearsoncorr = df_pearson_year.corr(method='pearson')
        pearson_gas_leaks = pearsoncorr['gas_leaks_per_person']
        for i in range(len(attributes)-1):
            matrix[i].append(pearson_gas_leaks[i+1])

    for i in range(len(attributes)-1):
        matrix[i].append(pearson_gas_leaks_all[i+1])

    years.append('all')

    heatmap = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=years,
            y=attributes[1:],
            colorscale='RdBu',
            colorbar=dict(title='Pearson coef.'),
            xgap=20,
            zmax=0.8,
            zmin=-0.8,
            zmid=0))

    heatmap.update_layout(
        xaxis={'type': 'category'},
        title={
            'text': '<b>Pearson correlation coefficient by year</b>',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        height=500,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        autosize=True)
    return heatmap

######################################################################################################################
# timeline_by_month callback
######################################################################################################################


@app.callback(
    Output('timeline_by_month', 'figure'),
    [
        Input('map_graph', 'selectedData'),
        Input('dropdownNta', 'value'),
        Input('par_coord_range', 'children'),
        Input('scatter_matrix', 'selectedData')
    ])
def display_selected_data(selectedAreaMap, selectedAreaDropdown, constraintrange_parcoord, scatter_selection):

    df_selected = months_centers_df[months_centers_df['nta'].str[:6]
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

    if constraintrange_parcoord:
        if len(constraintrange_parcoord) == 0:
            pass
        else:
            df_selected = df_selected[df_selected[key].isin(
                constraintrange_parcoord)]

    elif scatter_selection:
        points = scatter_selection["points"]
        area_names = np.unique([str(point["text"].split("<br>")[2])
                                for point in points])
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

######################################################################################################################
# property use barchart callback
######################################################################################################################


@app.callback(
    Output('property_use_barchart', 'figure'),
    [
        Input('map_graph', 'selectedData'),
        Input('dropdownNta', 'value'),
        Input('timeline', 'value'),
        Input('par_coord_range', 'children'),
        Input('scatter_matrix', 'selectedData')
    ])
def display_selected_data(selectedAreaMap, selectedAreaDropdown, selectedYear, constraintrange_parcoord, scatter_selection):
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
            df_selected = df_selected[df_selected['nta'].isin(
                selectedAreaDropdown)]

    elif selectedAreaMap is not None:
        points = selectedAreaMap["points"]
        area_names = [str(point["text"].split("<br>")[2])
                      for point in points]
        df_selected = df_selected[df_selected[key].isin(area_names)]

    if constraintrange_parcoord:
        if len(constraintrange_parcoord) == 0:
            pass
        else:
            df_selected = df_selected[df_selected[key].isin(
                constraintrange_parcoord)]

    elif scatter_selection:
        points = scatter_selection["points"]
        area_names = np.unique([str(point["text"].split("<br>")[2])
                                for point in points])
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


if __name__ == '__main__':
   # app.config['suppress_callback_exceptions'] = True
    app.run_server(debug=True)

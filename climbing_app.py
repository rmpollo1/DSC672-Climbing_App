import dash
import dash_core_components as dcc 
import dash_html_components as html 
from dash.dependencies import Input, Output
import plotly.express as px 
import pandas as pd 
#import sqlite3 as sql 
import os 
# Local Python Files
import climbing_queries as sql
import recommenders as rec 
import ui_components as ui 


# Load Up Recommenders
mf_recommender = rec.SurpriseRecommender()
mf_recommender.load_recommender('svd')

knn_rec = rec.SurpriseRecommender()
knn_rec.load_recommender('knn')

autorec = rec.AutoRecWrapper()
autorec.load_model(rec.AUTOREC)

caser = rec.CaserWrapper()
caser.load_from_path(rec.CASER7)

recommenders = {
    'svd':mf_recommender, 
    'knn':knn_rec, 
    'autorec':autorec,
    'caser':caser
}

TMP_UIDS = mf_recommender.sample()

app = dash.Dash(__name__,suppress_callback_exceptions=True)

app.layout = html.Div(
    children=[
        dcc.Location('url',refresh=False),
        
    ],
    id = 'main-wrap'
)


@app.callback(
    Output('main-wrap','children'),
    [Input('url','pathname')]
)
def change_page(pathname):
    if pathname == '/':
        main_content = html.Div(
            [
                html.A(html.H1('Route Recommender'),href='./user'),
                html.A(html.H1('Trip Planner'),href='./trip')
            ]
        )
    if pathname:
        top_dir = pathname[1:]

    if top_dir == 'user':
        main_content = html.Div([html.Div(
            [
                dcc.Dropdown(
                    'user_input',
                    options=[{'label':uid,'value':uid} for i,uid in enumerate(TMP_UIDS)],value=TMP_UIDS[0],
                    clearable=False
                ),
                dcc.Dropdown(
                    'rec_type',
                    options=[
                        {'label':'KNN-Rec','value':'knn'},
                        {'label':'Matrix Factorization','value':'svd'},
                        {'label':'AutoRec','value':'autorec'},
                        {'label':'Caser','value':'caser'}
                    ],
                    value='knn',
                    clearable=False
                ),
                dcc.Input(
                    'email',
                    placeholder='Enter Moutain Project Email ...',
                    type='text',
                    value='',
                    style={'width':'100%'}
                ) 
            ], 
                id='left-box'
            ),
        html.H2('Past Routes',id='past_title'),
        html.Div(id='past-routes'),
        html.H2('Recommended Routes',id='rec_title'),
        html.Div(id='rec-routes')],id='main')

    if top_dir == 'trip':
        main_content = html.Div(html.H1('Under Construction'))

    if not pathname:
        main_content = html.H1('Page Not Found')
    return main_content


@app.callback(
    Output('past-routes','children'),
    [
        Input('user_input','value'),
        Input('email','value')
    ]
)
def show_climbs(uid,email=None):
    if email:
        uid = sql.get_uid(email)

    recent = sql.get_most_recent(uid)
    routes = sql.get_route_info(list(recent['route_id']))
    return ui.route_lists(routes)

@app.callback(
    Output('rec-routes','children'),
    [
        Input('user_input','value'), 
        Input('rec_type','value'),
        Input('email','value')
    ],
)
def rec_climbs(uid,rec_type,email=None):
    if email:
        uid = sql.get_uid(email)
    recommender = recommenders[rec_type]
    if rec_type == 'caser':
        recent = sql.get_most_recent(uid,str(recommender.L))
        past_iids = recent['route_id'].to_numpy()[::-1]
        rec_iids, _ = recommender.top_k_mask(uid,past_iids)
        rec_info = sql.get_route_info(rec_iids.tolist())
    else:
        rec_iids, _ = recommender.top_k(uid)
        rec_info = sql.get_route_info(rec_iids.tolist())
    return ui.route_lists(rec_info)

if __name__ == '__main__':
    
    app.run_server(debug=True)
import dash_html_components as html 
import pandas as pd
 
def route_tile(row):
    if not row['img_link']:
        row['img_link'] = 'https://www.rosspollock.design/img/no_climb_img.png'

    if not pd.isnull(row['pitches']):
        pitches = str(int(row['pitches']))
    else:
        pitches = "Not Available"
    tile = html.Div(children=[
        html.A(html.H3(row['name']),href=row['link'],className='name'),
        html.P(f'Difficulty {row["rating"]}',className='diff'),
        html.Div(
            html.Img(src=row['img_link']),className='photo'
        ),
        html.Div(
           [ 
            html.P('Route Type: ' + row['type']),
            html.P('Pitches: ' + pitches),
            html.P(' > '.join([row['state'],row['city']]))
           ],
        className='loc')
    ],className='route-tile')
    return tile

def route_lists(df):
    list_tiles = html.Div(
        children=[route_tile(row) for _, row in df.iterrows()],
        className='route-list'
    )
    return list_tiles
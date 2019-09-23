"""
Dash(Pythonの可視化ライブラリ)使ったサンプルコード
棒グラフ表示+Dataframe表示　etc アプリ
app_bar_plot.pyにhtmlの要素のところに、styleを付け加えて背景などを変える
https://qiita.com/OgawaHideyuki/items/6df65fbbc688f52eb82c
https://qiita.com/shimopino/items/ddc46adcbd6332511b92#%E8%A1%A8%E3%82%92%E4%BD%9C%E6%88%90%E3%81%99%E3%82%8B
https://qiita.com/shimopino/items/8f524916eeac8c445cf0#_reference-892c93e32a3984eb45a5
https://dash.plot.ly/datatable/callbacks
Usage:
    $ activate tfgpu_py36_v3
    $ python app_bar_plot_add_stylesheets_tab_etc.py
    → http://127.0.0.1:8050/
"""
import dash
import dash_core_components as dcc # dash_core_componentsにはGraphと呼ばれるコンポーネントが存在し、Ploylyの35種類以上の図に対応 https://plot.ly/python/
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_table

# Dataframeでロードするcsv
#df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/c78bf172206ce24f77d6363a2d754b59/raw/c353e8ef842413cae56ae3920b8fd78468aa4cb2/usa-agricultural-exports-2011.csv')
df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')

df1 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

df2 = pd.read_csv('https://gist.githubusercontent.com/chriddyp/cb5392c35661370d95f300086accea51/raw/8e0768211f6b747c0db42a9ce9a0937dafcbd8b2/indicators.csv')
available_indicators = df2['Indicator Name'].unique()

df_page = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')
#PAGE_SIZE = 15
operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]

def generate_table(dataframe, max_row=10):
    """ dataframeをHTMLの表へ変換する """
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_row))]
    )

def split_filter_part(filter_part):
    """
    DataFrameにfilterつける
    https://dash.plot.ly/datatable/callbacks
    """
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value
    return [None] * 3

# Markdown
markdown_text = '''
### Dash and Markdown

Dash apps can be written in Markdown.
Dash uses the [CommonMark](http://commonmark.org/)
specification of Markdown.
Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
if this is your first introduction to Markdown!
'''

# 付け加え　外部スタイルシート
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# 付け加え　色
colors = {
    'background': 'limegreen',
    'text': '#7FDBFF'
}

# 付け加え　外部スタイルシート
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Tabs(id="tabs", children=[
        ############################## 棒グラフ ##############################
        dcc.Tab(label='Hello Dash', children=[
            html.H1('Hello Dash',
            # スタイルを設定(辞書型で渡す)
            style={
                'textAlign': 'center', # テキストセンター寄せ
                'color': colors['text'], # 文字色
                }
            ),
            # 棒グラフ
            html.Div([
                dcc.Graph(
                    id = "first-graph",
                    figure = {
                        'data': [
                            {'x': [1,2,3,4],
                            'y':[3,2,4,6],
                            'type': 'bar',
                            'name': '東京'},
                            {'x':[1,2,3,4],
                            'y':[2,4,3,2],
                            'type': 'bar',
                            'name': '大阪'},
                            {'x': [1,2,3,4], # データ２つ足す
                            'y':[2,1,4,6],
                            'type': 'bar',
                            'name': '京都'},
                            {'x': [1,2,3,4],
                            'y':[1,3,4,7],
                            'type': 'bar',
                            'name': '福岡'},
                        ],
                        'layout': {
                            'title': '棒グラフ',
                            'paper_bgcolor': colors['background'], # グラフの外の背景色
                            'plot_bgcolor': colors['background'] # グラフの中の背景色
                            }
                    }
                )
            ])
        ]),
        ############################## 散布図 ##############################
        dcc.Tab(label='Dataframe', children=[
            html.H4(children='life-exp-vs-gdp'),#children='US Agriculture Exports (2011)'),
            # 散布図
            html.Div([
                dcc.Graph(
                    id='life-exp-vs-gdp',
                    figure={
                        'data': [
                            go.Scatter(
                                # 1人当たりGDP
                                x=df[df['continent'] == i]['gdp per capita'],
                                # 寿命
                                y=df[df['continent'] == i]['life expectancy'],
                                text=df[df['continent'] == i]['country'],
                                mode='markers',
                                opacity=0.7,
                                marker={
                                    'size': 15,
                                    'line': {'width': 0.5, 'color': 'white'}
                                },
                                name=i
                            # 大陸ごとのデータを抽出する。
                            ) for i in df.continent.unique()
                        ],
                        'layout': go.Layout(
                            # x軸はログスケール
                            xaxis={'type': 'log', 'title': 'GDP Per Capita'},
                            yaxis={'title': 'Life Expectancy'},
                            # left, bottom, top, right
                            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                            legend={'x': 0, 'y': 1},
                            hovermode='closest'
                        )
                    }
                )
            ]),
            # Dataframe
            html.Div(children=[
                generate_table(df)
            ])
        ]),
        ############################## Markdown を使用する ##############################
        dcc.Tab(label='Markdown', children=[
            html.Div([
                dcc.Markdown(children=markdown_text)
            ])
        ]),
        ############################## 他のdash_core_componentsを試す ##############################
        dcc.Tab(label='Dash core components', children=[
            html.Div([
                html.Label('Dropdown'),
                dcc.Dropdown(
                    options=[
                        {'label': 'New York City', 'value': 'NYC'},
                        {'label': u'Montréal', 'value': 'MTL'},
                        {'label': 'San Francisco', 'value': 'SF'}
                    ],
                    # ここに選択した値が入る（ここではデフォルト値を設定している）
                    value='MTL'
                ),

                html.Label('Multi-Select Dropdown'),
                dcc.Dropdown(
                    options=[
                        {'label': 'New York City', 'value': 'NYC'},
                        {'label': u'Montréal', 'value': 'MTL'},
                        {'label': 'San Francisco', 'value': 'SF'}
                    ],
                    value=['MTL', 'SF'],
                    # 複数の選択を可能にする引数
                    multi=True
                ),

                html.Label('Radio Items'),
                dcc.RadioItems(
                    options=[
                        {'label': 'New York City', 'value': 'NYC'},
                        {'label': u'Montréal', 'value': 'MTL'},
                        {'label': 'San Francisco', 'value': 'SF'}
                    ],
                    value='MTL'
                ),

                html.Label('Checkboxes'),
                dcc.Checklist(
                    options=[
                        {'label': 'New York City', 'value': 'NYC'},
                        {'label': u'Montréal', 'value': 'MTL'},
                        {'label': 'San Francisco', 'value': 'SF'}
                    ],
                    # リスト型で与えていることに注意
                    value=['MTL', 'SF']
                ),

                html.Label('Text Input'),
                dcc.Input(value='MTL', type='text'),

                html.Label('Slider'),
                dcc.Slider(
                    min=0,
                    max=9,
                    marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(1, 6)},
                    value=5,
                ),
            # 表示を2列に分けるstyleを設定
            ], style={'columnCount': 2})
        ]),
        ############################## Callback で対話形式に変更内容表示する　##############################
        dcc.Tab(label='Callback', children=[
            html.Div([
                # Inputボックスの中に入力した値が、ボックスの直下にあるDivタグ内の値に反映される
                dcc.Input(
                    id='my-id',
                    value='initial value',
                    type='text'
                ),
                html.Div(id='my-div')
            ])
        ]),
        ############################## Sliderからグラフを操作　##############################
        dcc.Tab(label='Graph-with-slider', children=[
            html.Div([
                # 最初グラフには何も設定しない。（上書きされるため）
                dcc.Graph(id='graph-with-slider'),
                dcc.Slider(
                    # 入力コンポーネントとしてIDを設定
                    id='year-slider',
                    # スライダーの最小値・最大値を設定
                    min=df1['year'].min(),
                    max=df1['year'].max(),
                    # 今回の入力値(初期値)
                    value=df1['year'].min(),
                    step=None,
                    # スライダーのラベルを設定
                    marks={str(year): str(year) for year in df1['year'].unique()}
                )
            ]),
            # Dataframe
            html.Div(children=[
                generate_table(df1)
            ])
        ]),
        ############################## Dropdownから2つ、RadioItemsから2つ、Sliderから1つの合計5つの入力値を受けてグラフを操作 ##############################
        dcc.Tab(label='Graph-with-any-Input', children=[
            html.Div([

                html.Div([
                    dcc.Dropdown(
                        id='xaxis-column',
                        # available_indicator変数を選択肢として設定
                        options=[{'label': i, 'value': i} for i in available_indicators],
                        # 初期値
                        value='Fertility rate, total (births per woman)'
                    ),
                    dcc.RadioItems(
                        id='xaxis-type',
                        options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                        # 初期値、ここでは1つだけ選択する
                        value='Linear',
                        labelStyle={'display': 'inline-block'}
                    )
                ],
                # 上の2つの要素をひとまとめにstyleを設定する
                style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Dropdown(
                        id='yaxis-column',
                        options=[{'label': i, 'value': i} for i in available_indicators],
                        # 初期値
                        value='Life expectancy at birth, total (years)'
                    ),
                    dcc.RadioItems(
                        id='yaxis-type',
                        options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                        # 初期値
                        value='Linear',
                        labelStyle={'display': 'inline-block'}
                    )
                # 上の2つの要素をひとまとめにstyleを設定する
                ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ]),

            # ここに出力する図を挿入する
            dcc.Graph(id='indicator-graphic'),

            dcc.Slider(
                id='year--slider',
                min=df2['Year'].min(),
                max=df2['Year'].max(),
                # 初期値
                value=df2['Year'].max(),
                step=None,
                marks={str(year): str(year) for year in df2['Year'].unique()}
            ),

            # Dataframe
            html.Div(children=[
                generate_table(df2)
            ])
        ]),
        ############################## DataFrameのページングやfilterやグラフ ##############################
        dcc.Tab(label='DataFrame paging-with-graph', children=[
            html.Div(
                className="row",
                children=[
                    html.Div(
                        dash_table.DataTable(
                            id='table-paging-with-graph',
                            columns=[
                                {"name": i, "id": i} for i in sorted(df_page.columns)
                            ],
                            page_current=0,
                            page_size=df_page.shape[0],#20,
                            page_action='custom',

                            filter_action='custom',
                            filter_query='',

                            sort_action='custom',
                            sort_mode='multi',
                            sort_by=[]
                        ),
                        style={'height': 750, 'overflowY': 'scroll'},
                        className='six columns'
                    ),
                    html.Div(
                        id='table-paging-with-graph-container',
                        className="five columns"
                    )
                ]
            )
        ])
    ])
])

@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    """ input_valueの内容返す """
    return 'You have entered "{}"'.format(input_value)

@app.callback(
    dash.dependencies.Output('graph-with-slider', 'figure'),
    [dash.dependencies.Input('year-slider', 'value')])
def update_figure(selected_year):
    """ 入力値と出力値を設定し、年ごとのデータを出力 """
    # 入力値には`value`、つまり年のデータが送られる。
    # 年ごとのデータを抽出する。
    filtered_df = df1[df1.year == selected_year]
    # figureを格納する空リストを作成
    traces = []
    # 大陸別にグループ分けを行う
    for i in filtered_df.continent.unique():
        df_by_continent = filtered_df[filtered_df['continent'] == i]
        traces.append(go.Scatter(
            x=df_by_continent['gdpPercap'],
            y=df_by_continent['lifeExp'],
            text=df_by_continent['country'],
            mode='markers',
            opacity=0.7,
            marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=i
        ))

    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }

@app.callback(
    dash.dependencies.Output('indicator-graphic', 'figure'),
    [dash.dependencies.Input('xaxis-column', 'value'),
     dash.dependencies.Input('yaxis-column', 'value'),
     dash.dependencies.Input('xaxis-type', 'value'),
     dash.dependencies.Input('yaxis-type', 'value'),
     dash.dependencies.Input('year--slider', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_value):
    """ 複数の入力値と出力値を設定し、図に反映 """
    dff = df2[df2['Year'] == year_value]

    return {
        'data': [go.Scatter(
            x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
            y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
            text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            }
        )],
        'layout': go.Layout(
            xaxis={
                'title': xaxis_column_name,
                'type': 'linear' if xaxis_type == 'Linear' else 'log'
            },
            yaxis={
                'title': yaxis_column_name,
                'type': 'linear' if yaxis_type == 'Linear' else 'log'
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }

@app.callback(
    Output('table-paging-with-graph', "data"),
    [Input('table-paging-with-graph', "page_current"),
     Input('table-paging-with-graph', "page_size"),
     Input('table-paging-with-graph', "sort_by"),
     Input('table-paging-with-graph', "filter_query")])
def update_table(page_current, page_size, sort_by, filter):
    """
    https://dash.plot.ly/datatable/callbacks
    """
    filtering_expressions = filter.split(' && ')
    dff = df_page
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    if len(sort_by):
        dff = dff.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )

    return dff.iloc[
        page_current*page_size: (page_current + 1)*page_size
    ].to_dict('records')

@app.callback(
    Output('table-paging-with-graph-container', "children"),
    [Input('table-paging-with-graph', "data")])
def update_graph_table_paging(rows):
    """
    https://dash.plot.ly/datatable/callbacks
    """
    try:
        dff = pd.DataFrame(rows)
        return html.Div(
            [
                dcc.Graph(
                    id=column,
                    figure={
                        "data": [
                            {
                                "x": dff["country"],
                                "y": dff[column] if column in dff else [],
                                "type": "bar",
                                "marker": {"color": "#0074D9"},
                            }
                        ],
                        "layout": {
                            "xaxis": {"automargin": True},
                            "yaxis": {"automargin": True},
                            "height": 250,
                            "margin": {"t": 10, "l": 10, "r": 10},
                        },
                    },
                )
                # この↓処理はpageなくなるとエラーになるのでtryで避ける
                for column in ["pop", "lifeExp", "gdpPercap"]
            ]
        )
    except Exception as e:
        pass
if __name__=='__main__':
    # 接続ポートも指定して外部からアクセスできるようにする（http://localhost/80でアクセス）
    # https://qiita.com/tomboyboy/items/122dfdb41188176e45b5
    #app.run_server(debug=True, host='0.0.0.0', port=80)
    app.run_server(debug=True)

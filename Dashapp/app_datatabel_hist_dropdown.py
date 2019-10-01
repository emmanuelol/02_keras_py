# -*- coding: utf-8 -*-
"""
Dash(Pythonの可視化ライブラリ)使ったサンプルコード
100行単位でデータフレーム表示して、指定列のヒストグラムをplotするアプリ
https://dash.plot.ly/datatable/interactivity
（https://dash.plot.ly/dash-core-components）
Usage:
    $ activate tfgpu_py36
    $ python app_datatabel_hist_dropdown.py
    → http://xceedsys:6004/
"""
import dash
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

######## 入力データ（ロードするcsv変更する場合はここだけ変えたら良い） ########
#X_COL = 'country' # csvファイルでid列に変更する列名
#HIST_COL_LIST = ['All'] + ['pop', 'lifeExp', 'gdpPercap'] # 棒グラフ出す列名
#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

X_COL = 'METHOD_DETAIL' # csvファイルでid列に変更する列名
HIST_COL_LIST = ['All'] + ['DEC_VAL_REL'] # 棒グラフ出す列名
df = pd.read_csv('/gpfsx01/home/aaa00162/jupyterhub/notebook/H3-051/work/HIKARI/raw_represent.IC50.tsv', sep='\t')
##########################################################################

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Show dataframe and histgram'),

    dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
        ],
        data=df.to_dict('records'),
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        #row_selectable="multi",
        row_deletable=True,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 100,
        fixed_rows={'headers': True, 'data': 0}, # 縦スクロールしてもヘッダ固定
        style_table={
        #    'overflow': 'hidden', # コンテンツを1行に保持し、コンテンツが長すぎてセルに収まらない場合は「…」で表示。うまく行かない
        #    'textOverflow': 'ellipsis', # コンテンツを1行に保持し、コンテンツが長すぎてセルに収まらない場合は「…」で表示。うまく行かない
        #    'maxWidth': 0, # コンテンツを1行に保持し、コンテンツが長すぎてセルに収まらない場合は「…」で表示。うまく行かない
            'whiteSpace': 'normal', # セルにスペースを含むテキストが含まれている場合、コンテンツを複数行にオーバーフロー
            'height': 'auto', # セルにスペースを含むテキストが含まれている場合、コンテンツを複数行にオーバーフロー
            'overflowX': 'scroll', # 横スクロールバーつける
            'maxHeight': '10', # 最大縦幅
            'overflowY': 'scroll' # 縦スクロールバーつける
        }
    ),

    # 列名のDropdown
    html.Div([
        html.H4('Select histgram column'),
        dcc.Dropdown(
            id='xaxis-column',
            # available_indicator変数を選択肢として設定
            options=[{'label': i, 'value': i} for i in HIST_COL_LIST],
            # 初期値（allにして膨大な数plotされるの避けるためid=1のを初期表示とする）
            value=HIST_COL_LIST[1]#'all'
        )
    ],
    # 上の要素をひとまとめにstyleを設定する
    style={'width': '48%', 'display': 'inline-block'}),

    # Histgram
    html.Div(id='datatable-interactivity-container')
])

@app.callback(
    Output('datatable-interactivity', 'style_data_conditional'),
    [Input('datatable-interactivity', 'selected_columns')]
)
def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]

@app.callback(
    Output('datatable-interactivity-container', "children"),
    [Input('datatable-interactivity', "derived_virtual_data"),
     Input('datatable-interactivity', "derived_virtual_selected_rows"),
     Input('xaxis-column', 'value')
     ])
def update_graphs(rows, derived_virtual_selected_rows
                  , xaxis_column_name):
    # テーブルが最初にレンダリングされるとき、
    # `derived_virtual_data`と` derived_virtual_selected_rows`は `None`になります。
    # これは、Dashの特異性によるものです
    # （提供されていないプロパティは常にNoneであり、Dashはコンポーネントが最初にレンダリングされるときに依存コールバックを呼び出します）。
    # したがって、「rows」が「None」の場合、コンポーネントはレンダリングされたばかりで、その値はコンポーネントのデータフレームと同じになります。
    # ここで「なし」を設定する代わりに、コンポーネントを初期化するときに「derived_virtual_data = df.to_rows（ 'dict'）」を設定することもできます。
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dff = df if rows is None else pd.DataFrame(rows)

    colors = ['#7FDBFF' if i in derived_virtual_selected_rows else '#0074D9'
              for i in range(len(dff))]

    # Dropdownで指定した列のヒストグラムだけ表示するか
    if xaxis_column_name == HIST_COL_LIST[0]:
        plot_columns = HIST_COL_LIST
    else:
        plot_columns = [xaxis_column_name]

    return [
        dcc.Graph(
            id=column,
            figure={
                "data": [
                    {
                        "x": dff[X_COL],
                        "y": dff[column],
                        "type": "histogram"#,
                        #"marker": {"color": colors},
                    }
                ],
                "layout": {
                    "xaxis": {"automargin": True},
                    "yaxis": {
                        "automargin": True,
                        "title": {"text": column}
                    },
                    "height": 250,
                    "margin": {"t": 10, "l": 10, "r": 10},
                },
            },
        )
        # 列が存在するかどうかを確認します
        # column.deletable= False`の場合、このチェックを行う必要はありません。
        for column in plot_columns if column in dff
    ]


if __name__ == '__main__':
    # 接続ポートも指定して外部からアクセスできるようにする（http://localhost/80でアクセス）
    # https://qiita.com/tomboyboy/items/122dfdb41188176e45b5
    app.run_server(debug=True, host='0.0.0.0', port=6004)

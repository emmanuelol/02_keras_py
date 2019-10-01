"""
Dash(Pythonの可視化ライブラリ)使ったサンプルコード
10行単位でデータフレーム表示して、指定列のヒストグラムをplotするアプリ（ロードするデータフレームにid列必要）
https://dash.plot.ly/datatable/interactivity
（https://dash.plot.ly/dash-core-components）
Usage:
    $ activate tfgpu_py36_v3
    $ python app_datatable_id_hist.py
    → http://localhost/80
"""
import dash
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
#import plotly.express as px

######## 入力データ（ロードするcsv変更する場合はここだけ変えたら良い） ########
ID_COL = 'country' # csvファイルでid列に変更する列名
HIST_COL_LIST = ['pop', 'lifeExp', 'gdpPercap'] # 棒グラフ出す列名
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')
##########################################################################

# id列を追加し、インデックスとして設定します。
# この場合、一意のIDは国名にすぎないため、「country」の名前を「id」に変更することもできます
# （ただし、表示名は「country」になります）が、ここでは より一般的なパターンを示すためだけに複製されています。
df['id'] = df[ID_COL]
df.set_index('id', inplace=True, drop=False)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Show dataframe and histgram'),
    dash_table.DataTable(
        id='datatable-row-ids',
        columns=[
            {'name': i, 'id': i, 'deletable': True} for i in df.columns
            # omit the id column
            if i != 'id'
        ],
        data=df.to_dict('records'),
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode='multi',
        #row_selectable='multi',
        row_deletable=True,
        #selected_rows=[],
        page_action='native',
        page_current= 0,
        page_size= 10,
    ),
    html.Div(id='datatable-row-ids-container')
    ]
)


@app.callback(
    Output('datatable-row-ids-container', 'children'),
    [Input('datatable-row-ids', 'derived_virtual_row_ids'),
     Input('datatable-row-ids', 'selected_row_ids'),
     Input('datatable-row-ids', 'active_cell')
    ])
def update_graphs(row_ids, selected_row_ids, active_cell):
    # テーブルが最初にレンダリングされるとき、
    # `derived_virtual_data`と` derived_virtual_selected_rows`は `None`になります。
    # これは、Dashの特異性によるものです
    # （提供されていないプロパティは常にNoneであり、Dashはコンポーネントが最初にレンダリングされるときに依存コールバックを呼び出します）。
    # したがって、「rows」が「None」の場合、コンポーネントはレンダリングされたばかりで、その値はコンポーネントのデータフレームと同じになります。
    # ここで「なし」を設定する代わりに、コンポーネントを初期化するときに「derived_virtual_data = df.to_rows（ 'dict'）」を設定することもできます。
    selected_id_set = set(selected_row_ids or [])

    if row_ids is None:
        dff = df
        # pandas Series works enough like a list for this to be OK
        row_ids = df['id']
    else:
        dff = df.loc[row_ids]

    #active_row_id = active_cell['row_id'] if active_cell else None
    #
    #colors = ['#FF69B4' if id == active_row_id
    #          else '#7FDBFF' if id in selected_id_set
    #          else '#0074D9'
    #          for id in row_ids]

    return [
        dcc.Graph(
            id=column + '--row-ids',
            figure={
                # histgram
                # https://github.com/plotly/dash-recipes/blob/master/dash-plotly-132-histogram-data-simple.py
                'data': [{
                    'x': dff[column],
                    'name': column,
                    'type': 'histogram'
                }],
                'layout': {
                    'xaxis': {'automargin': True},
                    'yaxis': {
                        'automargin': True,
                        'title': {'text': column}
                    },
                    'height': 250,
                    'margin': {'t': 10, 'l': 10, 'r': 10},
                },
            },
        )
        # 列が存在するかどうかを確認します
        # column.deletable= False`の場合、このチェックを行う必要はありません。
        for column in HIST_COL_LIST if column in dff
    ]

if __name__=='__main__':
    # 接続ポートも指定して外部からアクセスできるようにする（http://localhost/80でアクセス）
    # https://qiita.com/tomboyboy/items/122dfdb41188176e45b5
    app.run_server(debug=True, host='0.0.0.0', port=80)

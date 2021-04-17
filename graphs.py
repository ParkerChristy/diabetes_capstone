import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd


# import diabetes data
diabetes = pd.read_csv("data/diabetes.csv")

app = dash.Dash(__name__)


app.layout = html.Div([
    html.P("This is a graph showing the distribution of Glucose and Insulin for both outcomes:",
            style={'text-align':'center'}),
    dcc.Graph(id="graph"),
    html.P("Select Distribution:"),
    dcc.RadioItems(
        id='dist-marginal',
        options=[{'label': x, 'value': x}
                 for x in ['box', 'violin', 'rug']],
        value='box'
    )
])


@app.callback(
    Output("graph", "figure"),
    [Input("dist-marginal", "value")])
def display_graph(marginal):
    fig = px.histogram(
        diabetes, x="Insulin", y="Glucose", color="Outcome",
        marginal=marginal, range_x=[-5, 400],
        hover_data=diabetes.columns)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

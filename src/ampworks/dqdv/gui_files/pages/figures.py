import plotly.graph_objs as go

from dash import dcc
from plotly.subplots import make_subplots

from ampworks.plotutils._plotly import PLOTLY_CONFIG, PLOTLY_TEMPLATE

placeholder_fig = go.Figure()

placeholder_fig.update_layout(
    uirevision='constant',
    plot_bgcolor='lightgrey',
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    paper_bgcolor='lightgrey',
    margin=dict(l=20, r=20, t=20, b=20),
)

placeholder_fig.add_annotation(
    align='center', showarrow=False,
    font=dict(size=20, color='black'),
    text='Upload data to populate figure',
    xref='paper', yref='paper', x=0.5, y=0.5,
)

figure_div = dcc.Graph(
    id='figure-div',
    responsive=True,
    config=PLOTLY_CONFIG,
    figure=placeholder_fig,
    style={'height': '450px', 'width': '100%'},
)

# 2 rows, 2 columns layout
figure = make_subplots(
    rows=2, cols=2,
    shared_xaxes='columns',
    vertical_spacing=0.05,
    horizontal_spacing=0.125,
    specs=[[{'rowspan': 2, 'secondary_y': True}, {}],
           [None, {}]]  # None spanned by left plot
)

for idx in [1, 2]:
    figure.update_xaxes(row=idx, col=idx, title_text='q [-]')

ylabels = ['dq/dV [1/V]', 'dV/dq [V]']
for i, row in enumerate([1, 2]):
    figure.update_yaxes(row=row, col=2, title_text=ylabels[i])

figure.update_yaxes(row=1, col=1, title_text='Voltage (pos/cell) [V]')
figure.update_yaxes(
    row=1, col=1, title_text='Voltage (neg) [V]', secondary_y=True)

figure.update_yaxes(mirror=False, row=1, col=1)
figure.update_yaxes(secondary_y=True, showgrid=False, row=1, col=1)

# vertical lines for soc=0,1
figure.add_shape(
    type='line',
    x0=0, x1=0,
    y0=0, y1=1,
    xref='x1', yref='paper',
    line=dict(color='rgba(64,64,64,1)', width=2, dash='dash')
)

figure.add_shape(
    type='line',
    x0=1, x1=1,
    y0=0, y1=1,
    xref='x1', yref='paper',
    line=dict(color='rgba(64,64,64,1)', width=2, dash='dash')
)

# data
figure.add_trace(go.Scatter(
    x=[], y=[], mode='markers',
    name='Data', legendgroup='data',
    marker=dict(
        size=8,
        symbol='circle',
        color='#c5c5c5',
    )),
    row=1, col=1,
)
figure.add_trace(go.Scatter(
    x=[], y=[], mode='markers',
    name='Data', legendgroup='data', showlegend=False,
    marker=dict(
        size=8,
        symbol='circle',
        color='#c5c5c5',
    )),
    row=1, col=2,
)
figure.add_trace(go.Scatter(
    x=[], y=[], mode='markers',
    name='Data', legendgroup='data', showlegend=False,
    marker=dict(
        size=8,
        symbol='circle',
        color='#c5c5c5',
    )),
    row=2, col=2,
)

# model
figure.add_trace(go.Scatter(
    x=[], y=[], mode='lines',
    name='Model', legendgroup='model',
    line=dict(
        width=2,
        color='black',
    )),
    row=1, col=1,
)
figure.add_trace(go.Scatter(
    x=[], y=[], mode='lines',
    name='Model', legendgroup='model', showlegend=False,
    line=dict(
        width=2,
        color='black',
    )),
    row=1, col=2,
)
figure.add_trace(go.Scatter(
    x=[], y=[], mode='lines',
    name='Model', legendgroup='model', showlegend=False,
    line=dict(
        width=2,
        color='black',
    )),
    row=2, col=2,
)

# positive electrode
figure.add_trace(go.Scatter(
    x=[], y=[], mode='lines', name='Pos',
    line=dict(
        width=2,
        color='#d62728',
    )),
    row=1, col=1,
)

# negative electrode
figure.add_trace(go.Scatter(
    x=[], y=[], mode='lines', name='Neg',
    line=dict(
        width=2,
        color='#1f77b4',
    )),
    row=1, col=1, secondary_y=True,
)

# annotations
figure.add_annotation(
    text="MAPE=nan%",
    x=0.5, y=0.95,
    xref="x domain", yref="y domain",
    showarrow=False,
    row=1, col=1,
)
figure.add_annotation(
    text="MAPE=nan%",
    x=0.9, y=0.9,
    xref="x domain", yref="y domain",
    showarrow=False,
    row=1, col=2,
)
figure.add_annotation(
    text="MAPE=nan%",
    x=0.9, y=0.9,
    xref="x domain", yref="y domain",
    showarrow=False,
    row=2, col=2,
)

figure.update_layout(
    uirevision='constant',
    template=PLOTLY_TEMPLATE,
    margin=dict(l=100, r=20, t=20, b=50),
)

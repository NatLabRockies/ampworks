import numpy as np
import dash_ag_grid as dag
import dash_bootstrap_components as dbc

from dash import dcc, html


# Sliders
# Keys must be built-in types; orjson can't serialize numpy floats as keys
marks = {0: '0.0', 1: '1.0'}
marks.update({float(i): f"{i:.1f}" for i in np.linspace(0.1, 0.9, 9)})

for k, v in marks.copy().items():
    marks[k] = {'label': v, 'style': {'fontSize': '1em'}}

sliders = dbc.Stack(
    [
        dbc.Label('Negative Electrode: [0.00, 1.00]',
                  id='neg-slider-label', class_name='bold-label mt-3'),
        dcc.RangeSlider(
            id='neg-slider', updatemode='drag',
            min=0.0, max=1.0, step=0.01, value=[0., 1.], marks=marks,
        ),

        dbc.Label('Positive Electrode: [0.00, 1.00]',
                  id='pos-slider-label', class_name='bold-label mt-3'),
        dcc.RangeSlider(
            id='pos-slider', updatemode='drag',
            min=0.0, max=1.0, step=0.01, value=[0., 1.], marks=marks,
        ),
    ],
    class_name='mx-auto',
    style={'maxWidth': '900px'},
)

# Optimization buttons
optimize_btns = dbc.Stack(
    [
        dbc.Button(
            'Grid Search',
            id='grid-btn',
            class_name='amp-btn',
        ),
        dbc.Button(
            'Minimize Error',
            id='min-err-btn',
            class_name='amp-btn',
        ),
    ],
    gap=3,
    direction='horizontal',
    style={'maxWidth': '900px'},
    class_name='justify-content-end my-3 mx-auto',
)

page_spinner = dbc.Spinner(
    size='lg',
    color='white',
    fullscreen=True,
    id='page-spinner',
    children=html.Div(id='spinner-div'),
    fullscreen_style={
        'zIndex': '2000',
        'backgroundColor': 'rgba(0, 0, 0, 0.5)',
    },
)

# Terminal output
terminal = dbc.Row(
    [
        dcc.Markdown(
            id='terminal-out',
            className='dbc markdown',
        ),
    ],
    style={'maxWidth': '900px'},
    class_name='pb-3 pt-0 px-0 m-0 mx-auto',
)

# Logging buttons
download = dcc.Download(id="download-csv")
logging_btns = dbc.Stack(
    [
        dbc.Button('Log Output', id='add-row-btn',
                   class_name='log-btn success'),
        dbc.Button('Delete Selected', id='delete-btn',
                   class_name='log-btn danger'),
        dbc.Button('Export Data', id='export-btn',
                   class_name='log-btn secondary'),
    ],
    gap=3,
    direction='horizontal',
    style={'maxWidth': '900px'},
    class_name='justify-content-end mx-auto mb-3',
)

# Logging table
column_defs = [
    {
        'minWidth': 125,
        'field': 'filename',
        'headerName': 'filename',
        'checkboxSelection': True,
        'headerCheckboxSelection': True,
    },
    *[
        {'headerName': f"{c}", 'field': f"{c}", 'minWidth': 100}
        for c in [
            'Ah', 'xn0', 'xn0_std', 'xn1', 'xn1_std', 'xp0', 'xp0_std', 'xp1',
            'xp1_std', 'iR', 'iR_std', 'fun', 'success', 'message',
        ]
    ],
]

logging_table = dbc.Row(
    dag.AgGrid(
        rowData=[],
        id='ag-grid',
        columnDefs=column_defs,
        defaultColDef={
            'flex': 1,
            'minWidth': 80,
            'sortable': False,
        },
        dashGridOptions={
            'domLayout': 'normal',
            'rowSelection': 'multiple',
        },
    ),
    style={'maxWidth': '900px'},
    class_name='amp-ag-grid mx-auto mb-3',
)

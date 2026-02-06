import dash
import numpy as np
import pandas as pd

from dash import dcc, html, Output, Input, State

from ampworks.utils import RichResult
from ampworks.dqdv.gui_files.pages.figures import figure_div
from ampworks.dqdv.gui_files.pages.components import (
    sliders, optimize_btns, page_spinner, terminal, logging_btns,
    download, logging_table,
)

dash.register_page(
    __name__,
    path='/dqdv-fitting',
    title='dQdV Fitting',
    page_components=[
        figure_div,
        sliders,
        optimize_btns,
        page_spinner,
        terminal,
        logging_btns,
        download,
        logging_table,
    ],
)

# Page layout
layout = html.Div()

# Callbacks


@dash.callback(
    Output('terminal-out', 'children'),
    Input('summary-store', 'data'),
)
def update_terminal(summary):
    if not summary:
        summary = {
            'message': None,
            'success': None,
            'nfev': None,
            'niter': None,
            'fun': None,
            'Ah': None,
            'x': None,
            'x_std': None,
            'x_map': ['xn0', 'xn1', 'xp0', 'xp1', 'iR'],
        }

    if summary['x'] is not None:
        summary['x'] = np.array(summary['x'])

    if summary['x_std'] is not None:
        x_std = [np.nan if std is None else std for std in summary['x_std']]
        summary['x_std'] = np.array(x_std)

    formatted = RichResult(**summary)

    return f"```text\n{formatted!r}\n```"


@dash.callback(
    Output('ag-grid', 'rowData'),
    Input('add-row-btn', 'n_clicks'),
    State('ag-grid', 'rowData'),
    State('summary-store', 'data'),
    State({'type': 'filename', 'index': 'cell'}, 'children'),
    prevent_initial_call=True,
)
def log_new_row(_, current_data, summary, filename):

    if not current_data:
        current_data = []

    if not summary:
        return current_data

    row = {}
    row['filename'] = filename.removesuffix('.csv')
    row['Ah'] = summary['Ah']

    for name, x, std in zip(summary['x_map'], summary['x'], summary['x_std']):
        row[name] = x
        row[name + '_std'] = 'nan' if std is None else std

    row['fun'] = summary['fun']
    row['success'] = str(summary['success'])
    row['message'] = summary['message']

    current_data.append(row)

    return current_data


@dash.callback(
    Output('ag-grid', 'deleteSelectedRows'),
    Input('delete-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def selected(_):
    return True


@dash.callback(
    Output('download-csv', 'data'),
    Input('export-btn', 'n_clicks'),
    State('ag-grid', 'rowData'),
    State('ag-grid', 'columnDefs'),
    prevent_initial_call=True,
)
def export_to_csv(_, row_data, column_defs):
    if row_data:
        df = pd.DataFrame(row_data)
        ordered_cols = [c['field'] for c in column_defs]
        df = df[ordered_cols]
        return dcc.send_data_frame(df.to_csv, 'DVQ_Data.csv', index=False)
    else:
        return dash.no_update

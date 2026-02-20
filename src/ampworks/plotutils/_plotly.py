from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence, TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:  # pragma: no cover
    from plotly.graph_objs._figure import Figure

__all__ = ['PLOTLY_TEMPLATE', 'PLOTLY_CONFIG', '_render_plotly']

PLOTLY_TEMPLATE = go.layout.Template(
    layout=dict(
        dragmode='pan',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12, color='#212529'),
        xaxis=dict(
            showline=True, linecolor='#212529', title_standoff=7,
            mirror='all', ticks='inside', tickcolor='#212529',
            minor=dict(
                ticklen=2,
                ticks='inside',
            ),
        ),
        yaxis=dict(
            showline=True, linecolor='#212529', title_standoff=7,
            mirror='all', ticks='inside', tickcolor='#212529',
            minor=dict(
                ticklen=2,
                ticks='inside',
            ),
        ),
        legend=dict(
            orientation='h',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            entrywidth=0.125, entrywidthmode='fraction',
            xanchor='center', x=0.5, yanchor='top', y=1.15,
        ),
        margin=dict(l=80, r=80, t=60, b=80),
    )
)

PLOTLY_CONFIG = {
    'scrollZoom': True,
    'responsive': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
}


def _render_plotly(
    fig: Figure,
    figsize: Sequence[int, int] | None = None,
    save: str | None = None,
) -> None:
    """
    Render a plotly figure.

    Automatically determines whether the code is running in a notebook or from
    a script. When run from a notebook, the figure is rendered inline. From a
    script, the figure is opened in a local web browser. It is either opened
    from the save location, or from a temporary directory, if not saved.

    Parameters
    ----------
    fig : Figure
        The plotly figure to be rendered.
    figsize : Sequence[int, int] | None, optional
        The size of the figure (width, height), by default None. If None, the
        default plotly size is used. You may also specify one dimension as None
        to make it responsive (i.e., adjust to the page) in that dimension.
    save : str | None, optional
        The file path to save the figure, by default None.

    """
    from ampworks import _in_notebook

    # Configure size, with optional responsiveness
    config = PLOTLY_CONFIG.copy()
    if figsize is not None:
        fig.update_layout(width=figsize[0], height=figsize[1])
        config['responsive'] = any([size is None for size in figsize])

    # Save or create temp file to display when not in notebook
    if save is not None:
        path = Path(save)
        if not path.suffix.lower() == '.html':
            path = path.with_suffix('.html')

        path.parent.mkdir(parents=True, exist_ok=True)

    else:
        tmp = NamedTemporaryFile(delete=False, suffix='.html')
        path = Path(tmp.name)
        tmp.close()

    # Optionally write to file, then display
    in_nb = _in_notebook()

    if (not in_nb) or (save is not None):
        auto_open = True if not in_nb else False
        fig.write_html(path, auto_open=auto_open, config=config)

    if in_nb:
        fig.show(config=config)

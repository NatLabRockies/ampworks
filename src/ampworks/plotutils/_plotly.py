from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from tempfile import NamedTemporaryFile

import plotly.graph_objects as go

if TYPE_CHECKING:  # pragma: no cover
    from plotly.graph_objs._figure import Figure as PlotlyFigure

__all__ = [
    'PLOTLY_TEMPLATE',
    'PLOTLY_CONFIG',
    '_apply_plotly_style',
    '_render_plotly',
]

PLOTLY_TEMPLATE = go.layout.Template(
    layout=dict(
        hovermode='x',
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


def _apply_plotly_style(fig: PlotlyFigure) -> None:
    """
    Style a plotly figure.

    Parameters
    ----------
    fig : PlotlyFigure
        The plotly figure to be styled.

    """
    fig.update_layout(template=PLOTLY_TEMPLATE)


def _render_plotly(
    fig: PlotlyFigure,
    figsize: tuple[int, int] | None = None,
    save: str | None = None,
) -> None:
    """
    Render a plotly figure.

    Determine whether to render the figure inline in a notebook or open in the
    browser from a user-saved or temporary HTML file.

    Parameters
    ----------
    fig : PlotlyFigure
        The plotly figure to be rendered.
    figsize : tuple[int, int] | None, optional
        The size of the figure (width, height), by default None. Set either or
        both dimensions to None to allow them to stretch.
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

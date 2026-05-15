from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from tempfile import NamedTemporaryFile

from IPython.display import display, HTML
from bokeh.models import LinearAxis, Span, CustomJS
from bokeh.io import (
    reset_output, output_notebook, output_file, show, save as bk_save,
)

if TYPE_CHECKING:  # pragma: no cover
    from bokeh.plotting import figure as BokehFigure

__all__ = ['_apply_bokeh_style', '_render_bokeh']


def _apply_bokeh_style(fig: BokehFigure) -> None:
    """
    Style a bokeh figure.

    Parameters
    ----------
    fig : BokehFigure
        The bokeh figure to be styled.

    """
    # margin and borders
    fig.margin = (7, 7, 7, 7)  # (left, right, top, bottom)

    fig.min_border_top = 60
    fig.min_border_left = 80
    fig.min_border_right = 80
    fig.min_border_bottom = 80

    # adjust xrange (no left/right padding)
    fig.x_range.range_padding = 0

    # Primary axes (bottom and left)
    for ax in fig.axis:
        ax.major_tick_in = 6
        ax.major_tick_out = 0
        ax.minor_tick_in = 3
        ax.minor_tick_out = 0

        ax.axis_label_text_font_size = '10pt'
        ax.axis_label_text_font_style = 'normal'

        ax.major_label_text_font_size = '9pt'
        ax.major_label_text_font_style = 'normal'

    # Mirrored axes on top and right (ticks only, no labels)
    for position in ('above', 'right'):
        ax = LinearAxis(
            major_tick_in=6,
            major_tick_out=0,
            minor_tick_in=3,
            minor_tick_out=0,
        )

        ax.major_label_text_font_size = '0pt'

        fig.add_layout(ax, position)

    # Add spanning grid lines for the x=0 and y=0 axes
    for direction in ('width', 'height'):
        span = Span(
            location=0,
            line_width=1,
            line_color='black',
            dimension=direction,
        )

        fig.add_layout(span)

    # Hide Bokeh toolbar logo
    fig.toolbar.logo = None

    # JS Callback to trigger reset tool on double-click
    callback = CustomJS(args=dict(fig=fig), code='fig.reset.emit()')
    fig.js_on_event('doubletap', callback)


def _render_bokeh(
    fig: BokehFigure,
    figsize: tuple[int, int] | None = None,
    save: str | None = None,
) -> None:
    """
    Render a Bokeh figure.

    Automatically determines whether the code is running in a notebook or from
    a script. When run from a notebook, the figure is rendered inline. From a
    script, the figure is opened in a local web browser. It is either opened
    from the save location, or from a temporary directory, if not saved.

    Parameters
    ----------
    fig : BokehFigure
        The Bokeh figure to be rendered.
    figsize : tuple[int, int] | None, optional
        The size of the figure (width, height), by default None. If None, the
        default bokeh size is used. You may also specify one dimension as None
        to make it responsive (i.e., adjust to the page) in that dimension.
    save : str | None, optional
        The file path to save the figure, by default None.

    """
    from ampworks import _in_notebook

    fig.width, fig.height = figsize if figsize is not None else (None, None)

    fig.min_width = max(550, fig.width or 0)
    fig.min_height = max(300, fig.height or 0)

    if (fig.width is None) and (fig.height is None):
        fig.sizing_mode = 'stretch_both'
    elif fig.width is None:
        fig.sizing_mode = 'stretch_width'
    elif fig.height is None:
        fig.sizing_mode = 'stretch_height'
    else:
        fig.sizing_mode = 'fixed'

    reset_output()

    # Save or create temp file to display when not in notebook
    if save is not None:
        path = Path(save)
        if path.suffix.lower() != '.html':
            path = path.with_suffix('.html')

        path.parent.mkdir(parents=True, exist_ok=True)

    else:
        tmp = NamedTemporaryFile(delete=False, suffix='.html')
        path = Path(tmp.name)
        tmp.close()

    str_path = str(path)

    # Optionally write to file, then display
    in_nb = _in_notebook()

    if save is not None:
        bk_save(fig, filename=str_path, resources='cdn', title=path.name)

    if not in_nb:
        output_file(filename=str_path, mode='cdn', title=path.name)
        show(fig)
    elif save is not None:
        display(HTML(str_path))
    else:
        output_notebook(hide_banner=True)
        show(fig)

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from tempfile import NamedTemporaryFile

from IPython.display import display, HTML
from bokeh import io as bk_io, models as bk_m

if TYPE_CHECKING:  # pragma: no cover
    from bokeh.plotting import figure as BokehFigure

__all__ = [
    'BOKEH_TEMPLATE',
    'BOKEH_CONFIG',
    '_apply_bokeh_style',
    '_render_bokeh',
]

BOKEH_TEMPLATE = {
    'margin': (7, 7, 7, 7),  # (top, right, bottom, left)
    'border': {'top': 60, 'left': 80, 'right': 80, 'bottom': 80},
    'minor_tick_len': 3,
    'major_tick_len': 6,
    'font_family': 'Arial',
    'font_size': '10pt',
    'font_style': 'normal',
}

BOKEH_CONFIG = {
    'active_scroll': 'wheel_zoom',
    'tools': ['pan', 'box_zoom', 'wheel_zoom', 'save', 'reset'],
}


def _apply_bokeh_style(fig: BokehFigure) -> None:
    """
    Style a bokeh figure.

    Parameters
    ----------
    fig : BokehFigure
        The bokeh figure to be styled.

    """
    # margin and borders
    fig.margin = BOKEH_TEMPLATE['margin']

    fig.min_border_top = BOKEH_TEMPLATE['border']['top']
    fig.min_border_left = BOKEH_TEMPLATE['border']['left']
    fig.min_border_right = BOKEH_TEMPLATE['border']['right']
    fig.min_border_bottom = BOKEH_TEMPLATE['border']['bottom']

    # adjust xrange (no left/right padding)
    fig.x_range.range_padding = 0

    # Primary axes (bottom and left)
    for ax in fig.axis:
        ax.minor_tick_out = 0
        ax.major_tick_out = 0

        ax.minor_tick_in = BOKEH_TEMPLATE['minor_tick_len']
        ax.major_tick_in = BOKEH_TEMPLATE['major_tick_len']

        ax.axis_label_text_font_size = BOKEH_TEMPLATE['font_size']
        ax.axis_label_text_font_style = BOKEH_TEMPLATE['font_style']

        ax.major_label_text_font_size = BOKEH_TEMPLATE['font_size']
        ax.major_label_text_font_style = BOKEH_TEMPLATE['font_style']

    # Mirrored axes on top and right (ticks only, no labels)
    for position in ('above', 'right'):
        ax = bk_m.LinearAxis(
            minor_tick_out=0,
            major_tick_out=0,
            minor_tick_in=BOKEH_TEMPLATE['minor_tick_len'],
            major_tick_in=BOKEH_TEMPLATE['major_tick_len'],
        )

        ax.major_label_text_font_size = '0pt'

        fig.add_layout(ax, position)

    # Add spanning grid lines for the x=0 and y=0 axes
    for direction in ('width', 'height'):
        span = bk_m.Span(
            location=0,
            line_width=1,
            line_color='black',
            dimension=direction,
        )

        fig.add_layout(span)

    # Hide Bokeh toolbar logo
    fig.toolbar.logo = None

    # JS Callback to trigger reset tool on double-click
    callback = bk_m.CustomJS(args=dict(fig=fig), code='fig.reset.emit()')
    fig.js_on_event('doubletap', callback)


def _render_bokeh(
    fig: BokehFigure,
    figsize: tuple[int, int] | None = None,
    save: str | None = None,
) -> None:
    """
    Render a Bokeh figure.

    Determine whether to render the figure inline in a notebook or open in the
    browser from a user-saved or temporary HTML file.

    Parameters
    ----------
    fig : BokehFigure
        The bokeh figure to be rendered.
    figsize : tuple[int, int] | None, optional
        The size of the figure (width, height), by default None. Set either or
        both dimensions to None to allow them to stretch.
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

    bk_io.reset_output()

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
        bk_io.save(fig, filename=str_path, resources='cdn', title=path.name)

    if not in_nb:
        bk_io.output_file(filename=str_path, mode='cdn', title=path.name)
        bk_io.show(fig)
    elif save is not None:
        display(HTML(str_path))
    else:
        bk_io.output_notebook(hide_banner=True)
        bk_io.show(fig)

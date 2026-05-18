import pytest
import numpy as np

import plotly.graph_objects as go

from bokeh.plotting import figure
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from ampworks import plotutils as aplt
from ampworks.plotutils._bokeh import _render_bokeh
from ampworks.plotutils._plotly import _render_plotly


# tests for _colors submodule
class TestColorMap:

    def test_init(self):
        cm = aplt.ColorMap('viridis', (0, 1))

        assert cm._vmin == 0
        assert cm._vmax == 1
        assert hasattr(cm, '_sm')

        with pytest.raises(ValueError):
            aplt.ColorMap('viridis', (0,))  # norm must be length 2

        with pytest.raises(ValueError):
            aplt.ColorMap('viridis', (1, 0))  # vmin must be < vmax

    def test_get_color(self):
        cm = aplt.ColorMap('viridis', (0, 1))
        color = cm.get_color(0.5)

        assert isinstance(color, tuple)
        assert len(color) == 4  # RGBA

        with pytest.raises(ValueError):
            cm.get_color(1.5)

    def test_colors_from_size(self):
        size = 5
        colors = aplt.colors_from_size(size, 'viridis')

        assert isinstance(colors, list)
        assert len(colors) == size
        assert all(len(c) == 4 for c in colors)

    def test_colors_from_data(self):
        data = np.array([[0, 0.5], [0.8, 1]])
        colors = aplt.colors_from_data(data, 'viridis')

        assert isinstance(colors, np.ndarray)
        assert colors.shape == data.shape

        for row in colors:
            for c in row:
                assert len(c) == 4


# tests for _text submodule
class TestAddText:

    def test_add_text(self):
        fig, ax = plt.subplots()

        aplt.add_text(ax, 0.1, 0.1, 'First')
        aplt.add_text(ax, 0.9, 0.9, 'Second')

        texts = [t.get_text() for t in ax.texts]

        assert 'First' in texts
        assert 'Second' in texts
        assert len(ax.texts) == 2

        plt.close(fig)

    def test_add_text_alignment(self):
        fig, ax = plt.subplots()

        aplt.add_text(ax, 0.3, 0.7, 'Aligned', ha='left', va='top')

        text = ax.texts[0]

        assert text.get_ha() == 'left'
        assert text.get_va() == 'top'

        plt.close(fig)


# tests for _ticks submodule
class TestTicks:

    def test_minor_ticks_defaults(self):
        fig, ax = plt.subplots()
        aplt.minor_ticks(ax)

        assert isinstance(ax.xaxis.get_minor_locator(), AutoMinorLocator)
        assert isinstance(ax.yaxis.get_minor_locator(), AutoMinorLocator)

        plt.close(fig)

    def test_minor_ticks_custom(self):
        fig, ax = plt.subplots()
        aplt.minor_ticks(ax, xdiv=4, ydiv=3)

        xloc = ax.xaxis.get_minor_locator()
        yloc = ax.yaxis.get_minor_locator()

        assert isinstance(xloc, AutoMinorLocator) and xloc.ndivs == 4
        assert isinstance(yloc, AutoMinorLocator) and yloc.ndivs == 3

        plt.close(fig)

    def test_tick_direction_defaults(self):
        fig, ax = plt.subplots()
        aplt.tick_direction(ax)

        xparams = ax.xaxis.get_tick_params()
        yparams = ax.yaxis.get_tick_params()

        if 'top' not in xparams.keys():  # for backwards mpl, v <= 3.9
            xparams['top'] = xparams.get('right')

        assert xparams['direction'] == 'in' and xparams['top']
        assert yparams['direction'] == 'in' and yparams['right']

        plt.close(fig)

    def test_tick_direction_custom(self):
        fig, ax = plt.subplots()
        aplt.tick_direction(
            ax, xdir='out', ydir='inout', top=False, right=False,
        )

        xparams = ax.xaxis.get_tick_params()
        yparams = ax.yaxis.get_tick_params()

        if 'top' not in xparams.keys():  # for backwards mpl, v <= 3.9
            xparams['top'] = xparams.get('right')

        assert xparams['direction'] == 'out' and not xparams['top']
        assert yparams['direction'] == 'inout' and not yparams['right']

        plt.close(fig)

    def test_format_ticks(self):
        fig, ax = plt.subplots()
        aplt.format_ticks(
            ax,
            xdiv=4, ydiv=3,
            xdir='out', ydir='inout',
            top=False, right=False,
        )

        xloc = ax.xaxis.get_minor_locator()
        yloc = ax.yaxis.get_minor_locator()

        xparams = ax.xaxis.get_tick_params()
        yparams = ax.yaxis.get_tick_params()

        assert isinstance(xloc, AutoMinorLocator) and xloc.ndivs == 4
        assert isinstance(yloc, AutoMinorLocator) and yloc.ndivs == 3

        if 'top' not in xparams.keys():  # for backwards mpl, v <= 3.9
            xparams['top'] = xparams.get('right')

        assert xparams['direction'] == 'out' and not xparams['top']
        assert yparams['direction'] == 'inout' and not yparams['right']

        plt.close(fig)


# tests for plotutils._plotly._render_plotly
class TestRenderPlotly:

    @pytest.fixture(autouse=True)
    def _not_in_notebook(self, monkeypatch):
        monkeypatch.setattr('ampworks._in_notebook', lambda: False)

    def test_save_writes_file_and_opens(self, tmp_path, monkeypatch):
        opened = []
        monkeypatch.setattr(
            'webbrowser.open', lambda url, **kw: opened.append(url),
        )

        save_path = tmp_path / 'chart.html'
        _render_plotly(go.Figure(), save=str(save_path))

        assert len(opened) == 1
        assert save_path.exists()

    def test_save_adds_html_extension(self, tmp_path, monkeypatch):
        monkeypatch.setattr('webbrowser.open', lambda *a, **kw: None)

        _render_plotly(go.Figure(), save=str(tmp_path / 'chart'))

        assert (tmp_path / 'chart.html').exists()

    def test_no_save_creates_temp_html(self, tmp_path, monkeypatch):
        opened = []
        monkeypatch.setattr(
            'webbrowser.open', lambda url, **kw: opened.append(url),
        )

        _render_plotly(go.Figure())

        assert len(opened) == 1
        assert opened[0].endswith('.html')
        assert opened[0].startswith('file://')


# tests for plotutils._bokeh._render_bokeh
class TestRenderBokeh:

    @pytest.fixture(autouse=True)
    def _not_in_notebook(self, monkeypatch):
        monkeypatch.setattr('ampworks._in_notebook', lambda: False)

    def test_save_writes_file_and_opens(self, tmp_path, monkeypatch):
        shown = []
        monkeypatch.setattr(
            'bokeh.io.show', lambda *a, **kw: shown.append(True),
        )

        out = []
        monkeypatch.setattr(
            'bokeh.io.output_file', lambda *a, **kw: out.append(kw['filename']),
        )

        save_path = tmp_path / 'chart.html'
        _render_bokeh(figure(), save=str(save_path))

        assert len(shown) == 1
        assert save_path.exists()

        assert len(out) == 1
        assert out[0] == str(save_path)

    def test_save_adds_html_extension(self, tmp_path, monkeypatch):
        monkeypatch.setattr('bokeh.io.show', lambda *a, **kw: None)

        _render_bokeh(figure(), save=str(tmp_path / 'chart'))

        assert (tmp_path / 'chart.html').exists()

    def test_no_save_creates_temp_html(self, monkeypatch):
        shown = []
        monkeypatch.setattr(
            'bokeh.io.show', lambda *a, **kw: shown.append(True),
        )

        out = []
        monkeypatch.setattr(
            'bokeh.io.output_file', lambda *a, **kw: out.append(kw['filename']),
        )

        _render_bokeh(figure())

        assert len(shown) == 1

        assert len(out) == 1
        assert out[0].endswith('.html')

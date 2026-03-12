import pytest
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # non-interactive backend


@pytest.fixture(autouse=True)
def ignore_exithandler_plt_show(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)

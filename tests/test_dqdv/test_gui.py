import pytest

pytest.importorskip('dash')


@pytest.mark.parametrize('route', ['/', '/dqdv-fitting', '/_dash-layout'])
def test_gui_routes_serve(route):
    from ampworks.dqdv.gui_files._gui import app

    client = app.server.test_client()
    assert client.get(route).status_code == 200

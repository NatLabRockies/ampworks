import os

import pytest
import ampworks as amp


def test_list_datasets():
    from ampworks.datasets import RESOURCES

    # requested invalid or empty module
    with pytest.raises(ValueError):
        _ = amp.datasets.list_datasets('fake')

    # valid requests
    full_truth = []
    for folder in os.listdir(RESOURCES):
        subdir = RESOURCES.joinpath(folder)
        files = [folder + '/' + f for f in os.listdir(subdir)]
        full_truth.extend(files)

    full = amp.datasets.list_datasets()
    assert set(full) == set(full_truth)

    ici = amp.datasets.list_datasets('ici')
    ici_truth = [f for f in full_truth if f.startswith('ici/')]
    assert set(ici) == set(ici_truth)

    subset = amp.datasets.list_datasets('gitt', 'ici')
    subset_truth = [
        f for f in full_truth if f.startswith('gitt/') or f.startswith('ici/')
    ]
    assert set(subset) == set(subset_truth)


def test_download_all(tmp_path):
    from ampworks.datasets import RESOURCES

    amp.datasets.download_all(path=tmp_path)

    downloaded = os.listdir(tmp_path.joinpath('ampworks_datasets'))
    truth = os.listdir(RESOURCES)
    assert set(downloaded) == set(truth)


def test_load_datasets():

    # need at least one name
    with pytest.raises(ValueError):
        _ = amp.datasets.load_datasets()

    # requested invalid name
    with pytest.raises(ValueError):
        _ = amp.datasets.load_datasets('fake')

    # single dataset
    names = amp.datasets.list_datasets()
    names = [n for n in names if n.startswith('hppc')]

    assert len(names) > 0

    data = amp.datasets.load_datasets(names[0])

    assert isinstance(data, amp.Dataset)
    assert data.shape[0] > 0
    assert data.shape[1] > 0

    # more than one dataset
    data = amp.datasets.load_datasets(names[0], names[0])
    assert len(data) == 2

    hppc0, hppc1 = data
    assert hppc0.equals(hppc1)

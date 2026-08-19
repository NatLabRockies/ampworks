import os
import pathlib

import pytest
import pandas as pd
import polars as pl
import ampworks as amp

from ampworks.datasets import RESOURCES


class TestListDatasets:

    def test_invalid_module_raises(self):
        with pytest.raises(ValueError):
            amp.datasets.list_datasets('fake')

    def test_all_datasets(self):
        full_truth = []
        for folder in os.listdir(RESOURCES):
            subdir = RESOURCES.joinpath(folder)
            files = [folder + '/' + f for f in os.listdir(subdir)]
            full_truth.extend(files)

        full = amp.datasets.list_datasets()
        assert set(full) == set(full_truth)

    def test_single_module(self):
        full_truth = amp.datasets.list_datasets()

        ici = amp.datasets.list_datasets('ici')
        ici_truth = [f for f in full_truth if f.startswith('ici/')]
        assert set(ici) == set(ici_truth)

    def test_multiple_modules(self):
        full_truth = amp.datasets.list_datasets()

        subset = amp.datasets.list_datasets('gitt', 'ici')
        ici_truth = [f for f in full_truth if f.startswith('ici/')]
        gitt_truth = [f for f in full_truth if f.startswith('gitt/')]
        subset_truth = ici_truth + gitt_truth
        assert set(subset) == set(subset_truth)


class TestDownloadAll:
    full_truth = amp.datasets.list_datasets()

    def test_invalid_format_raises(self, tmp_path):
        with pytest.raises(ValueError):
            amp.datasets.download_all(path=tmp_path, format='fake')

    def test_default_format_copies_parquet(self, tmp_path):
        amp.datasets.download_all(path=tmp_path)

        downloaded = []
        for path in tmp_path.joinpath('ampworks_datasets').rglob('*.parquet'):
            downloaded.append('/'.join(path.parts[-2:]))

        assert set(downloaded) == set(self.full_truth)

    def test_default_path_uses_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        amp.datasets.download_all()

        downloaded = []
        for path in tmp_path.joinpath('ampworks_datasets').rglob('*.parquet'):
            downloaded.append('/'.join(path.parts[-2:]))

        assert set(downloaded) == set(self.full_truth)

    @pytest.mark.parametrize('format', ['csv', 'txt'])
    def test_formats_preserve_structure(self, tmp_path, format):
        amp.datasets.download_all(path=tmp_path, format=format)

        downloaded = []
        for path in tmp_path.joinpath('ampworks_datasets').rglob(f"*.{format}"):
            downloaded.append('/'.join(path.parts[-2:]))

        mod_truth = [
            f.replace('.parquet', f'.{format}') for f in self.full_truth
        ]

        assert set(downloaded) == set(mod_truth)

    @pytest.mark.parametrize('format', ['csv', 'txt'])
    def test_formats_preserve_content(self, tmp_path, format):
        amp.datasets.download_all(path=tmp_path, format=format)

        out_dir = tmp_path.joinpath('ampworks_datasets')
        sample = next(pathlib.Path(RESOURCES).rglob('*.parquet'))
        folder, file = sample.parts[-2:]

        original = pl.read_parquet(sample).to_pandas()

        file = file.replace('.parquet', f".{format}")
        new_path = out_dir.joinpath(folder, file)

        if format == 'csv':
            converted = pl.read_csv(new_path).to_pandas()
        elif format == 'txt':
            converted = pl.read_csv(new_path, separator='\t').to_pandas()

        pd.testing.assert_frame_equal(original, converted, check_dtype=False)


class TestLoadDatasets:

    def test_no_names_raises(self):
        with pytest.raises(ValueError):
            amp.datasets.load_datasets()

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError):
            amp.datasets.load_datasets('fake')

    def test_single_dataset(self):
        names = amp.datasets.list_datasets()
        names = [n for n in names if n.startswith('hppc')]
        assert len(names) > 0

        data = amp.datasets.load_datasets(names[0])

        assert isinstance(data, amp.Dataset)
        assert not data.empty

    def test_multiple_datasets(self):
        names = amp.datasets.list_datasets()
        names = [n for n in names if n.startswith('hppc')]

        data = amp.datasets.load_datasets(names[0], names[0])
        assert len(data) == 2

        hppc0, hppc1 = data
        assert hppc0.equals(hppc1)
        assert not hppc0.empty
        assert not hppc1.empty

    def test_parquet_extension_is_optional(self):
        names = amp.datasets.list_datasets()
        name = [n for n in names if n.startswith('hppc')][0]

        with_ext = amp.datasets.load_datasets(name)
        without_ext = amp.datasets.load_datasets(name.removesuffix('.parquet'))

        assert with_ext.equals(without_ext)

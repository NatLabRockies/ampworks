import pytest

import ampworks as amp

from ampworks._core._headers import DEFAULT_ALIASES


class TestHeaderAliases:

    def test_defaults_stored_under_standard_names(self):
        aliases = amp.HeaderAliases()
        assert aliases.keys() == list(DEFAULT_ALIASES.keys())
        for std_header, default in DEFAULT_ALIASES.items():
            assert aliases[std_header] == default

    def test_snake_case_keywords_map_to_standard_names(self):
        aliases = amp.HeaderAliases(
            seconds={'elapsed_s': None},
            datetime=['Timestamp'],
        )
        assert aliases['Seconds'] == {'elapseds': None}
        assert aliases['DateTime'] == {'timestamp'}
        assert aliases['Amps'] == DEFAULT_ALIASES['Amps']

    def test_pascal_case_keywords_raise(self):
        with pytest.raises(TypeError):
            amp.HeaderAliases(Seconds={'elapsed_s': None})

    def test_extend_defaults_subset_uses_keyword_names(self):
        aliases = amp.HeaderAliases(
            seconds={'elapsed_s': None},
            amps={'amps_raw': None},
            extend_defaults=['seconds'],
        )
        assert 'elapseds' in aliases['Seconds']
        assert set(DEFAULT_ALIASES['Seconds']).issubset(aliases['Seconds'])
        assert aliases['Amps'] == {'ampsraw': None}

    def test_extend_defaults_invalid_name_raises(self):
        with pytest.raises(ValueError, match='invalid'):
            amp.HeaderAliases(extend_defaults=['Seconds'])

    def test_errors_reference_keyword_name(self):
        with pytest.raises(TypeError, match="'cycle'"):
            amp.HeaderAliases(cycle='CycleNumber')

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            amp.HeaderAliases()['seconds']

import pytest
import pandas as pd

from ampworks._core._backend import (
    _build_default_alias,
    _strip_chars,
    _astype_float,
    _find_alias_match,
)


class TestBuildDefaultAlias:

    def test_units_none_returns_plain_set(self):
        result = _build_default_alias({'cycle', 'cyc'}, None)
        assert result == {'cycle', 'cyc'}

    def test_units_given_includes_bare_and_dotted_variants(self):
        result = _build_default_alias({'current'}, {'a': 1.0, 'ma': 1e-3})

        assert result['a'] == 1.0  # bare unit
        assert result['ma'] == 1e-3  # bare unit
        assert result['current'] == 1.0  # bare name, assumed base unit
        assert result['current.a'] == 1.0  # name.unit
        assert result['current.ma'] == 1e-3  # name.unit

    def test_multiple_names_share_the_same_units(self):
        result = _build_default_alias({'current', 'i'}, {'a': 1.0})
        assert result['current.a'] == 1.0
        assert result['i.a'] == 1.0


class TestStripChars:

    def test_none_passthrough(self):
        assert _strip_chars(None) is None

    def test_list_passthrough(self):
        result = _strip_chars(['Time [s]', 'Current_A'])
        assert result == ['time.s', 'current.a']

    @pytest.mark.parametrize(
        'raw, expected',
        [
            ('Time [s]', 'time.s'),
            ('Current_A', 'current.a'),  # underscore normalizes to a period
            ('Test_Time_s', 'test.time.s'),
            ('  Cycle Number  ', 'cyclenumber'),  # spaces are just removed
            ('Volts(V)', 'volts.v'),
            ('a,,b', 'a.b'),  # consecutive periods collapse to one
            ('_Test_Time_s', 'test.time.s'),  # leading period gets stripped
        ],
    )
    def test_normalizes(self, raw, expected):
        assert _strip_chars(raw) == expected


class TestAstypeFloat:

    def test_numeric_series_untouched_values(self):
        series = pd.Series([1, 2, 3])
        result = _astype_float(series)
        assert result.dtype == float
        assert result.to_list() == [1.0, 2.0, 3.0]

    def test_strips_commas_and_hashes(self):
        series = pd.Series(['1,000', '#2', '3'])
        result = _astype_float(series)
        assert result.to_list() == [1000.0, 2.0, 3.0]

    def test_non_numeric_becomes_nan(self):
        series = pd.Series(['abc', '1'])
        result = _astype_float(series)
        assert result.iloc[1] == 1.0
        assert pd.isna(result.iloc[0])


class TestFindAliasMatch:

    def test_exact_dotted_match(self):
        norm_raw = {'testtime.s': 'TestTime_s'}
        result = _find_alias_match(norm_raw, {'testtime.s': 1.0})
        assert result == ('testtime.s', 'TestTime_s')

    def test_extra_period_collapses_onto_registered_key(self):
        # 'Test_Time_s' normalizes to 'test.time.s' (2 periods), but only
        # 'testtime.s' (1 period) is registered; the extra separator between
        # 'test'/'time' should still collapse to reach that match
        norm_raw = {'test.time.s': 'Test_Time_s'}
        result = _find_alias_match(norm_raw, {'testtime.s': 1.0})
        assert result == ('testtime.s', 'Test_Time_s')

    def test_bare_key_glues_from_extra_separator(self):
        norm_raw = {'cycle.number': 'Cycle_Number'}
        result = _find_alias_match(norm_raw, {'cyclenumber'})
        assert result == ('cyclenumber', 'Cycle_Number')

    def test_plain_word_does_not_match_name_unit_alias(self):
        # a header with no real separator (e.g. a plural 'TestTimes') must not
        # be treated as if it encoded a 'testtime' + 's' unit split
        norm_raw = {'testtimes': 'TestTimes'}
        alias = {'testtime.s': 1.0, 'testtime': 1.0}
        result = _find_alias_match(norm_raw, alias)
        assert result is None

    def test_no_match_returns_none(self):
        norm_raw = {'foo': 'Foo'}
        result = _find_alias_match(norm_raw, {'bar': 1.0})
        assert result is None

    def test_returns_first_match_in_iteration_order(self):
        norm_raw = {'volts.v': 'Volts (V)', 'current.a': 'Current (A)'}
        result = _find_alias_match(norm_raw, {'current.a': 1.0})
        assert result == ('current.a', 'Current (A)')

    def test_invalid_alias_key_shape_raises(self):
        norm_raw = {'a.b.c': 'A_B_C'}
        with pytest.raises(ValueError):
            _find_alias_match(norm_raw, {'a.b.c': 1.0})

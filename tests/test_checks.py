import pytest

import ampworks as amp
import ampworks._checks as checks


def test_check_literal():

    valid = {'a', 'b', None}
    checks._check_literal('test', 'a', valid)
    checks._check_literal('test', 'b', valid)
    checks._check_literal('test', None, valid)

    with pytest.raises(ValueError):
        checks._check_literal('test', 'c', valid)


def test_check_columns():

    data = amp.Dataset({'col1': [1, 2], 'col2': [3, 4]})
    checks._check_columns(data, ['col1'])
    checks._check_columns(data, ['col2'])
    checks._check_columns(data, ['col1', 'col2'])

    with pytest.raises(ValueError):
        checks._check_columns(data, ['col3'])

    with pytest.raises(ValueError):
        checks._check_columns(data, ['col1', 'col3'])


def test_check_type():

    checks._check_type('test', 1, int)
    checks._check_type('test', 'a', str)
    checks._check_type('test', None, None)
    checks._check_type('test', None, type(None))
    checks._check_type('test', 1.0, (int, float, None))

    with pytest.raises(TypeError):
        checks._check_type('test', 1, str)

    with pytest.raises(TypeError):
        checks._check_type('test', 'a', int)

    with pytest.raises(TypeError):
        checks._check_type('test', 1.0, int)


def test_check_inner_type():

    checks._check_inner_type('test', [1, 2], int)
    checks._check_inner_type('test', ['a', 'b'], str)
    checks._check_inner_type('test', [1.0, 2.0], (int, float))

    with pytest.raises(TypeError):
        checks._check_inner_type('test', [1, 'a'], int)

    with pytest.raises(TypeError):
        checks._check_inner_type('test', ['a', 1], str)

    with pytest.raises(TypeError):
        checks._check_inner_type('test', [1.0, 'a'], (int, float))


def test_check_only_one():

    checks._check_only_one([True, False, False], 'Pass.')
    checks._check_only_one([False, True, False], 'Pass.')
    checks._check_only_one([False, False, True], 'Pass.')

    with pytest.raises(ValueError, match='Fail, two true.'):
        checks._check_only_one([True, True, False], 'Fail, two true.')

    with pytest.raises(ValueError, match='Fail, none true.'):
        checks._check_only_one([False, False, False], 'Fail, none true.')

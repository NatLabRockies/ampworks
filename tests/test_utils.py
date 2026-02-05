import time

from tempfile import NamedTemporaryFile

import pytest
import numpy as np
import pandas as pd
import ampworks as amp


def test_alphanum_sort():

    unsorted = [
        'apple3',
        'banana12',
        'grape10',
        'orange2',
        'pear7',
        'kiwi5',
        'date1',
        '10apple',
        '2banana',
        '100grape',
    ]

    sorted_list = [
        '2banana',
        '10apple',
        '100grape',
        'apple3',
        'banana12',
        'date1',
        'grape10',
        'kiwi5',
        'orange2',
        'pear7',
    ]

    sorted_test = amp.utils.alphanum_sort(unsorted)
    assert sorted_test == sorted_list

    sorted_list.reverse()

    reversed_test = amp.utils.alphanum_sort(unsorted, reverse=True)
    assert reversed_test == sorted_list
    assert reversed_test != sorted_test


def test_progbar_initialization():

    with pytest.raises(ValueError, match='conflicting'):
        _ = amp.utils.ProgressBar(iterable=[1, 2, 3], manual=True)

    with pytest.raises(ValueError, match='cannot be None'):
        _ = amp.utils.ProgressBar(iterable=None, manual=False)

    bar = amp.utils.ProgressBar(iterable=[1, 2, 3])
    assert bar._manual is False
    assert bar.total == 3

    bar = amp.utils.ProgressBar(manual=True)
    assert bar._manual is True
    assert bar.total == 1


def test_iterable_progbar():
    iterable = range(10)

    bar = amp.utils.ProgressBar(iterable)
    for i in bar:
        pass

    assert bar._manual is False


def test_manual_progbar():

    bar = amp.utils.ProgressBar(manual=True)
    for i in range(10):
        bar.set_progress(0.1*(i+1))

    assert bar._manual is True
    assert bar._iter == 10

    bar.reset()
    assert bar._iter == 0


def test_RichResult():

    # basic
    result = amp.utils.RichResult()
    assert result._order_keys == []
    assert repr(result) == 'RichResult()'

    # access via dict-style or attr-style
    result = amp.utils.RichResult(a=1, c=np.random.rand(5))
    assert (result.a == result['a']) and (result.a is result['a'])
    assert np.all(result.c == result['c']) and (result.c is result['c'])

    # subclassing
    class NewResult(amp.utils.RichResult):
        pass

    result = NewResult()
    assert repr(result) == 'NewResult()'

    # repr ordering
    class OrderedResult(amp.utils.RichResult):
        _order_keys = ['first', 'second',]

    new = NewResult(second=None, first=None)
    ordered = OrderedResult(second=None, first=None)
    assert new == ordered
    assert repr(new) != repr(ordered)
    assert dir(ordered) == sorted(ordered.keys())

    # copy
    copy = ordered.copy()
    assert isinstance(copy, amp.utils.RichResult)
    assert (copy == ordered) and (copy is not ordered)


def test_format_float_10():
    from ampworks.utils._rich_result import _format_float_10

    assert _format_float_10(np.inf) == '       inf'
    assert _format_float_10(-np.inf) == '      -inf'
    assert _format_float_10(np.nan) == '       nan'

    assert _format_float_10(0.123456789) == ' 1.235e-01'
    assert _format_float_10(1.234567890) == ' 1.235e+00'
    assert _format_float_10(1234.567890) == ' 1.235e+03'


def test_timer():

    # invalid units
    with pytest.raises(ValueError):
        timer = amp.utils.Timer(units='fake')

    # basic
    def f():
        time.sleep(1e-3)
        return 0.

    with amp.utils.Timer('success') as timer:
        _ = f()

    assert timer.name == 'success'
    assert timer.elapsed_time >= 0.
    assert timer._converter['s'](3600.) == 3600.
    assert timer._converter['min'](3600.) == 60.
    assert timer._converter['h'](3600.) == 1.


def test_RichTable():

    # basic
    df = pd.DataFrame({'a': [0, 1], 'b': [2, 3]})
    table = amp.utils.RichTable(df)

    assert df is not table.df
    assert table._required_cols == []
    assert repr(table) == repr(df)

    # access via dict-style or attr-style
    assert np.all(table.a == table['a'])
    assert np.shares_memory(table.a.to_numpy(), table['a'].to_numpy())
    assert np.all(table[['a', 'b']] == df)

    # no direct assignment
    with pytest.raises(TypeError):
        table['a'] = 1

    with pytest.raises(AttributeError):
        table.a = 1

    # subclassing
    class NewTable(amp.utils.RichTable):
        _required_cols = ['c', 'd']

    with pytest.raises(ValueError):
        _ = NewTable(df)

    df2 = df.rename(columns={'a': 'c', 'b': 'd'})
    new_table = NewTable(df2)
    assert new_table.df.equals(df2)

    # copy
    copy = new_table.copy()
    assert isinstance(copy, NewTable)
    assert isinstance(copy, amp.utils.RichTable)
    assert (copy.df.equals(new_table.df)) and (copy is not new_table)

    # to/from csv
    with NamedTemporaryFile(suffix='.csv') as tmp:
        tmp.close()

        new_table.to_csv(tmp.name)
        read = NewTable.from_csv(tmp.name)

        assert new_table.df.equals(read.df)

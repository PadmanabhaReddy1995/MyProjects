# Name : Pandas_cub
# Author: G V Padmanabha Reddy
# Description : A simple library based on Pandas named as pandas_cub
# Reference : Udemy course by "Ted Petrou"

import math
import numpy as np

__version__ = '0.0.1'


class DataFrame:

    def __init__(self, data):
        """
        A DataFrame holds two dimensional heterogeneous data. Create it by
        passing a dictionary of NumPy arrays to the values parameter

        Parameters
        ----------
        data: dict
            A dictionary of strings mapped to NumPy arrays. The key will
            become the column name.
        """


        # check for correct input types
        self._check_input_types(data)

        # check for equal array lengths
        self._check_array_lengths(data)

        # convert unicode arrays to object
        self._data = self._convert_unicode_to_object(data)

    ## The input must be a dictionary with keys being strings and values being 1-D numpy arrays
    def _check_input_types(self, data):
        if not isinstance(data, dict):
            raise TypeError("'data' must be a dictionary")
        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError("The keys of 'data' must be strings")
            if not isinstance(value, np.ndarray):
                raise TypeError("The values of 'data' must be numpy arrays")
            if not value.ndim == 1:
                raise ValueError("The values of 'data' musgt be 1-D numpy arrays" )

    ## All the numpy arrays in the input must be of equal length
    def _check_array_lengths(self, data):
        for index, value in enumerate(data.values()):
            if index == 0:
                length = len(value)
            elif len(value) != length:
                raise ValueError("All the numpy arrays in the input must be of equal length")

    def _convert_unicode_to_object(self, data):
        new_data = {}
        for key, value in data.items():
            if value.dtype.kind == 'U':
                new_data[key] = value.astype('object')
            else:
                new_data[key] = value

        return new_data

    def __len__(self):
        """
        Make the builtin len function work with our dataframe

        Returns
        -------
        int: the number of rows in the dataframe
        """
        return len(next(iter(self._data.values())))

    @property
    def columns(self):
        """
        _data holds column names mapped to arrays
        take advantage of internal ordering of dictionaries to
        put columns in correct order in list. Only works in 3.6+

        Returns
        -------
        list of column names
        """
        return list(self._data)

    ## To update the column names of the DataFrame
    ## Number of columns in the input must be equal to the number of columns in the DataFrame
    @columns.setter
    def columns(self, columns):
        """
        Must supply a list of columns as strings the same length
        as the current DataFrame

        Parameters
        ----------
        columns: list of strings

        Returns
        -------
        None
        """
        if not isinstance(columns, list):
            raise TypeError("The input must be a list")
        if not len(columns) == len(self._data):
            raise ValueError("Number of items in the columns must(input) be equal to the number of columns in the DataFrame")
        for i in columns:
            if not isinstance(i, str):
                raise TypeError("All the items in the 'columns' must be strings")

        if not len(columns) == len(set(columns)):
            raise ValueError("There should not be any duplicates in the 'columns'")

        self._data = dict(zip(columns, self._data.values()))


    @property
    def shape(self):
        """
        Returns
        -------
        two-item tuple of number of rows and columns
        """
        rows = len(next(iter(self._data.values())))
        columns = len(self._data)

        return rows, columns

    ## To display the DataFrame in the Jupyter Notebook in a tabular format
    def _repr_html_(self):
        """
        Used to create a string of HTML to nicely display the DataFrame
        in a Jupyter Notebook. Different string formatting is used for
        different data types.

        The structure of the HTML is as follows:
        <table>
            <thead>
                <tr>
                    <th>data</th>
                    ...
                    <th>data</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>{i}</strong></td>
                    <td>data</td>
                    ...
                    <td>data</td>
                </tr>
                ...
                <tr>
                    <td><strong>{i}</strong></td>
                    <td>data</td>
                    ...
                    <td>data</td>
                </tr>
            </tbody>
        </table>
        """
        html = '<table><thead><tr><th></th>'
        for col in self.columns:
            html += f"<th>{col:10}</th>"

        html += '</tr></thead>'
        html += "<tbody>"

        only_head = False
        num_head = 10
        num_tail = 10
        if len(self) <= 20:
            only_head = True
            num_head = len(self)

        for i in range(num_head):
            html += f'<tr><td><strong>{i}</strong></td>'
            for col, values in self._data.items():
                kind = values.dtype.kind
                if kind == 'f':
                    html += f'<td>{values[i]:10.3f}</td>'
                elif kind == 'b':
                    html += f'<td>{values[i]}</td>'
                elif kind == 'O':
                    v = values[i]
                    if v is None:
                        v = 'None'
                    html += f'<td>{v:10}</td>'
                else:
                    html += f'<td>{values[i]:10}</td>'
            html += '</tr>'

        if not only_head:
            html += '<tr><strong><td>...</td></strong>'
            for i in range(len(self.columns)):
                html += '<td>...</td>'
            html += '</tr>'
            for i in range(-num_tail, 0):
                html += f'<tr><td><strong>{len(self) + i}</strong></td>'
                for col, values in self._data.items():
                    kind = values.dtype.kind
                    if kind == 'f':
                        html += f'<td>{values[i]:10.3f}</td>'
                    elif kind == 'b':
                        html += f'<td>{values[i]}</td>'
                    elif kind == 'O':
                        v = values[i]
                        if v is None:
                            v = 'None'
                        html += f'<td>{v:10}</td>'
                    else:
                        html += f'<td>{values[i]:10}</td>'
                html += '</tr>'

        html += '</tbody></table>'
        return html

    @property
    def values(self):
        """
        Returns
        -------
        A list of lists - with each list being one column in the DataFrame
        """
        # return np.column_stack(list(self._data.values()))
        length = self.__len__()
        index = -1
        ans = []
        for i in range(length):
            l = []
            index += 1
            for value in self._data.values():
                l.append(value[index])
            ans.append(l)

        return ans


    @property
    def dtypes(self):
        """
        Returns
        -------
        A two-column DataFrame of column names in one column and
        their data type in the other
        """
        DTYPE_NAME = {'O': 'string', 'i': 'int', 'f': 'float', 'b': 'bool'}

        column_names = np.array(list(self._data.keys()))
        dtypes = np.array([DTYPE_NAME[value.dtype.kind] for value in self._data.values()])
        new_data = {'Column_names' : column_names, 'dtypes' : dtypes}

        return DataFrame(new_data)

    def __getitem__(self, item):
        """
        Use the brackets operator to simultaneously select rows and columns
        A single string selects one column -> df['colname']
        A list of strings selects multiple columns -> df[['colname1', 'colname2']]
        Row and column selection simultaneously -> df[rs, cs]


        Returns
        -------
        A subset of the original DataFrame
        """
        if isinstance(item, str):
            return DataFrame({item : self._data[item]})

        if isinstance(item, list):
            return DataFrame({i:self._data[i] for i in item})

        if isinstance(item, tuple):
            row_selection, column_selection = item
            if isinstance(column_selection, int):
                column_selection = self.columns[column_selection]

            row = list(self._data[column_selection])
            row = np.array([row[row_selection]])

            return DataFrame({column_selection: row})

    def _ipython_key_completions_(self):
        # allows for tab completion when doing df['c
        return self.columns

    def __setitem__(self, key, value):
        # adds a new column or overwrites an old column
        self._data[key] = value

    def head(self, n=5):
        """
        Return the first n rows

        Parameters
        ----------
        n: int

        Returns
        -------
        DataFrame
        """
        new_data = dict()

        for key, value in self._data.items():
            new_data[key] = value[:n-1]

        return DataFrame(new_data)

    def tail(self, n=5):
        """
        Return the last n rows

        Parameters
        ----------
        n: int

        Returns
        -------
        DataFrame
        """

        new_data = dict()
        for key, value in self._data.items():
            new_data[key] = value[-1:-n-1:-1]
        return DataFrame(new_data)

    #### Aggregation Methods ####

    def min(self, column):
        key = column
        value = self._data[key]
        if value.dtype.kind == 'i' or value.dtype.kind == 'f':
            return min(list(value))

    def max(self, column):
        key = column
        value = self._data[key]
        if value.dtype.kind == 'i' or value.dtype.kind == 'f':
            return min(list(value))
        else:
            return "unsupported DataType - method only applicable for int and float values"

    def mean(self, column):
        key = column
        value = self._data[key]
        if value.dtype.kind == 'i' or value.dtype.kind == 'f':
            mean = sum(self._data[key])/len(self._data)
            return mean
        else:
            return "unsupported DataType - method only applicable for int and float values"

    def sum(self, column):
        key = column
        s = sum(list(self._data[key]))
        return s

    def var(self, column):
        key = column
        value = self._data[key]
        if value.dtype.kind == 'i' or value.dtype.kind == 'f':
            m = self.mean(key)
            s = 0
            for i in value:
                s += (i - m)
            variance = s/self.__len__()
        else:
            return "unsupported DataType - method only applicable for int and float values"

        return variance

    def std(self, column):
        var = self.var(column)
        standard_deviation = math.sqrt(var)

        return standard_deviation

    def argmax(self, column):
        key = column
        value = list(self._data[key])
        max_item = value.index(max(value))

        return max_item

    def argmin(self, column):
        key = column
        value = list(self._data[key])
        min_item = value.index(min(value))

        return min_item

    def unique(self):
        """
        Finds the unique values of each column

        Returns
        -------
        A list of one-column DataFrames
        """
        new_data = {}
        for key, value in self._data.items():
            new_data[key] = [set(value)]

        return new_data

    def nunique(self):
        """
        Find the number of unique values in each column

        Returns
        -------
        A DataFrame
        """
        new_data = {}
        for key, value in self._data.items():
            new_data[key] = [set(value), len(set(value))]

        return new_data


    def rename(self, columns):
        """
        Renames columns in the DataFrame

        Parameters
        ----------
        columns: dict
            A dictionary mapping the old column name to the new column name

        Returns
        -------
        A DataFrame
        """
        new_data = {}
        for key,value in self._data.items():
            if key in columns:
                new_data[columns[key]] = value
            else:
                new_data[key] = value

        return DataFrame(new_data)

    def drop(self, columns):
        """
        Drops one or more columns from a DataFrame

        Parameters
        ----------
        columns: str or list of strings

        Returns
        -------
        A DataFrame
        """
        new_data = {}
        for key, value in self._data.items():
            if key not in columns:
                new_data[key] = value

        return DataFrame(new_data)

    #### Non-Aggregation Methods ####

    def abs(self, columns):
        """
        Takes a list of columns for which the absolute value is to calculated.

        Returns
        -------
        A DataFrame
        """
        new_data = dict()
        if not isinstance(columns, list):
            raise TypeError("'columns' should be a list with one or multiple keys")
        for key in columns:
            if self._data[key].dtype.kind == 'i' or self._data[key].dtype.kind == 'f':
                new_data[key] = abs(self._data[key])
            else:
                raise TypeError("abs() can be calculated only for Integer and Float values")

        return DataFrame(new_data)


    def clip(self,column, lower=None, upper=None):
        """
        All values less than lower will be set to lower
        All values greater than upper will be set to upper

        Parameters
        ----------
        lower: number or None
        upper: number or None

        Returns
        -------
        A DataFrame
        """

        for key, value in self._data.items():
            if key == column:
                if value.dtype.kind == 'i' or value.dtype.kind == 'f':
                    for index, v in enumerate(value):
                        if v < lower:
                            value[index] = lower
                        elif v > upper:
                            value[index] = upper

        return DataFrame(self._data)


    def round(self, n):
        """
        Rounds values to the nearest n decimals

        Returns
        -------
        A DataFrame
        """
        for key, value in self._data.items():
            if value.dtype.kind == 'f':
                for index, v in enumerate(value):
                    value[index] = round(v, n)

        return DataFrame(self._data)

    def title(self, col):
        for key, value in self._data.items():
            if key == col:
                for index, v in enumerate(value):
                    value[index] = v.title()

        return DataFrame(self._data)

    def lower(self, col):
        for key, value in self._data.items():
            if key == col:
                for index, v in enumerate(value):
                    value[index] = v.lower()

        return DataFrame(self._data)

    def upper(self, col):
        for key, value in self._data.items():
            if key == col:
                for index, v in enumerate(value):
                    value[index] = v.upper()

        return DataFrame(self._data)

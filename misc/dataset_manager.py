import os
import numpy as np
import pandas as pd

import misc.utils as utils

import logging
logger = logging.getLogger(__name__)

class DatasetManager(object):
    """ Handle standard parsing operations for sklearn-style datasets.

    Additionally, this class extracts several dataset properties needed by
    auto-sklearn.

    Properties
    ----------
    data_file: string or None
        The path to the data file if it is read

    df: pandas.DataFrame
        The data frame containing the data instances (not including the
        target variable)

    target_variable: string
        The name of the target variable

    X, y: np.arrays
        Arrays suitable for use as inputs to sklearn. The variables are not,
        for example, one-hot encoded, standardized, etc.

    numerical_fields, categorical_fields, datatime_fields, timedelta_fields:
            dictionaries mapping from field names to column indices

        Dictionaries mapping from a field name to the original dataset index
        of the respective types
    """
    
    def __init__(self, 
            data_file=None,
            df=None,
            cleanup=None,
            target_variable=None,
            drop_rows_with_missing_values=False,
            drop_rows_with_missing_target=True,
            convert_int_to_float=True,
            *args, **kwargs):
        
        # we must have either a data_path or a df
        self.data_file = data_file
        self.df = df
        
        if data_file is not None:
            # make sure we do not have both
            if df is not None:
                msg = ("[DatasetManager]: init was given both a data file "
                    "and data frame. Please provide only one or the other.")
                raise ValueError(msg)
                
            if not os.path.exists(data_file):
                msg = ("[DatasetManage]: could not find the data file. "
                "Looking here: {}".format(data_file))
                raise FileNotFoundError(msg)

            self.df = pd.read_csv(data_file, *args, **kwargs)
            
        if self.df is None:
            msg = ("[DatasetManager]: init was given neither a data file"
                "nor a data frame. Please provide exactly one or the other.")
            raise ValueError(msg)
            
        if cleanup is not None:            
            self.df = self.df.replace(cleanup)
            
        # ignore missing values for now
        if drop_rows_with_missing_values:
            self.df = self.df.dropna()
            self.df = self.df.reset_index(drop=True)
        
        # pull out the target if we were given one
        self.y = None
        self.target_variable = None
        if target_variable is not None:
            self.target_variable = target_variable
            self.y = self.df[target_variable].values
            self.df = self.df.drop([target_variable], axis=1)
            
        # remove rows with missing target variables
        if drop_rows_with_missing_target:
            m_y_is_missing = ~np.isnan(self.y)
            self.y = self.y[m_y_is_missing]
            self.df = self.df[m_y_is_missing]
                        

        # get the field names for the different types
        numerical_types = ['float64', 'int64']
        numerical_fields = self.df.select_dtypes(include=numerical_types)
        self.numerical_fields = list(numerical_fields.columns)
        self.numerical_fields = {
            f: self.df.columns.get_loc(f) for f in self.numerical_fields
        }

        categorical_types = ['object', 'category']
        categorical_fields = self.df.select_dtypes(include=categorical_types)
        self.categorical_fields = list(categorical_fields.columns)
        self.categorical_fields = {
            f: self.df.columns.get_loc(f) for f in self.categorical_fields
        }


        datetime_types = [np.datetime64, 'datetime', 'datetime64']
        datetime_fields = self.df.select_dtypes(include=datetime_types)
        self.datetime_fields = list(datetime_fields.columns)
        self.datetime_fields = {
            f: self.df.columns.get_loc(f) for f in self.datetime_fields
        }


        timedelta_types = [np.timedelta64, 'timedelta', 'timedelta64']
        timedelta_fields = self.df.select_dtypes(include=timedelta_types)
        self.timedelta_fields = list(timedelta_fields.columns)
        self.timedelta_fields = {
            f: self.df.columns.get_loc(f) for f in self.timedelta_fields
        }

        # convert int
        if convert_int_to_float:
            msg = ("[DatasetManager] converting int columns to floats")
            logger.info(msg)
            numerical_fields = self.get_numerical_field_names()
            self.df[numerical_fields] = self.df[numerical_fields].astype(float)

        self.X = self.df.values

    def get_categorical_df(self):
        """ Return a data frame copy containing only the categorical fields
        """
        cat_df = self.df[list(self.categorical_fields.keys())]
        cat_df = cat_df.copy()
        return cat_df

    def get_numerical_df(self):
        """ Return a data frame copy containing only the numerical fields
        """
        num_df = self.df[list(self.numerical_fields.keys())]
        num_df = num_df.copy()
        return num_df

    def get_datetime_df(self):
        """ Return a data frame copy containing only the datetime fields
        """
        dt_df = self.df[list(self.datetime_fields.keys())]
        dt_df = dt_df.copy()
        return dt_df

    def get_timedelta_df(self):
        """ Return a data frame copy containing only the timedelta_fields
        """
        td_df = self.df[list(self.timedelta_fields.keys())]
        td_df = td_df.copy()
        return td_df

    def get_categorical_field_names(self):
        """ Return an np.array of the categorical field names.

        The array is sorted by the field index.
        """
        return np.array(utils.sort_dict_keys_by_value(self.categorical_fields))

    def get_numerical_field_names(self):
        """ Return an np.array of the numerical field names.

        The np.array is sorted by the field index.
        """
        return np.array(utils.sort_dict_keys_by_value(self.numerical_fields))

    def get_datetime_field_names(self):
        """ Return an np.array of the datetime field names.

        The np.array is sorted by the field index.
        """
        return np.array(utils.sort_dict_keys_by_value(self.datetime_fields))

    def get_timedelta_field_names(self):
        """ Return an np.array of the timedelta field names.

        The np.array is sorted by the field index.
        """
        return np.array(utils.sort_dict_keys_by_value(self.timedelta_fields))

    def get_categorical_field_indices(self):
        """ Return an np.array of the categorical field indices.

        The array is sorted by the field index.
        """
        return np.array(sorted(self.categorical_fields.values()))

    def get_numerical_field_indices(self):
        """ Return an np.array of the numerical field indices.

        The np.array is sorted by the field index.
        """
        return np.array(sorted(self.numerical_fields.values()))

    def get_datetime_field_indices(self):
        """ Return an np.array of the datetime field indices.

        The np.array is sorted by the field index.
        """
        return np.array(sorted(self.datetime_fields.values()))

    def get_timedelta_field_indices(self):
        """ Return an np.array of the timedelta field indices.

        The np.array is sorted by the field index.
        """
        return np.array(sorted(self.timedelta_fields.values()))








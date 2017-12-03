import os
import numpy as np
import pandas as pd

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
            
        self.X = self.df.values

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


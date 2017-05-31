###
#
#   This library contains various helper functions for parallelizing independent loops.
#   The functions mostly wrap joblib.Parallel.
#
###

import logging
import sys 

def apply_parallel_iter(items, num_procs, func, *args, progress_bar=False, total=None, num_groups=None):
    """ This function parallelizes applying a function to all items in an iterator using the 
        joblib library. In particular, func is called for each of the items in the list. (Unless
        num_groups is given. In this case, the iterator must be "split-able", e.g., a list or
        np.array. Then func is called with a list of items.)
        
        This function is best used when func has little overhead compared to the item processing.

        It always returns a list of the values returned by func. The order of the
        returned list is dependent on the semantics of joblib.Parallel, but it is typically
        in the same order as the groups

        Args:
            items (list-like): A list (or anything that can be iterated over in a for loop)

            num_procs (int): The number of processors to use

            func (function pointer): The function to apply to each group

            args (variable number of arguments): The other arguments to pass to func

            progress_bar (boolean) : whether to use a tqdm progress bar

            total (int) : the total number of items in the list. This only needs to be used if
                len(items) is not defined. It is only used for tqdm, so it is not necessary.

            num_groups (int) : if given, the number of groups into which the input is split. In
                case it is given, then each call to func will be passed a list of items.

        Returns:
            list: the values returned from func for each item (in the order specified by
                joblib.Parallel). If num_groups is given, then this is likely to be a list.

        Imports:
            joblib
            numpy
            tqdm, if progress_bar is True
    """
    import joblib
    import numpy as np

    # check if we want groups
    if num_groups is not None:
        items_per_group = int(np.ceil(len(items) / num_groups))
        # then make items a list of lists, where each internal list contains items from the original list
        items = [ items[i * items_per_group : (i+1) * items_per_group] for i in range(num_groups)]

    if progress_bar:
        import tqdm
        ret_list = joblib.Parallel(n_jobs=num_procs)(joblib.delayed(func)(item, *args) 
            for item in tqdm.tqdm(items, leave=True, file=sys.stdout, total=total))
    else:
        ret_list = joblib.Parallel(n_jobs=num_procs)(joblib.delayed(func)(item, *args) for item in items)
    return ret_list


def apply_parallel_groups(groups, num_procs, func, *args, progress_bar=False):
    """ This function parallelizes applying a function to groupby results using the 
        joblib library. In particular, func is called for each of the groups in groups. 
        
        This function is best used when func has little overhead compared to the group processing.

        Unlike DataFrame.groupby().apply(func), this function does not attempt to guess
        the type of return value (i.e., DataFrame sometimes, Series sometimes, etc.). It
        just always returns a list of the values returned by func. The order of the
        returned list is dependent on the semantics of joblib.Parallel, but it is typically
        in the same order as the groups

        Args:
            groups (pandas.Groupby): The result of a call to DataFrame.groupby

            num_procs (int): The number of processors to use

            func (function pointer): The function to apply to each group

            args (variable number of arguments): The other arguments to pass to func

        Returns:
            list: the values returned from func for each group (in the order specified by
                joblib.Parallel)

        Imports:
            joblib
            tqdm, if progress_bar is True
    """
    import joblib
    if len(groups) == 0:
        return []

    if progress_bar:
        import tqdm
        ret_list = joblib.Parallel(n_jobs=num_procs)(joblib.delayed(func)(group, *args) 
            for name,group in tqdm.tqdm(groups, total=len(groups), leave=True, file=sys.stdout))
    else:
        ret_list = joblib.Parallel(n_jobs=num_procs)(joblib.delayed(func)(group, *args) 
            for name, group in groups)
    return ret_list

def apply_groups(groups, func, *args, progress_bar=False):
    """ Apply func to each group in groups.

    The primary reason to use this over groups.apply(.) is that this directly
    returns an array of the results rather than coercing things to a series.

    Parameters
    ----------
    groups: pd.GroupBy
        The groups

    func: function pointer
        The function to call for each group

    args: variable number of arguments
        The other arguments to pass to func

    progress_bar: bool
        Whether to show a progress bar

    Returns
    -------
    results: list
        The values returned from func for each group. The order is specified by
        joblib.Parallel; typically, this is the same order as the groups. 
    """
    num_procs = 1
    return apply_parallel_groups(
        groups, 
        num_procs, 
        func, 
        *args, 
        progress_bar=progress_bar
    )

def apply_parallel_split(data_frame, num_procs, func, *args, progress_bar=False, num_groups=None):
    """ This function parallelizes applying a function to the rows of a data frame using the
        joblib library. The data frame is first split into num_procs equal-sized groups, and
        then func is called on each of the groups.

        This function is best used when func has a large amount of overhead compared to the
        other processing. For example, if a large file needs to be read for processing each
        group, then it would be better to use this function than apply_parallel_groups.

        Otherwise, the semantics are the same for this function as for apply_parallel_groups.

        
        Args:
            data_frame (pandas.DataFrame): A data frame

            num_procs (int): The number of processors to use

            func (function pointer): The function to apply to each row in the data frame

            args (variable number of arguments): The other arguments to pass to func

        Returns:
            list: the values returned from func for each group (in the order specified by
                joblib.Parallel)

        Imports:
            numpy
            joblib (indirectly)
            tqdm (indirectly), if progress_bar is True
    """
    import numpy as np

    if num_groups is None:
        num_groups = num_procs

    parallel_indices = np.arange(len(data_frame)) // (len(data_frame) / num_groups)
    split_groups = data_frame.groupby(parallel_indices)
    res = apply_parallel_groups(split_groups, num_procs, func, *args, progress_bar=progress_bar)
    return res

def apply_parallel(data_frame, num_procs, func, *args, progress_bar=False):
    """ This function parallelizes applying a function to the rows of a data frame using the
        joblib library. The function is called on each row individually.

        This function is best used when func does not have much overhead compared to
        the row-specific processing. For example, this function is more appropriate than
        apply_parallel_split when all of the processing in func is dependent only on the 
        values in the data rows.

        Args:
            data_frame (pandas.DataFrame): A data frame

            num_procs (int): The number of processors to use

            func (function pointer): The function to apply to each row in the data frame

            args (variable number of arguments): The other arguments to pass to func

        Returns:
            list: the values returned from func for each row (in the order specified by
                joblib.Parallel)

        Imports:
            joblib
            tqdm, if progress_bar is True
    """
    import joblib

    if len(data_frame) == 0:
        return []

        
    if progress_bar:
        import tqdm
        ret_list = joblib.Parallel(n_jobs=num_procs)(joblib.delayed(func)(row[1], *args) 
            for row in tqdm.tqdm(data_frame.iterrows(), total=len(data_frame), 
                leave=True, file=sys.stdout))
    else:
        ret_list = joblib.Parallel(n_jobs=num_procs)(joblib.delayed(func)(row[1], *args) 
            for row in data_frame.iterrows())
    return ret_list

def apply_df_simple(data_frame, func, *args, progress_bar=False):
    """ This function applies func to all rows in data_frame, passing in arguments args. It
        collects the results as a list with the return value of func(row, *args) as each item
        in the list. It is not parallelized in any way.

        This function is preferable to pd.DataFrame.apply when func does not return something
        that is easily parsed by pandas. An example of such a return type is when func
        returns numpy arrays of varying length based on the values of the rows in data_frame.

        Args:
            data_frame (pandas.DataFrame): A data frame

            func (function pointer): The function to apply to each row in the data frame

            args (variable number of arguments): The other arguments to pass to func

        Returns:
            list: the values returned from func for each row (in the order specified by
                pd.DataFrame.iterrows

        Imports:
            joblib, indirectly

    """
    num_procs = 1
    return apply_parallel(data_frame, num_procs, func, *args, progress_bar=progress_bar) 


def apply_iter_simple(items, func, *args, progress_bar=False, total=None, num_groups=None):
    """ This function applies func to all items in the iterator items, passing in arguments args. It
        collects the results as a list with the return value of func(row, *args) as each item
        in the list. It is not parallelized in any way.

        This function is preferable to map or other functions when func does not return something
        that is easily parsed by pandas or the function has constant arguments. It is also 
        a convenience function for wrapping a progress bar around the function calls.

        Args:

            items (list-like): A list (or anything that can be iterated over in a for loop)

            func (function pointer): The function to apply to each group

            args (variable number of arguments): The other arguments to pass to func

            progress_bar (boolean) : whether to use a tqdm progress bar

            total (int) : the total number of items in the list. This only needs to be used if
                len(items) is not defined. It is only used for tqdm, so it is not necessary.

            num_groups (int) : if given, the number of groups into which the input is split. In
                case it is given, then each call to func will be passed a list of items.

        Returns:
            list: the values returned from func for each item (in the order specified by
                joblib.Parallel). If num_groups is given, then this is likely to be a list.

        Imports:
            joblib
            numpy
            tqdm, if progress_bar is True
    """


    num_procs = 1
    return apply_parallel_iter(items, num_procs, func, *args, progress_bar=progress_bar, 
        total=total, num_groups=num_groups)



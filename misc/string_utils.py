""" Utilities for working with strings
"""
import logging
logger = logging.getLogger(__name__)

import re
import typing


_TRUE_STRING = {'true', 'yes', 't', 'y', '1'}

def str2bool(s:str) -> bool:
    """ Convert `s` to a boolean value, if possible
    
    Parameters
    ----------
    s : string
        A string which may represent a boolean value
        
    Returns
    -------
    bool_s : boolean
        `True` if `s` is in `_TRUE_STRING`, and `False` otherwise
    """
    return (string.lower() in _TRUE_STRING)

def try_parse_int(s:str) -> typing.Optional[int]:
    """ Convert `s` to an integer, if possible
    
    Parameters
    ----------
    s : string
        A string which may represent an integer
        
    Returns
    -------
    int_s : int
        An integer
    --- OR ---
    None
        If `s` cannot be parsed into an `int`.
    """
    try:
        return int(s)
    except ValueError:
        return None

def try_parse_float(s:str) -> typing.Optional[float]:
    """ Convert `s` to a float, if possible
    
    Parameters
    ----------
    s : string
        A string which may represent a float
        
    Returns
    -------
    float_s : float
        A float
    --- OR ---
    None
        If `s` cannot be parsed into a `float`.
    """
    try:
        return float(s)
    except ValueError:
        return None

def bytes2human(n:int, format:str="%(value)i%(symbol)s") -> str:
    """ Convert `n` bytes to a human-readable format
    
    This code is adapted from: http://goo.gl/zeJZl
    
    Parameters
    ----------
    n : int
        The number of bytes
        
    format : string
        The format string
        
    Returns
    -------
    human_str : string
        A human-readable version of the number of bytes
    
    Examples
    --------
    
    >>> bytes2human(10000)
    '9K'
    >>> bytes2human(100001221)
    '95M'
    
    """
    symbols = ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i+1)*10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)

def human2bytes(s : str) -> int:
    """ Convert a human-readable byte string to an integer
    
    This code is adapted from: http://goo.gl/zeJZl
    
    Parameters
    ----------
    s : string
        The human-readable byte string
        
    Returns
    -------
    num_bytes : int
        The number of bytes
        
    Examples
    --------
    >>> human2bytes('1M')
    1048576
    >>> human2bytes('1G')
    1073741824
    """
    # first, check if s is already a number
    if is_int(s):
        return s

    symbols = ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    letter = s[-1:].strip().upper()
    num = s[:-1]
    assert num.isdigit() and letter in symbols
    num = float(num)
    prefix = {symbols[0]:1}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i+1)*10
    return int(num * prefix[letter])

def simple_fill(text:str, width:int=60) -> str:
    """ Split `text` into equal-sized chunks of length `width`
    
    This is a simplified version of `textwrap.fill`.

    The code is adapted from: http://stackoverflow.com/questions/11781261

    Parameters
    ----------
    text : string
        The text to split

    width : int
        The (exact) length of each line after splitting

    Returns
    -------
    split_str : string
        A single string with lines of length `width` (except possibly
        the last line)
    """
    return '\n'.join(text[i:i+width] 
                        for i in range(0, len(text), width))


def split(s:str, delimiters:typing.Iterable[str], maxsplit:int=0) -> typing.List[str]:
    """ Split `s` on any of the given `delimiters`
    
    This code is adapted from: http://stackoverflow.com/questions/4998629/

    Parameters
    ----------
    s : string
        The string to split
        
    delimiters : list of strings
        The strings to use as delimiters
        
    maxsplit : int
        The maximum number of splits (or 0 for no limit)

    Returns
    -------
    splits : list of strings
        the split strings
    """
    regex_pattern = '|'.join(map(re.escape, delimiters))
    return re.split(regex_pattern, string, maxsplit)
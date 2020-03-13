""" Utilities for working with strings
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import re
import tqdm

from typing import Iterable, List, Mapping, Optional, Sequence, Union
encoding_map_type = Mapping[str, np.ndarray]
np_array_or_list = Union[np.ndarray,List]

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
    return (s.lower() in _TRUE_STRING)

def try_parse_int(s:str) -> Optional[int]:
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

def try_parse_float(s:str) -> Optional[float]:
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
    s_i = try_parse_int(s)
    if s_i is not None:
        return s_i

    symbols = ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    letter = s[-1:].strip().upper()
    num = s[:-1]
    assert num.isdigit() and letter in symbols
    num = float(num)
    prefix = {symbols[0]:1}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i+1)*10
    return int(num * prefix[letter])

def encode_all_sequences(
        sequences:Iterable[str],
        encoding_map:encoding_map_type,
        maxlen:Optional[int]=None,
        align:str="start",
        pad_value:str='J',
        same_length:bool=False,
        flatten:bool=False,
        return_as_numpy:bool=True,
        swap_axes:bool=False,
        progress_bar:bool=True) -> np_array_or_list:
    """ Extract the amino acid feature vectors for each peptide sequence

    See `get_peptide_aa_features` for more details.

    Parameters
    ----------
    sequences : typing.Iterable[str]
        The sequences

    encoding_map : typing.Mapping[str, numpy.ndarray]
        The features for each character

    maxlen : typing.Optional[int]

    align : str

    pad_value : str

    same_length : bool

    flatten : bool
        Whether to (attempt to) convert the features of each peptide into
        a single long vector (`True`) or leave as a (presumably) 2d
        position-feature vector.

    return_as_numpy : bool
        Whether to return as a 2d or 3d numpy array (`True`) or a list
        containing 1d or 2d numpy arrays. (The dimensionality depends
        upon `flatten`.)

    swap_axes : bool
        If the values are returned as a numpy tensor, swap axes 1 and 2.

        N.B. This flag is only compatible with `return_as_numpy=True` and
        `flatten=False`.

    progress_bar : bool
        Whether to show a progress bar for collecting the features.

    Returns
    -------
    all_encoded_peptides : typing.Union[numpy.ndarray, typing.List]
        The resulting features. See the `flatten` and `return_as_numpy`
        parameters for the expected output.
    """
    
    if not same_length:
        sequences = pad_trim_sequences(
            sequences,
            maxlen=maxlen,
            align=align,
            pad_value=pad_value
        )
    
    it = sequences
    if progress_bar:
        it = tqdm.tqdm(it)

    all_encoded_sequences = [
        encode_sequence(
            sequence,
            encoding_map,
            flatten=flatten
        ) for sequence in it
    ]

    if return_as_numpy:
        all_encoded_sequences = np.array(all_encoded_sequences)

        #TODO: make the meaning of this more clear
        # Elsewhere, this was referred to as setting the "channel" to "first".
        if swap_axes:
            all_encoded_sequences = np.swapaxes(all_encoded_sequences, 1, 2)

    return all_encoded_sequences

def encode_sequence(
        sequence:str,
        encoding_map:encoding_map_type,
        flatten:bool=False) -> np.ndarray:
    """ Extract the amino acid properties of the given sequence

    This function is designed with the idea of mapping from a sequence to
    numeric features (such as chemical properties or BLOSUM features for amino
    acid sequences). It may fail if other features are included in
    `encoding_map`.

    Parameters
    ----------
    sequence : str
        The sequence

    encoding_map : typing.Mapping[str, numpy.ndarray]
        A mapping from each character to a set of features. Presumably, the
        features are numpy-like arrays, though they need not be.

    flatten : bool
        Whether to flatten the encoded sequence into a single, 1d array or
        leave them as-is.

    Returns
    -------
    encoded_sequence : numpy.ndarray
        A 1d or 2d np.array, depending on `flatten`. By default
        (`flatten=False`), this is a 1d array of objects, in which the
        outer dimension indexes the position in the epitope. If `flatten`
        is `True`, then the function attempts to reshape the features
        into a single long feature vector. This will likely fail if the
        `encoding_map` values are not numpy-like arrays.
    """

    try:
        encoded_sequence = np.array([
            encoding_map.get(a) for a in sequence
        ])
    except KeyError as ke:
        msg = "Found invalid character. Sequence: {}".format(sequence)
        logger.warning(msg)
        return None

    if flatten:
        encoded_sequence = np.hstack(encoded_sequence)

    return encoded_sequence

def pad_sequence(
        seq:str,
        max_seq_len:int,
        pad_value:str="J",
        align:str="end") -> str:
    """ Pad `seq` to `max_seq_len` with `value` based on the `align` strategy

    If `seq` is already of length `max_seq_len` *or longer* it will not be
    changed.

    Parameters
    ----------
    seq : str
        The character sequence

    max_seq_len : int
        The maximum length for a sequence

    pad_value : str
        The value for padding. This should be a single character

    align : str
        The strategy for padding the string. Valid options are `start`, `end`,
        and `center`

    Returns
    -------
    padded_seq : str
        The padded string. In case `seq` was already long enough or longer, it
        will not be changed. So `padded_seq` could be longer than `max_seq_len`.
    """
    seq_len = len(seq)
    assert max_seq_len >= seq_len
    if align == "end":
        n_left = max_seq_len - seq_len
        n_right = 0
    elif align == "start":
        n_right = max_seq_len - seq_len
        n_left = 0
    elif align == "center":
        n_left = (max_seq_len - seq_len) // 2 + (max_seq_len - seq_len) % 2
        n_right = (max_seq_len - seq_len) // 2
    else:
        raise ValueError("align can be of: end, start or center")

    # normalize for the length
    n_left = n_left // len(pad_value)
    n_right = n_right // len(pad_value)

    return pad_value * n_left + seq + pad_value * n_right

def pad_trim_sequences(
        seq_vec:Sequence[str],
        pad_value:str='J',
        maxlen:Optional[int]=None,
        align:str="start") -> List[str]:
    """Pad and/or trim a list of sequences to have common length
    
    The procedure is as follows:
        1. Pad the sequence with `pad_value`
        2. Trim the sequence

    Parameters
    ----------
    seq_vec : typing.Sequence[str]
        List of sequences that can have various lengths

    pad_value : str
        Neutral element with which to pad the sequence. This should be a single
        character.

    maxlen: typing.Optional[int]
        Length of padded/trimmed sequences. If `None`, `maxlen` is set to the
        longest sequence length.

    align : str
        To which end to align the sequences when triming/padding. Valid options
        are `start`, `end`, `center`

    Returns
    -------
    padded_sequences : typing.List[str]
        The padded and/or trimmed sequences
    """    
    if isinstance(seq_vec, str):
        seq_vec = [seq_vec]
    assert isinstance(seq_vec, list) or \
        isinstance(seq_vec, pd.Series) or \
        isinstance(seq_vec, np.ndarray)
    assert isinstance(seq_vec[0], str)
    assert len(pad_value) == 1
    
    max_seq_len = max([len(seq) for seq in seq_vec])
    if maxlen is None:
        maxlen = max_seq_len
    else:
        maxlen = int(maxlen)

    if max_seq_len < maxlen:
        msg = ("Maximum sequence length (%s) is less than maxlen (%s)" % (max_seq_len, maxlen))
        logger.warning(msg)
        max_seq_len = maxlen
        
    # pad and trim
    padded_seq_vec = [
        pad_sequence(
            seq, max(max_seq_len, maxlen), pad_value=pad_value, align=align
        ) for seq in seq_vec
    ]

    padded_seq_vec = [
        trim_sequence(seq, maxlen, align=align)
            for seq in padded_seq_vec
    ]
    
    return padded_seq_vec

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


def split(s:str, delimiters:Iterable[str], maxsplit:int=0) -> List[str]:
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
    return re.split(regex_pattern, s, maxsplit)

def trim_sequence(
        seq:str,
        maxlen:int,
        align:str="end") -> str:
    """ Trim `seq` to at most `maxlen` characters using `align` strategy

    Parameters
    ----------
    seq : str
        The (amino acid) sequence

    maxlen : int
        The maximum length

    align : str
        The strategy for trimming the string. Valid options are `start`, `end`,
        and `center`

    Returns
    -------
    trimmed_seq : str
        The trimmed string. In case `seq` was already an appropriate length, it
        will not be changed. So `trimmed_seq` could be shorter than `maxlen`. 
    """
    seq_len = len(seq)

    assert maxlen <= seq_len
    if align == "end":
        return seq[-maxlen:]
    elif align == "start":
        return seq[0:maxlen]
    elif align == "center":
        dl = seq_len - maxlen
        n_left = dl // 2 + dl % 2
        n_right = seq_len - dl // 2
        return seq[n_left:n_right]
    else:
        raise ValueError("align can be of: end, start or center")
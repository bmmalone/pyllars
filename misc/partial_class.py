import functools

def partialclass(cls, *args, **kwds):
    """ Create a subclass which has constructor defaults
    
    The motivation for this function is for cases where we need to
    subclass from an external library, but our subclass needs parameters
    to the constructor. Further, the external library makes direct
    calls to the constructor without passing any of our extra parameters.
    
    See this SO thread: https://stackoverflow.com/questions/38911146
    
    Parameters
    ----------
    cls: class
        The base class
        
    args, kwargs: positional and key=value parameters
        The values to pass to the constructor of the base class
    """

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls

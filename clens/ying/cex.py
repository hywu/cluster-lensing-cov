#Last-modified: 01 Mar 2013 11:40:54 PM


""" Exception Handlers.
"""

class Error(Exception):
    """ Base class for cosmopy exceptions."""
    pass

class UnknownParameterError(Error):
    """
    Exception raised when a parameter is not in the default dict.
    """
    pass

class NonFlatUniverseError(Error):
    """
    Exception raised when the given cosmology is not flat.
    """
    pass

class NonCallableObject(Error):
    """
    Exception raised when the object does not have the callable attr.
    """
    pass

class ExtrapolationRequired(Error):
    """
    Exception raised when extrapolation rather than interpolation is 
    needed.
    """
    pass

class CosmologyUnapplicable(Error):
    """
    Given cosmology does not apply to the operation.
    """
    pass

class ParameterOutsideDefaultRange(Error):
    """
    Given parameter is outside the interpolation range.
    """
    pass

class ConditionNotReached(Error):
    """
    Condition fails to reach in a loop.
    """
    pass

class ParametersNotPairSet(Error):
    """
    Two parameters need to be set accordingly.
    """
    pass

"""Chonky document-processing application and reusable parser interfaces.

Purpose:
    Exposes parser classes lazily so importing the package does not initialize NLP models.
"""
from importlib import import_module

__version__ = '0.1.0'
__all__ = ['Processor', 'TextParser', 'WordParser', 'PdfParser', 'Error']


def __getattr__( name: str ) -> object:
    """Resolve the existing public parser and error interfaces.

    Args:
        name (str): Public interface name.

    Returns:
        object: Requested class.

    Raises:
        AttributeError: The requested interface does not exist.
    """
    if name in __all__:
        module = import_module( '.boogr' if name == 'Error' else '.processors', __name__ )
        return getattr( module, name )
    raise AttributeError( f'Module {__name__} has no attribute {name}' )

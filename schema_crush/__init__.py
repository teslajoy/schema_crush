"""schema_crush - multi-agent schema mapping system."""

import importlib.resources
from pathlib import Path

__version__ = "0.1.0"


def get_package_path(relative_path: str) -> Path:
    """
    get absolute path to file in package.

    args:
        relative_path: path relative to schema_crush package (ex. 'data/examples/case.csv')

    returns:
        absolute path object

    example:
        >>> case_path = get_package_path('data/examples/case.csv')
        >>> df = pd.read_csv(case_path)
    """
    return Path(importlib.resources.files('schema_crush') / relative_path)
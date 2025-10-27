"""Base loader interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd


class BaseLoader(ABC):
    """base class for data loaders."""
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def load(self, source: str, **kwargs) -> Dict[str, Any]:
        """load data from source and return structured information."""
        pass
    
    @abstractmethod
    def get_schema_info(self, loaded_data: Dict[str, Any]) -> Dict[str, Any]:
        """extract schema information from loaded data."""
        pass
    
    def _analyze_dataframe(self, df: pd.DataFrame, entity_name: str) -> Dict[str, Any]:
        """common dataframe analysis."""
        return {
            "entity": entity_name,
            "row_count": len(df),
            "columns": list(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_values": {
                col: df[col].dropna().head(3).tolist() 
                for col in df.columns
            },
            "null_counts": df.isnull().sum().to_dict(),
            "unique_counts": {
                col: df[col].nunique() for col in df.columns
            }
        }
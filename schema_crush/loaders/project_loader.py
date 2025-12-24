"""loader for GDC/HTAN project data with raw source and FHIR ground truth."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from .base import BaseLoader


class ProjectLoader(BaseLoader):
    """
    load GDC/HTAN project data with both raw source and transformed FHIR data.

    structure:
    projects/
    ├── HTAN/OHSU/
    │   ├── raw/
    │   │   ├── cases/table_data.tsv
    │   │   ├── biospecimens/table_data.tsv
    │   │   └── files/table_data.tsv
    │   └── META/
    │       ├── Patient.ndjson (FHIR ground truth)
    │       ├── Specimen.ndjson
    │       └── ...
    └── TCGA-BRCA/
        ├── cases.ndjson (raw)
        ├── files.ndjson (raw)
        └── META/
            ├── Patient.ndjson (FHIR ground truth)
            └── ...
    """

    def __init__(self, name: str = "project_loader"):
        super().__init__(name)
        self.project_data = {}

    def load_htan_project(self, project_path: Path) -> Dict[str, Any]:
        """
        load HTAN project data.

        args:
            project_path: path to HTAN project (e.g., projects/HTAN/OHSU)

        returns:
            dict with raw source data and FHIR ground truth
        """
        project_path = Path(project_path)

        result = {
            "project_type": "HTAN",
            "project_name": project_path.name,
            "raw_data": {},
            "fhir_ground_truth": {},
            "metadata": {
                "atlas": project_path.parent.name
            }
        }

        # load raw data (TSV files)
        raw_path = project_path / "raw"
        if raw_path.exists():
            for entity_dir in ["cases", "biospecimens", "files"]:
                entity_path = raw_path / entity_dir / "table_data.tsv"
                if entity_path.exists():
                    try:
                        df = pd.read_csv(entity_path, sep='\t')
                        result["raw_data"][entity_dir] = {
                            "dataframe": df,
                            "path": str(entity_path),
                            "row_count": len(df),
                            "columns": list(df.columns),
                            "column_count": len(df.columns)
                        }
                        print(f"  loaded {entity_dir}: {len(df)} rows, {len(df.columns)} columns")
                    except Exception as e:
                        print(f"  warning: failed to load {entity_path}: {e}")

        # load FHIR ground truth (NDJSON files)
        meta_path = project_path / "META"
        if meta_path.exists():
            for fhir_file in meta_path.glob("*.ndjson"):
                resource_type = fhir_file.stem  # e.g., "Patient", "Specimen"
                try:
                    resources = []
                    with open(fhir_file) as f:
                        for line in f:
                            if line.strip():
                                resources.append(json.loads(line))

                    result["fhir_ground_truth"][resource_type] = {
                        "resources": resources,
                        "path": str(fhir_file),
                        "resource_count": len(resources),
                        "sample": resources[0] if resources else None
                    }
                    print(f"  loaded FHIR {resource_type}: {len(resources)} resources")
                except Exception as e:
                    print(f"  warning: failed to load {fhir_file}: {e}")

        return result

    def load_tcga_project(self, project_path: Path) -> Dict[str, Any]:
        """
        load TCGA-GDC project data.

        args:
            project_path: path to TCGA project (e.g., projects/TCGA-BRCA)

        returns:
            dict with raw source data and FHIR ground truth
        """
        project_path = Path(project_path)

        result = {
            "project_type": "TCGA",
            "project_name": project_path.name,
            "raw_data": {},
            "fhir_ground_truth": {},
            "metadata": {}
        }

        # load raw NDJSON data
        for entity_file in ["cases.ndjson", "files.ndjson"]:
            entity_path = project_path / entity_file
            if entity_path.exists():
                entity_name = entity_file.replace(".ndjson", "")
                try:
                    records = []
                    with open(entity_path) as f:
                        for line in f:
                            if line.strip():
                                records.append(json.loads(line))

                    # convert to dataframe for easier analysis
                    df = pd.json_normalize(records)

                    result["raw_data"][entity_name] = {
                        "dataframe": df,
                        "records": records,
                        "path": str(entity_path),
                        "row_count": len(records),
                        "columns": list(df.columns),
                        "column_count": len(df.columns)
                    }
                    print(f"  loaded {entity_name}: {len(records)} records, {len(df.columns)} fields")
                except Exception as e:
                    print(f"  warning: failed to load {entity_path}: {e}")

        # load FHIR ground truth
        meta_path = project_path / "META"
        if meta_path.exists():
            for fhir_file in meta_path.glob("*.ndjson"):
                resource_type = fhir_file.stem
                try:
                    resources = []
                    with open(fhir_file) as f:
                        for line in f:
                            if line.strip():
                                resources.append(json.loads(line))

                    result["fhir_ground_truth"][resource_type] = {
                        "resources": resources,
                        "path": str(fhir_file),
                        "resource_count": len(resources),
                        "sample": resources[0] if resources else None
                    }
                    print(f"  loaded FHIR {resource_type}: {len(resources)} resources")
                except Exception as e:
                    print(f"  warning: failed to load {fhir_file}: {e}")

        return result

    def load(self, source: str, **kwargs) -> Dict[str, Any]:
        """
        load project data (auto-detects HTAN vs TCGA format).

        args:
            source: path to project directory

        returns:
            project data with raw and FHIR ground truth
        """
        project_path = Path(source)

        if not project_path.exists():
            raise FileNotFoundError(f"project path not found: {project_path}")

        print(f"\nloading project: {project_path}")
        print("-" * 70)

        # detect project type based on structure
        if (project_path / "raw").exists():
            # HTAN format
            return self.load_htan_project(project_path)
        elif (project_path / "cases.ndjson").exists() or (project_path / "files.ndjson").exists():
            # TCGA format
            return self.load_tcga_project(project_path)
        else:
            raise ValueError(f"unknown project structure at {project_path}")

    def extract_ground_truth_mappings(self, project_data: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
        """
        extract field mappings from raw -> FHIR ground truth.

        this analyzes the FHIR resources to infer which raw fields
        mapped to which FHIR paths.

        args:
            project_data: loaded project data

        returns:
            dict mapping entity -> [(source_field, fhir_path), ...]
        """
        mappings = {}

        # TODO: implement intelligent mapping extraction
        # for now, just return available fields

        for entity_name, entity_data in project_data["raw_data"].items():
            mappings[entity_name] = []
            source_fields = entity_data["columns"]

            # placeholder: would need sophisticated logic to match
            # raw field values to FHIR resource field values
            mappings[entity_name] = [(f, "Unknown") for f in source_fields[:5]]

        return mappings

    def get_schema_info(self, loaded_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        extract schema information from loaded data (required by BaseLoader).

        args:
            loaded_data: loaded project data

        returns:
            schema information
        """
        return self.get_schema_summary(loaded_data)

    def get_schema_summary(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        get summary of project schema.

        args:
            project_data: loaded project data

        returns:
            schema summary with entity counts, field counts, etc.
        """
        summary = {
            "project_name": project_data["project_name"],
            "project_type": project_data["project_type"],
            "raw_entities": {},
            "fhir_resources": {},
            "total_raw_records": 0,
            "total_fhir_resources": 0
        }

        # summarize raw data
        for entity_name, entity_data in project_data["raw_data"].items():
            summary["raw_entities"][entity_name] = {
                "row_count": entity_data["row_count"],
                "column_count": entity_data["column_count"],
                "columns": entity_data["columns"][:10]  # first 10 columns
            }
            summary["total_raw_records"] += entity_data["row_count"]

        # summarize FHIR ground truth
        for resource_type, resource_data in project_data["fhir_ground_truth"].items():
            summary["fhir_resources"][resource_type] = {
                "resource_count": resource_data["resource_count"]
            }
            summary["total_fhir_resources"] += resource_data["resource_count"]

        return summary
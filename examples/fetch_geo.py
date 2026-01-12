#!/usr/bin/env python
"""fetch GEO data using GEOparse.

downloads sample metadata and clinical characteristics from NCBI GEO.

usage:
    python examples/fetch_geo.py                    # download all pancreatic datasets
    python examples/fetch_geo.py GSE71729           # download specific dataset
    python examples/fetch_geo.py GSE71729 GSE62452  # download multiple datasets
    python examples/fetch_geo.py --list             # list available datasets

output:
    data/geo/GSE71729_metadata.csv   - full sample metadata
    data/geo/GSE71729_clinical.csv   - parsed clinical characteristics

references:
    https://geoparse.readthedocs.io/
    https://github.com/guma44/GEOparse
"""

import argparse
import sys
from pathlib import Path

# output directory at project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "geo"


# curated pancreatic cancer datasets with clinical metadata
PANCREATIC_DATASETS = {
    'GSE62452': 'PDAC with survival data (69 samples)',
    'GSE28735': 'PDAC tumor vs normal (90 samples)',
    'GSE15471': 'PDAC tumor vs normal (78 samples)',
    'GSE21501': 'PDAC with survival data (132 samples)',
    'GSE57495': 'PDAC subtype classification (63 samples)',
    'GSE71729': 'PDAC large cohort (145 samples) - TCGA-like',
}


def fetch_geo_series(accession: str, data_dir: Path) -> dict:
    """fetch a GEO series with expression data and metadata.

    args:
        accession: GEO series ID (e.g., GSE62452)
        data_dir: output directory

    returns:
        dict with 'metadata', 'expression', 'clinical' dataframes
    """
    import GEOparse
    import pandas as pd

    print(f"\n{'='*60}")
    print(f"Fetching {accession}")
    print(f"{'='*60}")

    # ensure output dir exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # download GSE (will cache in destdir)
    gse = GEOparse.get_GEO(geo=accession, destdir=str(data_dir))

    # extract sample metadata
    print("\n--- Sample Metadata ---")
    metadata_rows = []
    for gsm_name, gsm in gse.gsms.items():
        row = {'sample_id': gsm_name}
        row.update(gsm.metadata)
        # flatten lists to strings
        for k, v in row.items():
            if isinstance(v, list):
                row[k] = '; '.join(str(x) for x in v)
        metadata_rows.append(row)

    metadata_df = pd.DataFrame(metadata_rows)
    print(f"samples: {len(metadata_df)}")
    print(f"metadata columns: {len(metadata_df.columns)}")

    # extract expression data (if available)
    print("\n--- Expression Data ---")
    try:
        expression_df = gse.pivot_samples('VALUE')
        print(f"expression matrix: {expression_df.shape[0]} genes x {expression_df.shape[1]} samples")
    except Exception as e:
        print(f"no expression data: {e}")
        expression_df = None

    # platform info
    print("\n--- Platform ---")
    for gpl_name, gpl in gse.gpls.items():
        print(f"platform: {gpl_name}")

    # save metadata
    metadata_path = data_dir / f"{accession}_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\nsaved: {metadata_path}")

    # save expression
    if isinstance(expression_df, pd.DataFrame):
        expr_path = data_dir / f"{accession}_expression.csv"
        expression_df.to_csv(expr_path)
        print(f"saved: {expr_path}")

    # extract and save clinical characteristics
    clinical_df = extract_clinical_characteristics(metadata_df)
    clinical_path = data_dir / f"{accession}_clinical.csv"
    clinical_df.to_csv(clinical_path, index=False)
    print(f"saved: {clinical_path}")
    print(f"clinical columns: {list(clinical_df.columns)}")

    return {
        'metadata': metadata_df,
        'expression': expression_df,
        'clinical': clinical_df
    }


def extract_clinical_characteristics(metadata_df) -> 'pd.DataFrame':
    """parse clinical characteristics from GEO metadata.

    GEO stores clinical data in characteristics_ch1 field like:
    "tissue: Pancreatic tumor; grading: G2; Stage: IIA"
    """
    import pandas as pd

    clinical_rows = []
    for _, row in metadata_df.iterrows():
        clinical = {'sample_id': row.get('sample_id', '')}

        for col in metadata_df.columns:
            if 'characteristics' in col.lower():
                chars = row.get(col, '')
                if pd.notna(chars) and chars:
                    for char in str(chars).split(';'):
                        if ':' in char:
                            key, val = char.split(':', 1)
                            key = key.strip().lower().replace(' ', '_')
                            val = val.strip()
                            clinical[key] = val

        clinical_rows.append(clinical)

    return pd.DataFrame(clinical_rows)


def main():
    parser = argparse.ArgumentParser(
        description='Fetch GEO datasets for FHIR mapping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('datasets', nargs='*', help='GEO accession IDs (e.g., GSE71729)')
    parser.add_argument('--list', action='store_true', help='List available curated datasets')
    parser.add_argument('--output', '-o', type=Path, default=DATA_DIR, help='Output directory')
    parser.add_argument('--all', action='store_true', help='Download all pancreatic datasets')
    args = parser.parse_args()

    if args.list:
        print("Curated pancreatic cancer datasets:\n")
        for acc, desc in PANCREATIC_DATASETS.items():
            print(f"  {acc}: {desc}")
        print(f"\nUsage: python {sys.argv[0]} GSE71729")
        return

    # determine which datasets to fetch
    if args.all:
        datasets = list(PANCREATIC_DATASETS.keys())
    elif args.datasets:
        datasets = args.datasets
    else:
        # default: fetch all pancreatic datasets
        print("No datasets specified. Use --list to see options or --all to fetch all.")
        print("Fetching GSE71729 as example...\n")
        datasets = ['GSE71729']

    # check dependencies
    try:
        import GEOparse
        import pandas
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install GEOparse pandas")
        sys.exit(1)

    # fetch each dataset
    results = {}
    for accession in datasets:
        try:
            result = fetch_geo_series(accession, args.output)
            results[accession] = result
        except Exception as e:
            print(f"ERROR fetching {accession}: {e}")

    # summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Output directory: {args.output}")
    for acc, res in results.items():
        n_samples = len(res['metadata'])
        n_clinical = len(res['clinical'].columns) - 1  # exclude sample_id
        print(f"  {acc}: {n_samples} samples, {n_clinical} clinical fields")

    print(f"\nTo map to FHIR:")
    if results:
        first_acc = list(results.keys())[0]
        print(f"  python examples/map_csv.py {args.output}/{first_acc}_clinical.csv -e sample")


if __name__ == '__main__':
    main()
"""explore the flat mapping database tables.

run with: python -i examples/explore_flat_db.py
or in ipython: %run examples/explore_flat_db.py
"""

from schema_crush.mappings.flat_loader import load_flat_mappings
import pandas as pd

# load the database (use_cache=False to always read fresh from disk)
print("loading flat mapping database...")
db = load_flat_mappings(use_cache=False)

# convert to dataframes for easy exploration
print("converting to dataframes...")

sources_df = pd.DataFrame([
    {
        "id": s.id,
        "source": s.source,
        "schema": s.source_schema,
        "tier": s.tier.value if hasattr(s.tier, 'value') else str(s.tier),
        "context": s.source_context
    }
    for s in db.sources.values()
])

destinations_df = pd.DataFrame([
    {
        "source_id": d.source_id,
        "destination": d.destination,
        "dest_system": d.dest_system,
        "dest_code": d.dest_code,
        "dest_display": d.dest_display,
        "subject_ref": d.subject_ref,
        "focus_ref": d.focus_ref,
        "specimen_ref": d.specimen_ref
    }
    for d in db.destinations
])

content_values_df = pd.DataFrame([
    {
        "id": i,
        "source_value": cv.source_value,
        "source_category": cv.source_category,
        "code": cv.code,
        "system": cv.system,
        "display": cv.display
    }
    for i, cv in enumerate(db.content_values)
])

# summary
print()
print("=== database loaded ===")
print(f"sources_df:        {len(sources_df)} rows")
print(f"destinations_df:   {len(destinations_df)} rows")
print(f"content_values_df: {len(content_values_df)} rows")
print()
print("=== available dataframes ===")
print("  sources_df        - source terms (id, source, schema, tier, context)")
print("  destinations_df   - fhir destinations (source_id, destination, dest_code, refs)")
print("  content_values_df - coded values (source_value, code, system)")
print()
print("=== example queries ===")
print("  sources_df.head(10)")
print("  sources_df[sources_df['schema'] == 'gdc']")
print("  destinations_df[destinations_df['destination'].str.contains('Specimen')]")
print("  content_values_df[content_values_df['source_value'].str.contains('Stomach')]")
print()
print("=== quick lookups ===")
print("  db.lookup('sample_id')  # o(1) source->destination lookup")
print("  db.lookup_content('Adenocarcinoma', 'histology')")
print()

# show samples
print("=== sample sources ===")
print(sources_df.head(10).to_string(index=False))
print()
print("=== sample destinations ===")
print(destinations_df.head(10).to_string(index=False))
print()
print("=== sample content values ===")
print(content_values_df.head(10).to_string(index=False))

# per-source breakdown
print()
print("=" * 60)
print("=== per-source breakdown ===")
print("=" * 60)

schemas = sources_df['schema'].unique()
for schema in sorted(schemas):
    schema_sources = sources_df[sources_df['schema'] == schema]
    schema_source_ids = set(schema_sources['id'])
    schema_dests = destinations_df[destinations_df['source_id'].isin(schema_source_ids)]

    print(f"\n--- {schema.upper()} ---")
    print(f"  sources: {len(schema_sources)}")
    print(f"  destinations: {len(schema_dests)}")

    # specimen mappings
    specimen_dests = schema_dests[schema_dests['destination'].str.contains('Specimen', case=False, na=False)]
    if len(specimen_dests) > 0:
        print(f"  specimen mappings ({len(specimen_dests)}):")
        for _, row in specimen_dests.head(5).iterrows():
            src = sources_df[sources_df['id'] == row['source_id']]['source'].values
            src_name = src[0] if len(src) > 0 else row['source_id']
            print(f"    {src_name} -> {row['destination']}")

    # patient mappings
    patient_dests = schema_dests[schema_dests['destination'].str.contains('Patient', case=False, na=False)]
    if len(patient_dests) > 0:
        print(f"  patient mappings ({len(patient_dests)}):")
        for _, row in patient_dests.head(5).iterrows():
            src = sources_df[sources_df['id'] == row['source_id']]['source'].values
            src_name = src[0] if len(src) > 0 else row['source_id']
            print(f"    {src_name} -> {row['destination']}")

    # condition/diagnosis mappings
    condition_dests = schema_dests[schema_dests['destination'].str.contains('Condition', case=False, na=False)]
    if len(condition_dests) > 0:
        print(f"  condition mappings ({len(condition_dests)}):")
        for _, row in condition_dests.head(3).iterrows():
            src = sources_df[sources_df['id'] == row['source_id']]['source'].values
            src_name = src[0] if len(src) > 0 else row['source_id']
            print(f"    {src_name} -> {row['destination']}")

print()
print("=" * 60)
print("=== content values by category ===")
print("=" * 60)
if 'source_category' in content_values_df.columns:
    for cat in content_values_df['source_category'].unique()[:10]:
        cat_values = content_values_df[content_values_df['source_category'] == cat]
        print(f"\n--- {cat} ({len(cat_values)} values) ---")
        for _, row in cat_values.head(5).iterrows():
            print(f"  \"{row['source_value'][:40]}\" -> {row['code']} ({row['system'][:30]}...)")
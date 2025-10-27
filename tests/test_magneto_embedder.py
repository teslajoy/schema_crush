"""test magneto embedder with real data."""

import pytest
import importlib.resources
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from schema_crush.loaders.csv_loader import CSVLoader
from schema_crush.tools.embeddings.magneto_embedder import MagnetoEmbedder


@pytest.fixture
def magneto_embedder():
    """create magneto embedder instance."""
    return MagnetoEmbedder()


@pytest.fixture
def loaded_data():
    """load example csv data."""
    loader = CSVLoader()

    # use importlib.resources for package-relative paths
    base_path = Path(importlib.resources.files('schema_crush'))

    sources = {
        'case': str(base_path / 'data' / 'examples' / 'case.csv'),
        'biospecimen': str(base_path / 'data' / 'examples' / 'biospecimen.csv')
    }
    return loader.load(sources)


def test_embedder_initialization(magneto_embedder):
    """test that magneto embedder initializes correctly."""
    assert magneto_embedder.model is not None
    assert magneto_embedder.model_name == "mpnet"


def test_single_embedding(magneto_embedder):
    """test generating single embedding."""
    embedding = magneto_embedder.embed(["participant_id"])
    assert embedding.shape[0] == 1
    assert embedding.shape[1] > 0  # has embedding dimension


def test_batch_embedding(magneto_embedder):
    """test generating multiple embeddings."""
    texts = ["participant_id", "primary_diagnosis", "tissue_type"]
    embeddings = magneto_embedder.embed(texts)
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] > 0


def test_similarity(magneto_embedder):
    """test similarity calculation between two terms."""
    sim = magneto_embedder.similarity("participant", "patient")
    assert 0.0 <= sim <= 1.0
    assert sim > 0.3  # these should be somewhat similar


def test_batch_similarity(magneto_embedder):
    """test similarity matrix calculation."""
    sources = ["participant_id", "specimen_id"]
    targets = ["Patient.identifier", "Specimen.identifier"]

    sim_matrix = magneto_embedder.batch_similarity(sources, targets)
    assert sim_matrix.shape == (2, 2)

    # participant_id should match Patient.identifier better
    assert sim_matrix[0][0] > sim_matrix[0][1]

    # specimen_id should match Specimen.identifier better
    assert sim_matrix[1][1] > sim_matrix[1][0]


def test_with_loaded_data(magneto_embedder, loaded_data):
    """test magneto embedder with actual loaded csv data."""
    schema_loader = CSVLoader()
    schema_info = schema_loader.get_schema_info(loaded_data)

    # get case entity fields
    case_fields = list(schema_info['entities']['case']['fields'].keys())

    # test embedding all case fields
    embeddings = magneto_embedder.embed(case_fields)
    assert embeddings.shape[0] == len(case_fields)

    # test similarity with fhir targets
    fhir_targets = [
        "Patient.identifier",
        "Patient.birthDate",
        "Condition.code",
        "Patient.gender"
    ]

    sim_matrix = magneto_embedder.batch_similarity(case_fields, fhir_targets)
    assert sim_matrix.shape == (len(case_fields), len(fhir_targets))

    # print top matches for inspection
    print("\ntop field matches (name only):")
    for i, field in enumerate(case_fields):
        best_idx = sim_matrix[i].argmax()
        best_target = fhir_targets[best_idx]
        best_score = sim_matrix[i][best_idx]
        print(f"  {field} -> {best_target}: {best_score:.4f}")


def test_three_tier_embeddings(magneto_embedder, loaded_data):
    """test 3-tier embedding methods."""
    schema_loader = CSVLoader()
    schema_info = schema_loader.get_schema_info(loaded_data)

    # get case entity with full column vectors
    case_entity = loaded_data['case']
    case_df = case_entity['dataframe']

    print("\n=== tier 1: entity matching ===")
    # tier 1: entity embedding
    entity_emb = magneto_embedder.embed_entity('case')
    patient_emb = magneto_embedder.embed(['Patient'])
    entity_sim = cosine_similarity(entity_emb, patient_emb)[0][0]
    print(f"case -> Patient: {entity_sim:.4f}")
    assert entity_emb.shape[0] == 1

    print("\n=== tier 2: field matching ===")
    # tier 2: field embedding (name only)
    field_emb = magneto_embedder.embed_field('primary_diagnosis')
    condition_emb = magneto_embedder.embed(['Condition.code'])
    field_sim = cosine_similarity(field_emb, condition_emb)[0][0]
    print(f"primary_diagnosis -> Condition.code: {field_sim:.4f}")
    assert field_emb.shape[0] == 1

    print("\n=== tier 3: content matching ===")
    # tier 3: content embedding (name + values using magneto's encoder)
    diagnosis_values = case_df['primary_diagnosis'].tolist()
    content_emb = magneto_embedder.embed_content(
        'primary_diagnosis',
        diagnosis_values,
        dataframe=case_df
    )
    content_sim = cosine_similarity(content_emb, condition_emb)[0][0]
    print(f"primary_diagnosis (with values) -> Condition.code: {content_sim:.4f}")
    assert content_emb.shape[0] == 1

    print("\n=== comparison ===")
    print(f"tier 2 (field name): {field_sim:.4f}")
    print(f"tier 3 (content): {content_sim:.4f}")
    print(f"difference: {abs(field_sim - content_sim):.4f}")

    # test with participant_id (identifier field)
    print("\n=== identifier field test ===")
    participant_values = case_df['participant_id'].tolist()
    id_field_emb = magneto_embedder.embed_field('participant_id')
    id_content_emb = magneto_embedder.embed_content(
        'participant_id',
        participant_values,
        dataframe=case_df
    )
    patient_id_emb = magneto_embedder.embed(['Patient.identifier'])

    id_field_sim = cosine_similarity(id_field_emb, patient_id_emb)[0][0]
    id_content_sim = cosine_similarity(id_content_emb, patient_id_emb)[0][0]

    print(f"participant_id (field) -> Patient.identifier: {id_field_sim:.4f}")
    print(f"participant_id (content) -> Patient.identifier: {id_content_sim:.4f}")


def test_schema_matching(magneto_embedder, loaded_data):
    """test full schema matching between case and fhir-like target."""
    case_df = loaded_data['case']['dataframe']

    # create simplified fhir-like target
    fhir_df = pd.DataFrame({
        'Patient.identifier': ['P001', 'P002'],
        'Patient.birthDate': ['1980-01-01', '1990-05-15'],
        'Condition.code': ['C50.9', 'C61'],
        'Patient.gender': ['female', 'male']
    })

    # get matches
    matches = magneto_embedder.match_schemas(case_df, fhir_df)

    print("\n=== schema matching results ===")
    for (src, tgt), score in sorted(matches.items(), key=lambda x: x[1], reverse=True):
        src_col = src[1]
        tgt_col = tgt[1]
        print(f"{src_col} -> {tgt_col}: {score:.4f}")

    assert len(matches) > 0
    assert all(0.0 <= score <= 1.0 for score in matches.values())
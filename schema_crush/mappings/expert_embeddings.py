"""generate embeddings from RuleMatcher expertise for use in BioBERT/Magneto."""

from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer


def build_expertise_map(db) -> dict[str, list[str]]:
    """build source -> [destinations] from ALL tiers (entity, field, content)."""
    expertise = {}

    # entity + field tiers from sources table
    for src in db.sources.values():
        key = src.source.lower()
        if key not in expertise:
            expertise[key] = []
        for dest in db._dest_by_source.get(src.id, []):
            if dest.destination not in expertise[key]:
                expertise[key].append(dest.destination)

    # content tier from content_values table
    for cv in db.content_values:
        key = cv.source_value.lower()
        if key not in expertise:
            expertise[key] = []
        for t in db.content_fhir_targets:
            if t.content_value_id == cv.id and t.fhir_path not in expertise[key]:
                expertise[key].append(t.fhir_path)

    return expertise


def generate_embeddings(
    model_name: str = "all-MiniLM-L6-v2",
    output_path: Path = None,
) -> dict:
    """generate embeddings for all expertise mappings.

    args:
        model_name: sentence-transformer model
        output_path: where to save pkl (default: data/embeddings/expert_embeddings.pkl)

    returns:
        dict with expertise map and embeddings
    """
    from schema_crush.mappings.flat_loader import load_flat_mappings

    db = load_flat_mappings()
    expertise = build_expertise_map(db)

    print(f"expertise entries: {len(expertise)}")
    print(f"loading model: {model_name}")

    model = SentenceTransformer(model_name)

    # get unique values
    sources = list(expertise.keys())
    targets = list(set(p for paths in expertise.values() for p in paths))

    print(f"encoding {len(sources)} sources, {len(targets)} targets...")

    source_emb = model.encode(sources, show_progress_bar=True, convert_to_numpy=True)
    target_emb = model.encode(targets, show_progress_bar=True, convert_to_numpy=True)

    result = {
        "model_name": model_name,
        "expertise": expertise,
        "sources": sources,
        "targets": targets,
        "source_embeddings": source_emb,
        "target_embeddings": target_emb,
    }

    # save
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "embeddings" / "expert_embeddings.pkl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    print(f"saved: {output_path}")
    return result


def load_expert_embeddings(path: Path = None) -> dict:
    """load pre-computed expert embeddings."""
    if path is None:
        path = Path(__file__).parent.parent / "data" / "embeddings" / "expert_embeddings.pkl"

    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    generate_embeddings()
"""
UMAP visualization of agent mappings: source fields → FHIR targets

Run after demo to visualize what the agent mapped.
Uses BioBERT embeddings for both source and target (same 768-dim space).

Usage:
    python demo/plot_agent_umap.py
"""

import numpy as np
import plotly.graph_objects as go
from umap import UMAP

from schema_crush.mappings import load_flat_mappings
from schema_crush.tools.matchers import BioBERTMatcher


def plot_mappings_umap(mappings: list[dict], output_path: str = "agent_umap.html"):
    """
    Plot source→target mappings in UMAP space.

    Args:
        mappings: list of {"source": str, "target": str, "confidence": float}
        output_path: where to save HTML
    """
    # filter valid mappings
    valid = [m for m in mappings if m.get("target") and m["target"] != "(needs LLM)"]

    if len(valid) < 2:
        print("Need at least 2 mappings for UMAP")
        return

    sources = [m["source"] for m in valid]
    targets = [m["target"] for m in valid]
    confidences = [m.get("confidence", 0.5) for m in valid]

    # unique targets for positioning
    unique_targets = list(dict.fromkeys(targets))

    # embed with BioBERT (same model = same space)
    print("loading BioBERT embedder...")
    biobert = BioBERTMatcher(use_expert_embeddings=True)

    print(f"embedding {len(sources)} sources + {len(unique_targets)} targets...")
    source_embs = biobert.embedder.embed(sources)
    target_embs = biobert.embedder.embed(unique_targets)

    # combine for UMAP
    all_embs = np.vstack([source_embs, target_embs])

    # UMAP projection
    n_neighbors = min(5, len(all_embs) - 1)
    print(f"running UMAP (n_neighbors={n_neighbors})...")
    coords = UMAP(n_neighbors=n_neighbors, min_dist=0.3, random_state=42).fit_transform(all_embs)

    # build figure
    fig = go.Figure()

    # sources (blue circles)
    source_coords = coords[:len(sources)]
    fig.add_trace(go.Scatter(
        x=source_coords[:, 0],
        y=source_coords[:, 1],
        mode='markers+text',
        name='source fields',
        text=sources,
        textposition='top center',
        hovertext=[f"<b>{s}</b><br>→ {t}<br>conf: {c:.2f}"
                   for s, t, c in zip(sources, targets, confidences)],
        hoverinfo='text',
        marker=dict(
            size=16,
            color='rgba(70, 130, 180, 0.8)',  # steel blue
            symbol='circle',
            line=dict(width=2, color='darkblue')
        )
    ))

    # targets (green diamonds)
    target_coords = coords[len(sources):]
    fig.add_trace(go.Scatter(
        x=target_coords[:, 0],
        y=target_coords[:, 1],
        mode='markers+text',
        name='FHIR targets',
        text=[t.split('.')[-1] for t in unique_targets],  # show just field name
        textposition='bottom center',
        hovertext=[f"<b>{t}</b>" for t in unique_targets],
        hoverinfo='text',
        marker=dict(
            size=16,
            color='rgba(60, 179, 113, 0.8)',  # medium sea green
            symbol='diamond',
            line=dict(width=2, color='darkgreen')
        )
    ))

    # arrows from source to target
    for i, m in enumerate(valid):
        tgt_idx = len(sources) + unique_targets.index(m["target"])
        conf = m.get("confidence", 0.5)

        # color by confidence
        if conf >= 0.9:
            color = 'green'
        elif conf >= 0.7:
            color = 'orange'
        else:
            color = 'red'

        fig.add_annotation(
            x=coords[tgt_idx, 0], y=coords[tgt_idx, 1],
            ax=coords[i, 0], ay=coords[i, 1],
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor=color,
            opacity=0.6
        )

    fig.update_layout(
        title=dict(
            text="Agent Mappings: Source → FHIR Target",
            font=dict(size=20)
        ),
        xaxis=dict(title="UMAP-1", showgrid=False),
        yaxis=dict(title="UMAP-2", showgrid=False),
        template="plotly_white",
        width=1000,
        height=800,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.9)"
        ),
        hoverlabel=dict(font_size=14)
    )

    # save and show
    fig.write_html(output_path)
    print(f"saved: {output_path}")
    fig.show()


def main():
    """Demo with sample mappings from KB."""
    db = load_flat_mappings()

    # sample fields - mix of easy and hard
    test_fields = [
        "case_id", "gender", "primary_site", "tumor_grade",
        "sample_type", "vital_status", "age_at_diagnosis"
    ]

    # get mappings from KB
    mappings = []
    for field in test_fields:
        results = db.lookup(field)
        if results:
            _, dest = results[0]
            mappings.append({
                "source": field,
                "target": dest.destination,
                "confidence": 1.0
            })
        else:
            print(f"  {field} not in KB, skipping")

    print(f"\nplotting {len(mappings)} mappings...")
    plot_mappings_umap(mappings, "demo/agent_umap.html")


if __name__ == "__main__":
    main()
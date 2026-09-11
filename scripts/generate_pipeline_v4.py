#!/usr/bin/env python3
"""Generate Schema Crush Pipeline diagram v4.

Creates the complete pipeline architecture diagram with:
- 4-step cascade (KB Lookup -> Fuzzy -> Embeddings -> LLM)
- Knowledge Base and FHIR R5 output
- MCP Server with 14 tools (color-coded by category)
- 6-color legend including Feedback

Usage:
    python scripts/generate_pipeline_v4.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches


def generate_pipeline_diagram(output_path: str = "schema_crush_pipeline_v4.png"):
    """Generate the complete pipeline diagram."""

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')

    # Color palette (matching the original v2)
    colors = {
        'kb': '#7cb342',          # Green - KB Lookup
        'fuzzy': '#e89d5f',       # Orange - Fuzzy Match
        'embed': '#5b9bd5',       # Blue - Embeddings
        'llm': '#8e7cc3',         # Purple - LLM Reasoning
        'term': '#a8d5e5',        # Light Blue - Terminology
        'feedback': '#e89d9d',    # Coral - Feedback (NEW)
        'source': '#d4c896',      # Tan - Source data
        'mcp_bg': '#f5e6c8',      # Cream - MCP background
    }

    # ===================
    # TITLE
    # ===================
    ax.text(8, 11.5, 'Schema Crush Pipeline', fontsize=18, fontweight='bold', ha='center')

    # ===================
    # KNOWLEDGE BASE (top right)
    # ===================
    kb_box = FancyBboxPatch((8.5, 9.2), 2.2, 1.3, boxstyle="round,pad=0.05",
                             facecolor=colors['kb'], edgecolor='#555', linewidth=1.5)
    ax.add_patch(kb_box)
    ax.text(9.6, 10.15, 'Knowledge Base', fontsize=9, fontweight='bold', ha='center', color='white')
    ax.text(9.6, 9.8, '2,346 mappings', fontsize=8, ha='center', color='white')
    ax.text(9.6, 9.5, 'GDC | HTAN | CDA', fontsize=7, ha='center', color='white')
    ax.text(9.6, 9.25, 'ICGC | GTEx', fontsize=7, ha='center', color='white')

    # ===================
    # SOURCE DATA (left, aligned with Step 1)
    # ===================
    source = FancyBboxPatch((0.5, 8.0), 1.8, 0.9, boxstyle="round,pad=0.05",
                             facecolor=colors['source'], edgecolor='#888', linewidth=1.5)
    ax.add_patch(source)
    ax.text(1.4, 8.6, 'SOURCE DATA', fontsize=8, fontweight='bold', ha='center')
    ax.text(1.4, 8.25, 'CSV/JSON', fontsize=7, ha='center')

    # Arrow from source to Step 1
    ax.annotate('', xy=(3.0, 8.45), xytext=(2.3, 8.45),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

    # ===================
    # STEP BOXES (cascade)
    # ===================
    step_x = 4.0
    steps = [
        ('Step 1', 'KB Exact Lookup', colors['kb'], 8.0),
        ('Step 2', 'Fuzzy String Match', colors['fuzzy'], 6.3),
        ('Step 3', 'Embedding Similarity', colors['embed'], 4.6),
        ('Step 4', 'LLM Reasoning', colors['llm'], 2.9),
    ]

    for step_name, step_desc, color, y in steps:
        box = FancyBboxPatch((step_x, y), 2.2, 0.9, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='#555', linewidth=1.5)
        ax.add_patch(box)
        ax.text(step_x + 1.1, y + 0.6, step_name, fontsize=9, fontweight='bold',
                ha='center', color='white')
        ax.text(step_x + 1.1, y + 0.25, step_desc, fontsize=8, ha='center', color='white')

    # Vertical arrows between steps with "miss" labels
    for i in range(3):
        y_start = steps[i][3]
        y_end = steps[i+1][3] + 0.9
        ax.annotate('', xy=(step_x + 1.1, y_end), xytext=(step_x + 1.1, y_start),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))
        ax.text(step_x + 1.5, (y_start + y_end + 0.9) / 2, 'miss',
                fontsize=7, color='#666', style='italic')

    # ===================
    # ARROW: Step 1 -> Knowledge Base
    # ===================
    ax.annotate('', xy=(8.5, 9.5), xytext=(6.2, 8.6),
                arrowprops=dict(arrowstyle='->', color=colors['kb'], lw=1.5))
    ax.text(6.8, 9.2, 'conf = 1.0', fontsize=7, color='#555')

    # ===================
    # FHIR R5 MAPPING OUTPUT (right side)
    # ===================
    fhir_box = FancyBboxPatch((12.5, 5.8), 2.2, 1.4, boxstyle="round,pad=0.05",
                               facecolor=colors['kb'], edgecolor='#555', linewidth=1.5)
    ax.add_patch(fhir_box)
    ax.text(13.6, 6.8, 'FHIR R5', fontsize=9, fontweight='bold', ha='center', color='white')
    ax.text(13.6, 6.45, 'MAPPING', fontsize=9, fontweight='bold', ha='center', color='white')
    ax.text(13.6, 6.1, 'Resource.field', fontsize=8, ha='center', color='white')
    ax.text(13.6, 5.8, 'confidence', fontsize=8, ha='center', color='white')

    # Arrows from each step to FHIR output
    ax.annotate('', xy=(12.5, 6.5), xytext=(6.2, 8.45),
                arrowprops=dict(arrowstyle='->', color=colors['kb'], lw=1.2))
    ax.text(7.2, 7.7, 'conf = 0.85-0.95', fontsize=7, color='#555')

    ax.annotate('', xy=(12.5, 6.5), xytext=(6.2, 6.75),
                arrowprops=dict(arrowstyle='->', color=colors['fuzzy'], lw=1.2))

    ax.annotate('', xy=(12.5, 6.5), xytext=(6.2, 5.05),
                arrowprops=dict(arrowstyle='->', color=colors['embed'], lw=1.2))
    ax.text(7.8, 5.5, 'calibrated', fontsize=7, color='#555')

    ax.annotate('', xy=(12.5, 6.5), xytext=(6.2, 3.35),
                arrowprops=dict(arrowstyle='->', color=colors['llm'], lw=1.2))
    ax.text(8.0, 3.8, 'novel fields', fontsize=7, color='#555')

    # ===================
    # LEGEND (6 items including Feedback)
    # ===================
    legend_x = 12.5
    legend_y = 4.5
    legend_items = [
        ('KB Lookup', colors['kb']),
        ('Fuzzy Match', colors['fuzzy']),
        ('Embeddings', colors['embed']),
        ('LLM Reasoning', colors['llm']),
        ('Terminology', colors['term']),
        ('Feedback', colors['feedback']),  # NEW
    ]

    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y - i * 0.36
        rect = FancyBboxPatch((legend_x, y_pos), 0.35, 0.28, boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor='#888', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(legend_x + 0.5, y_pos + 0.14, label, fontsize=7, va='center')

    # ===================
    # MCP SERVER BOX (bottom)
    # ===================
    mcp_y = 0.3
    mcp_height = 2.2
    mcp_box = FancyBboxPatch((0.5, mcp_y), 15, mcp_height, boxstyle="round,pad=0.05",
                              facecolor=colors['mcp_bg'], edgecolor='#888', linewidth=1.5)
    ax.add_patch(mcp_box)
    ax.text(8, mcp_y + mcp_height - 0.2,
            'MCP Server (14 Tools) - Claude Desktop / Claude Code',
            fontsize=10, fontweight='bold', ha='center')

    # ===================
    # 14 MCP TOOLS (2 rows of 7)
    # ===================
    # Row 1: lookup_mapping, rule_match, biobert_match, magneto_match,
    #        find_similar, profile_csv, record_feedback
    # Row 2: search_fhir_fields, explore_fhir_resource, search_snomed, search_loinc,
    #        search_ontology, get_transformation_rules, get_feedback_stats

    tools_row1 = [
        ('lookup_mapping', colors['kb']),
        ('rule_match', colors['fuzzy']),
        ('biobert_match', colors['embed']),
        ('magneto_match', colors['embed']),
        ('find_similar', colors['embed']),
        ('profile_csv', colors['llm']),
        ('record_feedback', colors['feedback']),
    ]

    tools_row2 = [
        ('search_fhir_fields', colors['kb']),
        ('explore_fhir_resource', colors['kb']),
        ('search_snomed', colors['term']),
        ('search_loinc', colors['term']),
        ('search_ontology', colors['term']),
        ('get_transformation_rules', colors['kb']),
        ('get_feedback_stats', colors['feedback']),
    ]

    # Tool box dimensions
    box_width = 1.9
    spacing = 2.1
    tool_height = 0.55

    # Row 1
    for i, (tool, tc) in enumerate(tools_row1):
        x = 0.7 + i * spacing
        y = mcp_y + 1.2
        box = FancyBboxPatch((x, y), box_width, tool_height, boxstyle="round,pad=0.02",
                              facecolor=tc, edgecolor='#888', linewidth=0.5, alpha=0.4)
        ax.add_patch(box)
        # Dark text for light colors, white for dark colors
        text_color = '#333' if tc in [colors['term'], colors['feedback']] else '#333'
        ax.text(x + box_width/2, y + tool_height/2, tool,
                fontsize=6, ha='center', va='center', color=text_color)

    # Row 2
    for i, (tool, tc) in enumerate(tools_row2):
        x = 0.7 + i * spacing
        y = mcp_y + 0.5
        box = FancyBboxPatch((x, y), box_width, tool_height, boxstyle="round,pad=0.02",
                              facecolor=tc, edgecolor='#888', linewidth=0.5, alpha=0.4)
        ax.add_patch(box)
        text_color = '#333'
        ax.text(x + box_width/2, y + tool_height/2, tool,
                fontsize=6, ha='center', va='center', color=text_color)

    # ===================
    # SAVE
    # ===================
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    generate_pipeline_diagram('schema_crush_pipeline_v4.png')
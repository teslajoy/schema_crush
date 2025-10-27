# schema_crush

schema_crush is a modular framework for semantic schema matching that leverages multi-agent embedders (BioBERT, Magneto) and human-in-the-loop validation to align heterogeneous biomedical schemas. It supports 3-tier matching (entity, field, content) with weighted embedder fusion, confidence-based routing, and automated reporting. 

## install

```bash
pip install -r requirements.txt
```
or 
```bash 
pip install -e .
```

## usage

run the gdc embedder evaluation example:

```bash
python examples/evaluate_gdc_embedders.py
```

this evaluates biobert vs magneto on gdc->fhir mappings and generates a full report in `examples/reports/`.

## architecture

### 3-tier matching strategy

1. **entity matching**: table/entity name similarity
2. **field matching**: column name similarity
3. **content matching**: column name + sample values similarity

### embedders

- **biobert**: biomedical text embeddings (dmis-lab/biobert-v1.1)
- **magneto**: schema matching embeddings trained on gdc benchmark

### pearl agent workflow

```
perceive -> reason -> act -> hitl -> learn -> (next tier or end)
```

- **perceive**: extract schema metadata
- **reason**: generate matches using weighted embedder combination
- **act**: filter and rank matches by confidence
- **hitl**: queue medium confidence (85-95%) matches for human review
- **learn**: collect metrics and update from feedback

### confidence levels

- **high**: score >= 95% (auto-accept)
- **medium**: score 85-95% (requires hitl)
- **low**: score < 85% (filtered out)

## output files

all results saved to `results/` directory:

- `*_mappings.json`: final source -> target mappings
- `*_scores.json`: detailed scores with embedder contributions
- `*_reasoning.txt`: agent decision log
- `*_metadata.json`: run configuration and metrics
- `*_hitl_queue.json`: matches requiring human review
- `*_summary.txt`: human-readable summary

## testing

```bash
pytest tests/
```

## components

- `loaders/`: data loaders (csv, json, etc)
- `tools/embeddings/`: biobert and magneto embedders
- `orchestrator/`: pearl agent workflow
- `reporting/`: result report generation
- `data/`: example data and ground truth mappings
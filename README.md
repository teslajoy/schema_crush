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

```python
from schema_crush.loaders.csv_loader import CSVLoader
from schema_crush.tools.embeddings.biobert_embedder import BioBERTEmbedder
from schema_crush.tools.embeddings.magneto_embedder import MagnetoEmbedder
from schema_crush.orchestrator.pearl_agent import PEARLAgent
from schema_crush.reporting.mapping_report import MappingReport

# load source data
loader = CSVLoader()
source_data = loader.load({'case': 'data/examples/case.csv'})

# define target schema
target_schema = {
    'Patient': ['id', 'identifier', 'birthDate', 'gender'],
    'Condition': ['code', 'id']
}

# create embedders
embedders = {
    'biobert': BioBERTEmbedder(),
    'magneto': MagnetoEmbedder()
}

# run matching
agent = PEARLAgent(
    embedders=embedders,
    embedder_weights={'biobert': 0.3, 'magneto': 0.7},
    confidence_thresholds={'high': 0.95, 'medium': 0.85},
    top_k=5
)

results = agent.run(
    source_data=source_data,
    target_schema=target_schema,
    tiers=['entity', 'field', 'content']
)

# generate report
reporter = MappingReport(output_dir='results')
files = reporter.generate(
    results=results,
    source_name='gdc_case',
    target_name='fhir',
    embedders_used=['biobert', 'magneto'],
    embedder_weights={'biobert': 0.3, 'magneto': 0.7},
    confidence_thresholds={'high': 0.95, 'medium': 0.85}
)

print(f"results saved to: {files['summary']}")
```

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

## performance

evaluated on gdc -> fhir mappings (20 fields):

| embedder | precision@1 | recall@5 | mrr |
|----------|-------------|----------|-----|
| magneto  | 25.0%       | 65.0%    | 0.38 |
| biobert  | 5.0%        | 15.0%    | 0.09 |

magneto trained on gdc benchmark outperforms biobert 5x for schema matching tasks on GDC data particularly.

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
# Howard OpenContracts Viability Evaluation

This script evaluates whether stock OpenContracts plus a frontier model is a
viable foundation for Howard before adding custom legal-intelligence
infrastructure.

## Fixture Package

Fixture root:

```bash
opencontractserver/benchmarks/fixtures/howard_expedia_trx
```

Public SEC source documents:

| Key | Document | Source |
| --- | --- | --- |
| `msa_2007` | Expedia/TRX Master Services Agreement, effective January 1, 2007 | `https://www.sec.gov/Archives/edgar/data/1103025/000119312507035944/dex1015.htm` |
| `sow_2008` | Amended and Restated Statement of Work for Contact Center Services, effective January 1, 2008 | `https://www.sec.gov/Archives/edgar/data/1103025/000119312511141447/dex102.htm` |
| `amendment_2011` | Amendment No. 2 to the Amended and Restated SOW, effective March 1, 2011 | `https://www.sec.gov/Archives/edgar/data/1103025/000119312511307073/d233980dex106.htm` |

The fixture PDFs are stable text-preserving renders from the visible SEC HTML
exhibits. If regenerating them, fetch the same SEC pages, preserve the visible
text, and place the PDFs under
`opencontractserver/benchmarks/fixtures/howard_expedia_trx/corpus/` using the
filenames in `source_manifest.json`.

Do not repair or infer redacted text. Redactions are part of the test.

## Ingest Readiness Check

Run a no-write fixture validation first:

```bash
python manage.py run_howard_viability_eval \
  --user admin \
  --dry-run
```

Run the real ingestion check:

```bash
python manage.py run_howard_viability_eval \
  --user admin \
  --timeout-seconds 600 \
  --embedding-timeout-seconds 300
```

The command creates a fresh corpus, uploads each PDF through
`DocumentService.create_document`, waits for parsing, waits for document
embeddings, then adds the processed document to the corpus through
`CorpusDocumentService.add_document_to_corpus`. This order matches the
corpus-isolated copy model, where corpus documents inherit completed parsing
artifacts from their source document.

The command intentionally verifies document embeddings with:

```python
doc.get_embedding(
    "opencontractserver.pipeline.embedders.sent_transformer_microservice.MicroserviceEmbedder",
    384,
)
```

Do not use `Document.embedding` for this check. That legacy field can remain
null even when the current `Embedding` table has a valid document embedding.

## Question Set

Use the fixed 15-question set in `questions.json`. Every answer must include:

- Claim
- Source document
- Source location, page, or annotation
- Supporting text
- Confidence
- Conflicts
- Unknowns

The evaluation must independently compare every substantive claim against
stored OpenContracts text and citations. Do not use the answering model as its
own verifier.

## Failure Categories

Classify failures as:

- Parsing
- Retrieval or reranking
- Model reasoning
- Citation or provenance
- Cross-document effective-term resolution

## Acceptance Thresholds

Initial thresholds:

- 100% of citations resolve to actual stored text
- Zero unsupported substantive claims
- 100% accuracy on parties, dates, money, and controlling terms
- At least 90% overall answer accuracy
- Amendment supersession is resolved correctly
- The model states uncertainty when the documents are silent or redacted
- Comparable results across at least three runs

## Architecture Rule

Do not build a custom ontology, knowledge graph, relationship engine, or Howard
UI until this evaluation identifies a demonstrated gap. If stock OpenContracts
passes, Howard should start with the thinnest product layer. If it fails,
identify the smallest targeted addition required.

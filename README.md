# AgenticMatcher

**A Tool-Augmented LLM Agent for Schema Matching**

AgenticMatcher uses schema, instance, and documentation evidence to match each source attribute to target attributes. Its scorer adjusts confidence scores while preserving the returned candidates and their order.

This repository includes the matching runtime, pretrained scorers, examples, and an experiment reproduction entrypoint. Benchmark tables are downloaded separately.

## Installation

Python 3.11+ is required; Python 3.12 has been tested.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

If `venv` reports missing `ensurepip`, install your system's Python virtual-environment support.

For embedding-based retrieval, install the additional dependencies:

```bash
python -m pip install '.[embeddings]'
export ASM_ALLOW_REMOTE_EMBEDDING_DOWNLOAD=1
```

This permits downloading `BAAI/bge-large-en-v1.5`. With downloads disabled, the encoder must already be cached.

## API Configuration

Supply your own OpenAI-compatible API endpoint, model identifier, and key. No provider settings are supplied by default.

```bash
read -rp 'API base URL: ' ASM_BASE_URL
read -rp 'Model name: ' ASM_MODEL
read -rsp 'API key: ' ASM_API_KEY; echo
export ASM_BASE_URL ASM_MODEL ASM_API_KEY
```

Use an HTTPS endpoint, including its API path; HTTP is accepted only for localhost. Do not put credentials in the URL. Settings are read from environment variables. Keep keys out of source files and commands. Run `unset ASM_API_KEY` when finished.

## Datasets

The paper evaluates six datasets. Download their source materials from the links below; benchmark tables and reference mappings are not bundled with this repository. Suggested locations are relative to the repository root; `data/` is ignored by Git.

| Dataset | Source / download | Suggested location |
| --- | --- | --- |
| GDC | [Magneto data preparation](https://github.com/VIDA-NYU/magneto-matcher#data-preparation) | `data/gdc/` |
| Magellan | [Valentine datasets](https://zenodo.org/records/5084605), Magellan subset | `data/valentine/` |
| Wikidata | [Valentine datasets](https://zenodo.org/records/5084605), Wikidata subset | `data/valentine/` |
| Synthea-OMOP | [SMAT / OMAP benchmark](https://github.com/JZCS2018/SMAT/tree/main/datasets/omap), `omop_synthea_data.xlsx` | `data/synthea-omop/` |
| HRS-B | [ConStruM HRS-B](https://github.com/HoumingChen/construm/tree/main/hrs_b) | `data/hrs_b/` |
| PSID-B | [ConStruM PSID-B](https://github.com/HoumingChen/construm/tree/main/psid_b) | `data/psid_b/` |

Preserve the upstream directory layouts. The Synthea-OMOP link provides the upstream benchmark from which the paper's task was reconstructed. The included reproduction runner supports GDC; the other links document the sources used in the paper and do not imply additional dataset runners.

For GDC, place the extracted files in this layout:

```text
data/gdc/
├── source-tables/       # Ten original task CSV files
├── target-tables/
│   └── gdc_unique_columns_concat_values.csv
└── ground-truth/        # Ten reference-mapping CSV files
```

Run from the repository root and set:

```bash
export ASM_GDC_DATA_DIR="data/gdc"
```

You may use another location by changing `ASM_GDC_DATA_DIR`. Keep the original CSV filenames and contents unchanged. The GDC field dictionary is already bundled.

## Quick Start

Run the small synthetic example after configuring your API:

```bash
agenticmatcher match \
  --source examples/source.csv \
  --target examples/target.csv \
  --docs examples/docs.json \
  --output outputs/example-matches.json
```

Try offline scoring without API access:

```bash
agenticmatcher rescore \
  --input examples/snapshot.json \
  --output outputs/example-rescored.json
```

The matching example contains three synthetic product records with differently named columns; row order need not agree. The offline snapshot contains illustrative scores, not recorded model predictions. Existing output files are not overwritten.

## Experiment Reproduction

The included reproduction entrypoint uses GDC as a worked example. Prepare the data and set `ASM_GDC_DATA_DIR` as described in [Datasets](#datasets).

Install the embeddings extra, configure your API, and select the paper's Claude Sonnet 4.5 model using your provider's identifier. To reproduce the experiment with the provided implementation:

```bash
agenticmatcher-reproduce --dataset gdc
```

Add `--check-only` to verify data and models without API calls. The runner covers ten tasks and 569 source attributes, reporting per-task and macro MRR and Recall@GT before and after scoring. Each task uses a model trained on the other nine tasks; reference mappings are used only by the evaluator.

Every invocation creates a new run directory with predictions and `summary.json`. Set `ASM_OUTPUT_DIR` to choose the output directory. Failed attributes remain in the evaluation denominator. This entrypoint evaluates the supplied models; it does not train them.

## Custom Data

Use `agenticmatcher match --help` for your own CSV tables and optional documentation; see `examples/docs.json` for the documentation format. Use `agenticmatcher rescore --help` for offline scoring. Use the reproduction entrypoint above for benchmark evaluation.

API calls send input-derived evidence to your provider. Run generated Python only in an isolated environment without sensitive files.

## Availability

The code is publicly available for inspection. Open-source licensing is pending; see `LICENSE` and `THIRD_PARTY_NOTICES.txt` for the current terms.

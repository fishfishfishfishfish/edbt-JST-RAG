#!/usr/bin/env bash
# JSTRAG x TempEvalRAG benchmark
#
# Ingest (full ingest by default):
#  - Each document is first routed by models/spacy_suitability_logreg_th6.joblib:
#      pred=1 -> local spaCy triple extraction (zero LLM)
#      pred=0 -> llama3.2 (Ollama) extraction
#  - Builds a hybrid-only index; it does not overwrite existing full-LLM / spacy indexes
# Retrieval:
#  - BM25 recall 1000 -> full temporal_coeff -> TxC dual-axis partitions -> diagonal
#    levels L1..L6 -> sliding-window semantic reranking + hybrid scoring ->
#    dominance-gated early stop -> top_k
#
# Full ingest + evaluation by default (takes several hours; only ~15%-20% of
# documents go to the LLM); once the hybrid index is built, add --reuse-db to
# skip ingest and re-run evaluation.
#
# - Config:  configs/jstrag_default.yaml
# - Index:   data/nuggetindex_hybrid-full-TempEvalRAG.db (overridable with --db-path)
# - Dataset: data/TempEvalRAG directory (docs.jsonl + query.jsonl)
# - TempEvalRAG data has no reference_time, so the epoch source_date slow path is not triggered
#
# Usage:
#   conda activate tcrag
#   # Small trial run (20 passages ingested + 5 queries, retrieval only)
#   bash scripts/run_benchmark_jstrag_tempevalrag.sh \
#       --passage-limit 20 --query-limit 5 --no-qa
#   # First full run: build the hybrid index + full evaluation (retrieval + llama3.2 QA)
#   bash scripts/run_benchmark_jstrag_tempevalrag.sh
#   # Afterwards reuse the hybrid index for evaluation only
#   bash scripts/run_benchmark_jstrag_tempevalrag.sh \
#       --reuse-db --no-qa
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/configs/jstrag_default.yaml"

# TempEvalRAG-specific index produced by the hybrid constructor (overridable with --db-path)
DB_PATH="$REPO_ROOT/data/TempEvalRAG.db"
# Dataset: TempEvalRAG directory (contains docs.jsonl + query.jsonl)
DATASET="$REPO_ROOT/data/TempEvalRAG"

# Parse arguments this script cares about; everything else is forwarded as-is to the Python entrypoint
PASS_ARGS=()
REUSE_DB=0
NO_QA=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --db-path)
            DB_PATH="$2"; shift 2 ;;
        --dataset)
            DATASET="$2"; shift 2 ;;
        --reuse-db)
            REUSE_DB=1; PASS_ARGS+=("$1"); shift ;;
        --no-qa)
            NO_QA=1; PASS_ARGS+=("$1"); shift ;;
        *)
            PASS_ARGS+=("$1"); shift ;;
    esac
done

# Pick the interpreter with project dependencies: $PYTHON > python on PATH > conda tcrag env
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif python -c 'import yaml' >/dev/null 2>&1; then
    PY="python"
elif [[ -x "$HOME/miniconda3/envs/tcrag/bin/python" ]]; then
    PY="$HOME/miniconda3/envs/tcrag/bin/python"
else
    echo "ERROR: cannot find a Python with project dependencies; please run 'conda activate tcrag' or set \$PYTHON" >&2
    exit 1
fi

# Reuse mode requires the index to exist; full ingest mode does not (a new one
# is created and an existing one is backed up first).
if [[ "$REUSE_DB" -eq 1 && ! -f "$DB_PATH" ]]; then
    echo "ERROR: index file specified by --reuse-db does not exist: $DB_PATH" >&2
    echo "       please run once without --reuse-db to build the hybrid index" >&2
    exit 1
fi

# Dataset directory check (TempEvalRAG requires docs.jsonl + query.jsonl)
if [[ ! -f "$DATASET/docs.jsonl" || ! -f "$DATASET/query.jsonl" ]]; then
    echo "ERROR: TempEvalRAG dataset is incomplete: $DATASET must contain docs.jsonl and query.jsonl" >&2
    exit 1
fi

# Semantic reranker note: when there is no local cache and huggingface.co is
# unreachable, the runtime downgrades to BM25 ranking after the model fails to
# load (the cascade then falls back to the full path); warn here in advance.
RERANKER="${RERANKER_MODEL:-nvidia/NV-Embed-v2}"
case "$RERANKER" in
    /*) : ;;  # local path, skip cache check
    *)
        _hf_cache_dir="$HOME/.cache/huggingface/hub/models--${RERANKER//\//--}"
        if [[ ! -d "$_hf_cache_dir" ]] && ! curl -sf --max-time 5 -o /dev/null https://huggingface.co; then
            echo "WARNING: no local cache for $RERANKER and huggingface.co is unreachable;" >&2
            echo "         results will fall back to BM25 ranking and do not reflect semantic reranking performance." >&2
        fi
        ;;
esac

# Both the LLM branch of LR routing during ingest (non --reuse-db) and QA need
# Ollama; only the --reuse-db --no-qa combination is fully independent of Ollama.
if [[ "$REUSE_DB" -eq 0 || "$NO_QA" -eq 0 ]]; then
    if ! curl -sf --max-time 5 http://localhost:11434/api/tags >/dev/null; then
        echo "ERROR: Ollama is unreachable; please start it first: ollama serve" >&2
        echo "       (only the '--reuse-db --no-qa' retrieval-only evaluation can run without Ollama)" >&2
        exit 1
    fi
fi

"$PY" "$REPO_ROOT/scripts/run_benchmark_jstrag.py" \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --format tempevalrag \
    --db-path "$DB_PATH" \
    --per-query \
    "${PASS_ARGS[@]+"${PASS_ARGS[@]}"}"

"""Document and query loaders for common dataset formats.

Supports:
  - JSON / JSONL (TimeQA-style annotated records, or a list of
    ``{source_id, text, ...}`` records)
  - CSV

The loader auto-detects whether a file contains documents or queries based
on the presence of an ``answers`` / ``relevant_doc_ids`` field.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from tcrag.data.models import Document, Query


def _parse_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl" or text.startswith("{"):
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
    data = json.loads(text)
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _load_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load raw records from json / jsonl / csv."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    suffix = p.suffix.lower()
    if suffix in (".json", ".jsonl"):
        return _load_json_records(p)
    if suffix == ".csv":
        return _load_csv_records(p)
    raise ValueError(f"Unsupported dataset format: {suffix} (use .json/.jsonl/.csv)")


def _looks_like_query(rec: dict[str, Any]) -> bool:
    return "answers" in rec or "relevant_doc_ids" in rec or "answer" in rec


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        # Allow comma-separated lists in CSV.
        return [s.strip() for s in value.split(",") if s.strip()]
    return [str(value)]


def records_to_documents(records: Iterable[dict[str, Any]]) -> list[Document]:
    """Convert raw records into :class:`Document` objects.

    Expected fields: ``source_id`` (or ``id``), ``text`` (or ``content``/``passage``).
    """
    docs: list[Document] = []
    for rec in records:
        source_id = rec.get("source_id") or rec.get("id")
        text = rec.get("text") or rec.get("content") or rec.get("passage") or ""
        if not source_id or not text:
            continue
        docs.append(
            Document(
                source_id=str(source_id),
                text=str(text),
                uri=rec.get("uri") or rec.get("url"),
                reference_time=_parse_time(
                    rec.get("reference_time") or rec.get("source_date") or rec.get("timestamp")
                ),
                metadata={k: v for k, v in rec.items() if k not in {"source_id", "id", "text", "content", "passage", "uri", "url", "reference_time", "source_date", "timestamp"}},
            )
        )
    return docs


def records_to_queries(records: Iterable[dict[str, Any]]) -> list[Query]:
    """Convert raw records into :class:`Query` objects.

    Expected fields: ``id``, ``text`` (or ``question``/``query``),
    ``answers`` (or ``answer``), ``relevant_doc_ids`` (or ``relevant_ids``).
    """
    queries: list[Query] = []
    for rec in records:
        qid = rec.get("id") or rec.get("query_id")
        text = rec.get("text") or rec.get("question") or rec.get("query")
        if not qid or not text:
            continue
        answers = _coerce_list(rec.get("answers", rec.get("answer")))
        relevant = _coerce_list(rec.get("relevant_doc_ids", rec.get("relevant_ids")))
        queries.append(
            Query(
                id=str(qid),
                text=str(text),
                reference_time=_parse_time(
                    rec.get("reference_time") or rec.get("time") or rec.get("timestamp")
                ),
                relevant_doc_ids=relevant,
                answers=answers,
                metadata={k: v for k, v in rec.items() if k not in {"id", "query_id", "text", "question", "query", "answers", "answer", "relevant_doc_ids", "relevant_ids", "reference_time", "time", "timestamp"}},
            )
        )
    return queries


def load_documents(path: str | Path) -> list[Document]:
    """Convenience: load a file and interpret every record as a Document."""
    return records_to_documents(load_records(path))


def load_queries(path: str | Path) -> list[Query]:
    """Convenience: load a file and interpret every record as a Query."""
    return records_to_queries(load_records(path))


def load_dataset(path: str | Path) -> tuple[list[Document], list[Query]]:
    """Auto-detect and split a single dataset file into documents and queries."""
    records = load_records(path)
    docs = records_to_documents([r for r in records if not _looks_like_query(r)])
    queries = records_to_queries([r for r in records if _looks_like_query(r)])
    return docs, queries


# ---------------------------------------------------------------------------
# TimeQA-specific loader (annotated_dev.json format)
# ---------------------------------------------------------------------------

# Default location of the Wikidata property-id -> label mapping generated by
# ``data/fetch_wikidata_properties.py``. Used to translate TimeQA ``type``
# values (e.g. ``P1435``) into human-readable property names (e.g. "heritage
# designation") so that synthesized query text reads naturally.
_DEFAULT_WIKIDATA_MAPPING = Path(__file__).resolve().parents[2] / "data" / "wikidata_properties.json"


def _load_wikidata_property_mapping(
    mapping_path: str | Path | None = None,
    *,
    language: str = "en",
) -> dict[str, str]:
    """Load a Wikidata property-id -> label mapping (JSON).

    ``mapping_path`` defaults to ``data/wikidata_properties.json`` next to the
    project root. Returns an empty dict if the file is missing; callers fall
    back to the raw property id in that case.
    """
    path = Path(mapping_path) if mapping_path else _DEFAULT_WIKIDATA_MAPPING
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        pid: (labels.get(language) or labels.get("en") or pid)
        for pid, labels in data.items()
        if isinstance(labels, dict)
    }


def load_timeqa(
    path: str | Path,
    *,
    passage_limit: int | None = None,
    query_limit: int | None = None,
    property_mapping: dict[str, str] | str | Path | None = None,
    mapping_language: str = "en",
) -> tuple[list[Document], list[Query]]:
    """Load TimeQA ``annotated_dev.json`` into documents and queries.

    Each non-empty paragraph becomes a :class:`Document`. Each annotated
    answer window becomes a :class:`Query` whose ``relevant_doc_ids`` are
    the paragraphs named by its ``para`` annotations.

    ``property_mapping`` resolves Wikidata property IDs (e.g. ``P1435``) to
    human-readable labels. Pass a pre-built ``{pid: label}`` dict, a path to
    a JSON mapping file, or ``None`` to auto-load the default mapping at
    ``data/wikidata_properties.json``. Unresolved IDs fall back to the raw ID.
    """
    records = load_records(path)
    docs: list[Document] = []
    queries: list[Query] = []
    seen_doc_ids: set[str] = set()

    if isinstance(property_mapping, (str, Path)):
        prop_map = _load_wikidata_property_mapping(property_mapping, language=mapping_language)
    elif isinstance(property_mapping, dict):
        prop_map = property_mapping
    else:
        prop_map = _load_wikidata_property_mapping(language=mapping_language)

    # Fallback: if no question carries a parseable time, use datetime.min so
    # unannotated chunks still get a timezone-aware, very-early validity_start.
    fallback_valid_from =  datetime.min.replace(tzinfo=UTC)

    for record in records:
        wiki_link = str(record.get("link") or "")
        entity = wiki_link.removeprefix("/wiki/").replace("_", " ") or "entity"
        raw_type = str(record.get("type") or "property")
        property_type = prop_map.get(raw_type, raw_type)
        paragraphs = record.get("paras") or []
        if not isinstance(paragraphs, list):
            continue

        # Build passage validity windows from annotated questions.
        passage_validity: dict[int, tuple[datetime | None, datetime | None]] = {}
        for q_data in record.get("questions") or []:
            if not isinstance(q_data, list) or len(q_data) < 2:
                continue
            time_range = q_data[0] if isinstance(q_data[0], list) else []
            valid_from = _parse_time(time_range[0]) if len(time_range) > 0 else None
            valid_to = _parse_time(time_range[1]) if len(time_range) > 1 else None
            annotations = q_data[1] if isinstance(q_data[1], list) else []
            for ann in annotations:
                if isinstance(ann, dict) and ann.get("para") is not None:
                    passage_validity[int(ann["para"])] = (valid_from, valid_to)

        # Build documents for each paragraph.
        for idx, para in enumerate(paragraphs):
            text = str(para or "").strip()
            if not text:
                continue
            doc_id = f"timeqa_{wiki_link}_{idx}"
            if doc_id in seen_doc_ids:
                continue
            valid_from, valid_to = passage_validity.get(idx, (None, None))
            # Unannotated passages have no question-time mapping; fall back to
            # a very-earliest-question-time so they stay retrievable for every
            # query (instead of silently defaulting to ingestion now()).
            if valid_from is None:
                valid_from = fallback_valid_from
            docs.append(
                Document(
                    source_id=doc_id,
                    text=text,
                    uri=wiki_link,
                    reference_time=valid_from,
                    metadata={"valid_to": valid_to.isoformat()} if valid_to else {},
                )
            )
            seen_doc_ids.add(doc_id)
            if passage_limit is not None and len(docs) >= passage_limit:
                break
        if passage_limit is not None and len(docs) >= passage_limit:
            break

        # Build queries from annotated questions.
        for q_idx, q_data in enumerate(record.get("questions") or []):
            if not isinstance(q_data, list) or len(q_data) < 2:
                continue
            time_range = q_data[0] if isinstance(q_data[0], list) else []
            annotations = q_data[1] if isinstance(q_data[1], list) else []
            answers = [
                str(a.get("answer")).strip()
                for a in annotations
                if isinstance(a, dict) and str(a.get("answer") or "").strip()
            ]
            relevant = [
                f"timeqa_{wiki_link}_{int(a['para'])}"
                for a in annotations
                if isinstance(a, dict) and a.get("para") is not None
            ]
            if not answers or not relevant:
                continue
            ref_time = _parse_time(time_range[0]) if time_range else None
            q_text = f"What was {entity}'s {property_type}?"
            if time_range:
                q_text += f" (as of {time_range[0]})"
            queries.append(
                Query(
                    id=f"timeqa_{wiki_link}_{property_type}_{q_idx}",
                    text=q_text,
                    reference_time=ref_time,
                    relevant_doc_ids=relevant,
                    answers=answers,
                )
            )
            if query_limit is not None and len(queries) >= query_limit:
                break
        if query_limit is not None and len(queries) >= query_limit:
            break

    return docs, queries


def load_ravine(
    path: Path, 
    passage_limit: int | None = None, 
    query_limit: int | None = None) -> tuple[list[Document], list[Query]]:
    """
    Load RAVine nugget annotations.
    RAVine queries have no reference time and no answers; 
    they are only used to test retrieval capabilities.
    """
    docs_path = path / "nuggets_docs.jsonl"
    query_path = path / "nuggets_annotations.jsonl"
    docs: list[Document] = []
    queries: list[Query] = []
    with open(query_path) as f:
        for line in f:
            data = json.loads(line)
            docids = []
            for nugget in data.get("nuggets", []):
                docids.extend(nugget.get("docids", []))

            queries.append(Query(
                id=data["qid"],
                text=data["query"],
                reference_time=None,
                relevant_doc_ids=docids,
                answers=[""],
                metadata=data.get("meta", {}),
            ))
            if query_limit is not None and len(queries) >= query_limit:
                break

    fallback_reference_time = datetime.min.replace(tzinfo=UTC)
    with open(docs_path) as f:
        for line in f:
            data = json.loads(line)
            docs.append(Document(
                source_id=data["docid"],
                text=data["body"],
                uri=data["url"],
                reference_time=data.get("reference_time", fallback_reference_time),
                metadata={
                    "title": data.get("title", ""),
                    "headings": data.get("headings", ""),
                }
            ))
            if passage_limit is not None and len(docs) >= passage_limit:
                break
    return docs, queries

def load_tempevalrag(
    path: Path, 
    passage_limit: int | None = None, 
    query_limit: int | None = None) -> tuple[list[Document], list[Query]]:
    docs_path = path / "docs.jsonl"
    query_path = path / "query.jsonl"
    docs: list[Document] = []
    queries: list[Query] = []
    
    with open(query_path) as f:
        for line in f:
            data = json.loads(line)
            # reference_time is a string in query.jsonl (e.g. the year "2018");
            # parse it to datetime, falling back to the current time on failure.
            reference_time = _parse_time(data.get("reference_time"))
            if reference_time is None:
                reference_time = datetime.now(UTC)
            queries.append(Query(
                id=data["id"],
                text=data["text"],
                reference_time=reference_time,
                relevant_doc_ids=data.get("relevant_doc_ids", []),
                answers=data.get("answers", []),
                metadata={
                    "exact_time": data.get("exact_time", ""),
                    "time_relation": data.get("time_relation", ""),
                    "original_dataset": data.get("original_dataset", ""),
                    "original_id": data.get("original_id", ""),
                },
            ))
            if query_limit is not None and len(queries) >= query_limit:
                break

    fallback_reference_time = datetime.min.replace(tzinfo=UTC)
    with open(docs_path) as f:
        for line in f:
            data = json.loads(line)
            docs.append(Document(
                source_id=data["id"],
                text=data["text"],
                uri=data["title"] + "/" + data["section"],
                reference_time=data.get("reference_time", fallback_reference_time),
                metadata={
                    "title": data.get("title", ""),
                    "section": data.get("section", ""),
                }
            ))
            if passage_limit is not None and len(docs) >= passage_limit:
                break
    return docs, queries


def load_situatedqa(
    path: Path, 
    passage_limit: int | None = None, 
    query_limit: int | None = None) -> tuple[list[Document], list[Query]]:
    docs: list[Document] = []
    queries: list[Query] = []
    
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            # Get the question
            question = data.get("question", "")
            edited_question = data.get("edited_question", question)
            # Get the id
            qid = data.get("id", hash(edited_question))
            docid = f"situatedqa_passage_{qid}"
            # reference_time is a string in query.jsonl (e.g. the year "2018");
            # parse it to datetime, falling back to the current time on failure.
            origin_time = _parse_time(data.get("date"))
            q_time = origin_time
            doc_time = origin_time            
            if origin_time is None:
                q_time = datetime.now(UTC)
                doc_time = datetime.min.replace(tzinfo=UTC)    
            answers = data.get("answer", [])
            if not isinstance(answers, list):
                answer_str = str(answers) if answers else ""
                answers = [answer_str] if answer_str else []
            queries.append(Query(
                id=qid,
                text=edited_question,
                reference_time=q_time,
                relevant_doc_ids=[docid],
                answers=answers,
            ))
            docs.append(Document(
                source_id=docid,
                text=f"{question} {answers}",
                uri=question,
                reference_time=doc_time,
            ))
            if query_limit is not None and len(queries) >= query_limit:
                break
            if passage_limit is not None and len(docs) >= passage_limit:
                break
    return docs, queries
"""Fail-closed SkillPixel top-100 CSV exporter and validator."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hcmaic.skillpixel.retrieval import SkillPixelHit, load_skillpixel_questions

ANSWER_COUNT = 100
_FRAME_INDEX_RE = re.compile(r"^[0-9]+$")


class SubmissionValidationError(ValueError):
    """Raised when a submission cannot satisfy the organizer contract."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class SubmissionStats:
    n_queries: int
    answers_per_query: int
    output_path: Path


@dataclass(frozen=True)
class SubmissionValidationReport:
    path: Path
    n_queries: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _expected_header() -> list[str]:
    return ["query_id"] + [f"answer_{index:03d}" for index in range(1, ANSWER_COUNT + 1)]


def _load_corpus(path: Path) -> dict[str, int]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        filename_column = "video" if "video" in fields else "video_filename"
        missing = {filename_column, "frame_count"} - fields
        if missing:
            raise SubmissionValidationError(
                [f"corpus.csv missing required columns: {sorted(missing)}"]
            )
        corpus: dict[str, int] = {}
        for line, row in enumerate(reader, start=2):
            filename = (row.get(filename_column) or "").strip()
            if not filename:
                raise SubmissionValidationError([f"corpus.csv line {line}: empty video filename"])
            if filename in corpus:
                raise SubmissionValidationError(
                    [f"corpus.csv duplicate video filename {filename!r}"]
                )
            try:
                frame_count = int(str(row["frame_count"]).strip())
            except (KeyError, TypeError, ValueError) as exc:
                raise SubmissionValidationError(
                    [f"corpus.csv line {line}: invalid frame_count"]
                ) from exc
            if frame_count < 1:
                raise SubmissionValidationError(
                    [f"corpus.csv line {line}: frame_count must be >= 1"]
                )
            corpus[filename] = frame_count
    if not corpus:
        raise SubmissionValidationError(["corpus.csv has no videos"])
    return corpus


def _load_result_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SubmissionValidationError(
                    [f"results JSONL line {line_number}: invalid JSON"]
                ) from exc
            if not isinstance(payload, dict):
                raise SubmissionValidationError(
                    [f"results JSONL line {line_number}: expected an object"]
                )
            query_id = str(payload.get("query_id", "")).strip()
            if not query_id:
                raise SubmissionValidationError(
                    [f"results JSONL line {line_number}: empty query_id"]
                )
            if query_id in results:
                raise SubmissionValidationError([f"results JSONL duplicate query_id {query_id!r}"])
            answers = payload.get("answers", payload.get("hits", payload.get("results")))
            if not isinstance(answers, list):
                raise SubmissionValidationError(
                    [f"results JSONL query {query_id!r}: answers/hits list is missing"]
                )
            results[query_id] = [item for item in answers if isinstance(item, dict)]
            if len(results[query_id]) != len(answers):
                raise SubmissionValidationError(
                    [f"results JSONL query {query_id!r}: answer is not an object"]
                )
    if not results:
        raise SubmissionValidationError(["results JSONL has no queries"])
    return results


def _frame_index(value: Any, query_id: str, rank: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise SubmissionValidationError(
            [f"query {query_id!r} rank {rank}: source_frame_idx must be an integer"]
        )
    raw = str(value).strip()
    if not _FRAME_INDEX_RE.fullmatch(raw):
        raise SubmissionValidationError(
            [f"query {query_id!r} rank {rank}: source_frame_idx must be an integer"]
        )
    return int(raw)


def _answer_cells(
    query_id: str,
    answers: list[dict[str, Any]],
    corpus: Mapping[str, int],
) -> list[str]:
    errors: list[str] = []
    if len(answers) < ANSWER_COUNT:
        errors.append(
            f"query {query_id!r}: expected at least {ANSWER_COUNT} answers, got {len(answers)}"
        )
        raise SubmissionValidationError(errors)

    cells: list[str] = []
    for rank, answer in enumerate(answers[:ANSWER_COUNT], start=1):
        filename = str(answer.get("video_filename", "")).strip()
        if not filename:
            errors.append(f"query {query_id!r} rank {rank}: video_filename is missing")
            continue
        if filename not in corpus:
            errors.append(
                f"query {query_id!r} rank {rank}: video filename {filename!r} is not in corpus.csv"
            )
            continue
        if "source_frame_idx" not in answer:
            errors.append(
                f"query {query_id!r} rank {rank}: source_frame_idx is missing; "
                "keyframe_id cannot be exported"
            )
            continue
        try:
            source_frame_idx = _frame_index(answer["source_frame_idx"], query_id, rank)
        except SubmissionValidationError as exc:
            errors.extend(exc.errors)
            continue
        frame_count = corpus[filename]
        if source_frame_idx < 0 or source_frame_idx >= frame_count:
            errors.append(
                f"query {query_id!r} rank {rank}: source_frame_idx={source_frame_idx} "
                f"out of range [0, {frame_count}) for {filename!r}"
            )
            continue
        cells.append(f"{filename},{source_frame_idx}")

    if len(cells) != ANSWER_COUNT:
        raise SubmissionValidationError(errors or [f"query {query_id!r}: invalid answer count"])
    if len(set(cells)) != ANSWER_COUNT:
        raise SubmissionValidationError([f"query {query_id!r}: duplicate answer cell"])
    return cells


def _validate_result_contract(
    questions_path: Path, results_path: Path, corpus_path: Path
) -> tuple[list[str], dict[str, list[str]]]:
    questions = load_skillpixel_questions(questions_path)
    expected_ids = [item.query_id for item in questions]
    expected_set = set(expected_ids)
    corpus = _load_corpus(corpus_path)
    results = _load_result_rows(results_path)
    errors: list[str] = []
    missing = sorted(expected_set - set(results))
    extra = sorted(set(results) - expected_set)
    if missing:
        errors.append(f"missing query results: {missing}")
    if extra:
        errors.append(f"unexpected query results: {extra}")
    cells_by_query: dict[str, list[str]] = {}
    for query_id in expected_ids:
        if query_id not in results:
            continue
        try:
            cells_by_query[query_id] = _answer_cells(query_id, results[query_id], corpus)
        except SubmissionValidationError as exc:
            errors.extend(exc.errors)
    if errors:
        raise SubmissionValidationError(errors)
    return expected_ids, cells_by_query


def export_skillpixel_submission(
    questions_path: Path,
    results_path: Path,
    corpus_path: Path,
    output_path: Path,
) -> SubmissionStats:
    """Validate retrieval JSONL, write quoted CSV, and validate the written file."""
    expected_ids, cells_by_query = _validate_result_contract(
        Path(questions_path), Path(results_path), Path(corpus_path)
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
            writer.writerow(_expected_header())
            for query_id in expected_ids:
                writer.writerow([query_id] + cells_by_query[query_id])
        report = validate_submission_csv(temporary, questions_path, corpus_path)
        if not report.ok:
            raise SubmissionValidationError(list(report.errors))
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return SubmissionStats(len(expected_ids), ANSWER_COUNT, output_path)


def validate_submission_csv(
    submission_path: Path, questions_path: Path, corpus_path: Path
) -> SubmissionValidationReport:
    """Strictly parse and validate an already written organizer CSV."""
    errors: list[str] = []
    try:
        expected_ids = [item.query_id for item in load_skillpixel_questions(questions_path)]
        corpus = _load_corpus(corpus_path)
    except SubmissionValidationError as exc:
        return SubmissionValidationReport(Path(submission_path), 0, tuple(exc.errors))
    except (FileNotFoundError, ValueError) as exc:
        return SubmissionValidationReport(Path(submission_path), 0, (str(exc),))

    rows: list[list[str]] = []
    try:
        with Path(submission_path).open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle, strict=True))
    except (FileNotFoundError, csv.Error, UnicodeError) as exc:
        return SubmissionValidationReport(Path(submission_path), 0, (f"invalid CSV: {exc}",))

    if not rows:
        errors.append("submission CSV is empty")
        return SubmissionValidationReport(Path(submission_path), 0, tuple(errors))
    if rows[0] != _expected_header():
        errors.append("submission CSV header does not match query_id + answer_001..answer_100")
    seen: set[str] = set()
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != ANSWER_COUNT + 1:
            errors.append(
                f"submission row {row_number}: expected {ANSWER_COUNT + 1} columns, got {len(row)}"
            )
            continue
        query_id = row[0].strip()
        if query_id in seen:
            errors.append(f"submission duplicate query_id {query_id!r}")
        seen.add(query_id)
        if query_id not in set(expected_ids):
            errors.append(f"submission unexpected query_id {query_id!r}")
        answers: list[dict[str, Any]] = []
        for rank, cell in enumerate(row[1:], start=1):
            filename, separator, raw_index = cell.rpartition(",")
            if not separator or not filename:
                errors.append(f"query {query_id!r} rank {rank}: invalid answer cell {cell!r}")
                continue
            answers.append({"video_filename": filename, "source_frame_idx": raw_index})
        try:
            _answer_cells(query_id, answers, corpus)
        except SubmissionValidationError as exc:
            errors.extend(exc.errors)
    missing = sorted(set(expected_ids) - seen)
    if missing:
        errors.append(f"submission missing query_id(s): {missing}")
    if len(rows) - 1 != len(expected_ids):
        errors.append(f"submission query row count {len(rows) - 1} != expected {len(expected_ids)}")
    return SubmissionValidationReport(Path(submission_path), len(rows) - 1, tuple(errors))


def write_results_jsonl(results: Mapping[str, Sequence[SkillPixelHit]], output_path: Path) -> Path:
    """Persist retrieval hits in the exporter input format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for query_id, hits in results.items():
            payload = {"query_id": query_id, "answers": [hit.to_dict() for hit in hits]}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output_path

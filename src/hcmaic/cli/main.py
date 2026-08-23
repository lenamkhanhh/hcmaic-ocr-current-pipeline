"""hcmaic CLI.

Commands:
  ingest-video  --input <video file|dir> --output <dataset>
                [--interval 2.0] [--max-frames 500] [--video-id ID] [--force]
  ingest-raw    --input <raw video file|dir> --output <generated dataset>
                [--stride-frames 10|12] [--force]
  validate-data --input <dataset>
  build-index   --input <dataset> --output <artifacts>
                [--provider mock|clip] [--index exact-numpy|faiss]
  merge-ocr     --shard <output-dir> ... --output <merged-dir>
  index-ocr-es  --artifact <merged-dir> --url <es-url> --index <index>
  search-ocr-es --url <es-url> --index <index> --query "<text>"
  build-skillpixel-index --input <raw-generated dataset> --output <artifacts>
                [--provider siglip2|clip] [--allow-network]
  retrieve-skillpixel --index <artifacts> --questions <questions.csv>
                --results <results.jsonl> [--provider auto|siglip2|clip]
  export-skillpixel --queries <questions.csv> --results <results.jsonl>
                --corpus <corpus.csv> --output <submission.csv>
  search        --index <artifacts> --query "<text>" [--top-k 10] [--video-id V1,V2]
  serve         --index <artifacts> [--host 127.0.0.1] [--port 8000] [--data-root <path>]
  serve-dual    --artifacts <dual-merge> [--image-root <dir>] [--video-root <dir>]
                [--media-manifest <jsonl>] [--media-cache-root <dir>]
                [--ui-dir <bfe-dist>] [--host 127.0.0.1] [--port 8000]
  prepare-groundtruth-review --proposals <jsonl> --inventory <jsonl>
                --pts-dir <dir> --output <dir> [--raw-root <dir>] [--materialize]
  serve-groundtruth-review --review-root <dir> [--host 127.0.0.1] [--port 8000]
  evaluate      --index <artifacts> --queries <queries.jsonl> --qrels <qrels.jsonl> [--out <dir>]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def _cmd_ingest(args: argparse.Namespace) -> int:
    from hcmaic.ingestion.video import IngestError, ingest_dataset

    try:
        results, failures = ingest_dataset(
            Path(args.input),
            Path(args.output),
            video_id=args.video_id,
            interval_s=args.interval,
            max_frames=args.max_frames,
            force=args.force,
        )
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for result in results:
        info = result.info
        print(
            f"Ingested {info.video_id}: {result.n_kept} keyframe(s) "
            f"({result.n_candidates} candidates, {result.n_duplicates} "
            f"near-duplicates dropped), {info.width}x{info.height} "
            f"@ {info.fps:.2f} fps, {info.duration_s:.1f}s, "
            f"backend={info.backend}."
        )
        for warning in result.warnings:
            print(f"  warn: {warning}")
    for failure in failures:
        print(f"  FAILED {failure['file']}: {failure['error']}", file=sys.stderr)
    print(f"Dataset: {args.output}  (report: {Path(args.output) / 'ingest_report.json'})")
    if results:
        print(
            f"Next: uv run hcmaic validate-data --input {args.output} && "
            f"uv run hcmaic build-index --input {args.output} --output <artifacts>"
        )
    return 0 if not failures else 1


def _cmd_ingest_raw(args: argparse.Namespace) -> int:
    from hcmaic.skillpixel.raw import RawIngestError, ingest_raw_videos, validate_raw_dataset

    try:
        report = ingest_raw_videos(
            Path(args.input),
            Path(args.output),
            stride_frames=args.stride_frames,
            force=args.force,
        )
        stats = validate_raw_dataset(Path(args.output))
    except RawIngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_videos": report.n_videos,
                "n_frames": stats.n_frames,
                "sampling_policy": report.sampling_policy,
                "dataset_manifest": str(Path(args.output) / "dataset_manifest.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from hcmaic.ingestion.validator import validate_dataset, write_validation_report
    from hcmaic.ingestion.video import SUPPORTED_EXTENSIONS

    root = Path(args.input)
    raw_source = root.is_file() and root.suffix.lower() in SUPPORTED_EXTENSIONS
    if root.is_dir():
        raw_source = (
            any(
                path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
                for path in root.iterdir()
            )
            and not (root / "dataset_manifest.json").is_file()
        )
    if raw_source:
        from hcmaic.skillpixel.raw import validate_raw_video_source

        source_report = validate_raw_video_source(root)
        out = Path(args.report) if args.report else root / "data-validation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(source_report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"Validated raw source {root}: {source_report.n_videos} videos, "
            f"{len(source_report.errors)} error(s), {len(source_report.warnings)} warning(s)."
        )
        for error in source_report.errors[: args.max_warnings]:
            print(f"  ERROR {error}")
        print(f"Report: {out}")
        return 0 if source_report.ok else 1
    if not root.is_dir():
        print(f"error: dataset root {root} is not a directory", file=sys.stderr)
        return 2
    report = validate_dataset(root, check_images=not args.skip_image_check)
    out = Path(args.report) if args.report else root / "validation_report.json"
    write_validation_report(report, out)
    print(
        f"Validated {root}: {report.n_videos} videos, {report.n_frames} frames, "
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)."
    )
    for issue in report.errors:
        print(f"  ERROR [{issue.code}] {issue.message}")
    for issue in report.warnings[: args.max_warnings]:
        print(f"  warn  [{issue.code}] {issue.message}")
    hidden = len(report.warnings) - args.max_warnings
    if hidden > 0:
        print(f"  ... {hidden} more warning(s) in {out}")
    print(f"Report: {out}")
    return 0 if report.ok else 1


def _cmd_build_index(args: argparse.Namespace) -> int:
    from hcmaic.embedding.base import get_provider
    from hcmaic.indexing.artifacts import build_index_artifacts
    from hcmaic.ingestion.catalog import build_catalog
    from hcmaic.ingestion.validator import validate_dataset

    root = Path(args.input)
    report = validate_dataset(root, check_images=True)
    if not report.ok:
        print(
            f"error: dataset has {len(report.errors)} validation error(s); "
            f"run 'hcmaic validate-data --input {root}' for details.",
            file=sys.stderr,
        )
        return 1
    catalog = build_catalog(root)
    provider = get_provider(args.provider)
    out_dir = build_index_artifacts(
        root, catalog, provider, Path(args.output), index_provider=args.index
    )
    print(
        f"Built index artifacts in {out_dir}: {len(catalog)} frames, "
        f"dim {provider.dimension}, provider {provider.version}, "
        f"index {args.index}."
    )
    return 0


def _cmd_merge_ocr(args: argparse.Namespace) -> int:
    from hcmaic.ingestion.ocr_merge import merge_ocr_shards

    summary = merge_ocr_shards(
        [Path(value) for value in args.shards],
        Path(args.output),
        batch_size=args.batch_size,
        require_postflight_green=not args.allow_legacy_postflight,
        allow_report_failures=args.allow_report_failures,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _cmd_index_ocr_es(args: argparse.Namespace) -> int:
    from hcmaic.retrieval.ocr_elasticsearch import bulk_index_ocr, make_elasticsearch_client

    client = make_elasticsearch_client(
        args.url,
        api_key_env=args.api_key_env,
        username_env=args.username_env,
        password_env=args.password_env,
        allow_anonymous_local=args.allow_anonymous_local,
    )
    summary = bulk_index_ocr(
        client,
        Path(args.artifact),
        index_name=args.index,
        batch_size=args.batch_size,
        replace=args.replace,
        refresh=not args.no_refresh,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _cmd_search_ocr_es(args: argparse.Namespace) -> int:
    from hcmaic.retrieval.ocr_elasticsearch import make_elasticsearch_client, search_ocr

    client = make_elasticsearch_client(
        args.url,
        api_key_env=args.api_key_env,
        username_env=args.username_env,
        password_env=args.password_env,
        allow_anonymous_local=args.allow_anonymous_local,
    )
    video_ids = [value for value in (args.video_id or "").split(",") if value]
    hits = search_ocr(
        client,
        args.index,
        args.query,
        top_k=args.top_k,
        video_ids=video_ids or None,
        include_low_conf=not args.exclude_low_conf,
    )
    print(json.dumps([asdict(hit) for hit in hits], ensure_ascii=False))
    return 0


def _cmd_build_skillpixel_index(args: argparse.Namespace) -> int:
    import json

    from hcmaic.embedding.factory import get_real_visual_provider
    from hcmaic.skillpixel.index import build_skillpixel_index

    try:
        model_kwargs: dict[str, str] = {}
        if args.model_path:
            model_key = {
                "siglip2": "siglip2_model",
                "clip": "clip_model",
                "jina-clip-v2": "jina_model",
            }[args.provider]
            model_kwargs[model_key] = args.model_path
        provider, selection = get_real_visual_provider(
            prefer=args.provider,
            device=args.device,
            local_files_only=not args.allow_network,
            batch_size=args.batch_size,
            allow_fallback=not args.strict_provider,
            **model_kwargs,
        )
        index = build_skillpixel_index(Path(args.input), Path(args.output), provider)
        provider_report_path = Path(args.output) / "provider_report.json"
        provider_report = json.loads(provider_report_path.read_text(encoding="utf-8"))
        provider_report.update(
            {
                "requested_provider": args.provider,
                "selection": selection,
                "fallback": selection.get("fallback"),
            }
        )
        provider_report_path.write_text(
            json.dumps(provider_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_dir": str(args.output),
                "n_frames": index.size,
                "dimension": index.dimension,
                "provider": provider.info(),
                "selection": selection,
                "index_provider": "faiss-flat-ip",
            },
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_retrieve_skillpixel(args: argparse.Namespace) -> int:
    from hcmaic.embedding.factory import get_real_visual_provider
    from hcmaic.skillpixel.index import load_skillpixel_index
    from hcmaic.skillpixel.retrieval import SkillPixelRetriever, load_skillpixel_questions
    from hcmaic.skillpixel.submission import write_results_jsonl

    index = load_skillpixel_index(Path(args.index))
    expected_provider = str(index.provider_info.get("provider", ""))
    prefer = expected_provider if args.provider == "auto" else args.provider
    if prefer not in {"siglip2", "clip", "jina-clip-v2"}:
        print(f"error: unsupported index provider {expected_provider!r}", file=sys.stderr)
        return 2
    try:
        model_kwargs: dict[str, str] = {}
        if args.model_path:
            model_key = {
                "siglip2": "siglip2_model",
                "clip": "clip_model",
                "jina-clip-v2": "jina_model",
            }[prefer]
            model_kwargs[model_key] = args.model_path
        provider, selection = get_real_visual_provider(
            prefer=prefer,
            device=args.device,
            local_files_only=not args.allow_network,
            revision=args.revision or index.provider_info.get("model_revision"),
            batch_size=args.batch_size,
            allow_fallback=not args.strict_provider,
            **model_kwargs,
        )
        retriever = SkillPixelRetriever(index, provider)
        questions_path = Path(args.questions)
        questions = load_skillpixel_questions(questions_path)
        tkis = retriever.search_text_queries(
            [(item.query_id, item.text) for item in questions if item.task == "TKIS"],
            top_k=args.top_k,
        )
        vkis = retriever.search_image_queries(
            [
                (
                    item.query_id,
                    Path(item.query_image)
                    if Path(item.query_image).is_absolute()
                    else questions_path.parent / item.query_image,
                )
                for item in questions
                if item.task == "VKIS"
            ],
            top_k=args.top_k,
        )
        ordered = {
            item.query_id: (tkis if item.task == "TKIS" else vkis)[item.query_id]
            for item in questions
        }
        write_results_jsonl(ordered, Path(args.results))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "results": str(args.results),
                "n_queries": len(ordered),
                "top_k": args.top_k,
                "provider": provider.info(),
                "selection": selection,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_export_skillpixel(args: argparse.Namespace) -> int:
    from hcmaic.skillpixel.submission import (
        SubmissionValidationError,
        export_skillpixel_submission,
        validate_submission_csv,
    )

    try:
        stats = export_skillpixel_submission(
            Path(args.queries),
            Path(args.results),
            Path(args.corpus),
            Path(args.output),
        )
    except (FileNotFoundError, SubmissionValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report = validate_submission_csv(Path(args.output), Path(args.queries), Path(args.corpus))
    print(
        json.dumps(
            {
                "submission": str(stats.output_path),
                "n_queries": stats.n_queries,
                "answers_per_query": stats.answers_per_query,
                "valid": report.ok,
                "errors": list(report.errors),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.ok else 1


def _cmd_provider_doctor(args: argparse.Namespace) -> int:
    from hcmaic.embedding.registry import provider_doctor

    report = provider_doctor(args.provider, device=args.device, revision=args.revision)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _kis_runtime(args: argparse.Namespace):
    from hcmaic.runtime.kis import load_kis_runtime

    return load_kis_runtime(
        Path(args.index),
        provider=args.provider,
        device=args.device,
        local_files_only=not args.allow_network,
        batch_size=args.batch_size,
        ocr_artifact=Path(args.ocr_artifact) if args.ocr_artifact else None,
        ocr_es_url=args.ocr_es_url,
        ocr_es_index=args.ocr_es_index,
        ocr_es_manifest=Path(args.ocr_es_manifest) if args.ocr_es_manifest else None,
        ocr_es_api_key_env=args.ocr_es_api_key_env,
        ocr_es_username_env=args.ocr_es_username_env,
        ocr_es_password_env=args.ocr_es_password_env,
        ocr_es_include_low_conf=args.ocr_es_include_low_conf,
        object_artifact=Path(args.object_artifact) if args.object_artifact else None,
        object_sidecar=Path(args.object_sidecar) if args.object_sidecar else None,
        allow_engineering_proxy=args.allow_engineering_proxy,
        asr_artifact=Path(args.asr_artifact) if args.asr_artifact else None,
        asr_window_artifact=(Path(args.asr_window_artifact) if args.asr_window_artifact else None),
        asr_keyframe_manifest_hash=args.asr_keyframe_manifest_hash,
        asr_enabled=args.asr_enabled,
    )


def _queries_to_kis(questions_path: Path, *, top_k: int):
    from hcmaic.contracts.kis import KISQuery
    from hcmaic.skillpixel.retrieval import load_skillpixel_questions

    questions = load_skillpixel_questions(questions_path)
    queries = []
    for item in questions:
        image_path = Path(item.query_image)
        if item.task == "VKIS" and not image_path.is_absolute():
            image_path = questions_path.parent / image_path
        queries.append(
            KISQuery(
                query_id=item.query_id,
                task=item.task,
                text=item.text or None,
                image_path=image_path if item.task == "VKIS" else None,
                top_k=top_k,
            )
        )
    return questions, queries


def _write_kis_results(outputs, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for query_id, output in outputs.items():
            handle.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "task": output.query.task,
                        "answers": [result.to_dict() for result in output.results],
                        "executed_channels": list(output.executed_channels),
                        "unavailable_channels": output.unavailable_channels,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return output_path


def _cmd_search_kis(args: argparse.Namespace) -> int:
    from hcmaic.contracts.kis import KISQuery

    try:
        runtime = _kis_runtime(args)
        image_path = Path(args.image) if args.image else None
        query = KISQuery(
            query_id=args.query_id,
            task=args.task,
            text=args.query,
            image_path=image_path,
            top_k=args.top_k,
        )
        output = runtime.search(query)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "query_id": query.query_id,
                "task": query.task,
                "results": [result.to_dict() for result in output.results],
                "executed_channels": list(output.executed_channels),
                "unavailable_channels": output.unavailable_channels,
                "channel_status": runtime.channel_status,
                "channel_contracts": runtime.channel_contracts,
                "provider": runtime.provider.info(),
                "quality_status": "UNVALIDATED_ON_HCMAIC",
            },
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_retrieve_kis(args: argparse.Namespace) -> int:
    try:
        runtime = _kis_runtime(args)
        questions_path = Path(args.questions)
        _, queries = _queries_to_kis(questions_path, top_k=args.top_k)
        outputs = runtime.search_queries(queries)
        _write_kis_results(outputs, Path(args.results))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "results": str(args.results),
                "n_queries": len(outputs),
                "top_k": args.top_k,
                "provider": runtime.provider.info(),
                "channels": runtime.channel_status,
                "quality_status": "UNVALIDATED_ON_HCMAIC",
            },
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_export_kis(args: argparse.Namespace) -> int:
    from hcmaic.skillpixel.submission import (
        SubmissionValidationError,
        export_skillpixel_submission,
        validate_submission_csv,
    )

    try:
        runtime = _kis_runtime(args)
        questions_path = Path(args.questions)
        _, queries = _queries_to_kis(questions_path, top_k=100)
        outputs = runtime.search_queries(queries)
        results_path = Path(args.output).with_suffix(".kis-results.jsonl")
        _write_kis_results(outputs, results_path)
        try:
            stats = export_skillpixel_submission(
                questions_path,
                results_path,
                Path(args.corpus),
                Path(args.output),
            )
        finally:
            results_path.unlink(missing_ok=True)
        validation = validate_submission_csv(Path(args.output), questions_path, Path(args.corpus))
        if not validation.ok:
            raise SubmissionValidationError(list(validation.errors))
    except (FileNotFoundError, RuntimeError, ValueError, SubmissionValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "submission": str(stats.output_path),
                "n_queries": stats.n_queries,
                "answers_per_query": stats.answers_per_query,
                "quality_status": "UNVALIDATED_ON_HCMAIC",
            },
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_serve_kis(args: argparse.Namespace) -> int:
    import uvicorn

    from hcmaic.api.kis_app import create_kis_app

    try:
        runtime = _kis_runtime(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    app = create_kis_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cmd_benchmark_skillpixel(args: argparse.Namespace) -> int:
    from hcmaic.benchmark.skillpixel import (
        SkillPixelBenchmarkConfig,
        run_skillpixel_benchmark,
    )

    qrels: dict[str, object] | None = None
    if args.qrels:
        from hcmaic.evaluation.kis import load_kis_qrels

        qrel_set = load_kis_qrels(Path(args.qrels), source=args.qrels_source)
        qrels = {
            query_id: sorted(qrel.relevant_answer_cells)
            for query_id, qrel in qrel_set.qrels.items()
        }
    config = SkillPixelBenchmarkConfig(
        raw_root=Path(args.raw),
        index_dir=Path(args.index),
        questions_path=Path(args.questions),
        corpus_path=Path(args.corpus),
        output_dir=Path(args.out),
        top_k=args.top_k,
        qrels=qrels,
        qrels_source=args.qrels_source if args.qrels else None,
        ocr_artifact=Path(args.ocr_artifact) if args.ocr_artifact else None,
        object_artifact=Path(args.object_artifact) if args.object_artifact else None,
    )
    try:
        rows, paths = run_skillpixel_benchmark(
            config,
            provider_ids=tuple(item.strip() for item in args.providers.split(",") if item.strip()),
            device=args.device,
            batch_size=args.batch_size,
            allow_network=args.allow_network,
            build_missing=not args.no_build_missing,
            model_paths={
                provider: path
                for provider, path in {
                    "clip": args.clip_model,
                    "siglip2": args.siglip2_model,
                    "jina-clip-v2": args.jina_model,
                }.items()
                if path
            },
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    validated = [row for row in rows if row.get("status") == "validated-local"]
    print(
        json.dumps(
            {
                "benchmark_results": str(paths["csv"]),
                "benchmark_report": str(paths["report"]),
                "run_manifest": str(paths["manifest"]),
                "n_rows": len(rows),
                "n_validated": len(validated),
                "quality_status": (
                    "VALIDATED_ON_SKILLPIXEL_QRELS" if qrels else "UNVALIDATED_ON_SKILLPIXEL_QRELS"
                ),
            },
            ensure_ascii=False,
        )
    )
    has_visual = any(
        row.get("kind") == "visual" and row.get("status") == "validated-local" for row in rows
    )
    return 0 if has_visual else 2


def _cmd_package_kaggle_skillpixel(args: argparse.Namespace) -> int:
    from hcmaic.benchmark.kaggle import KagglePackageConfig, build_kaggle_package

    try:
        outputs = build_kaggle_package(
            KagglePackageConfig(
                output_dir=Path(args.out),
                raw_input=Path(args.raw_input),
                questions_path=Path(args.questions),
                corpus_path=Path(args.corpus),
                index_dir=Path(args.index) if args.index else None,
                max_file_bytes=args.max_file_bytes,
            )
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False))
    return 0


def _cmd_benchmark_kis(args: argparse.Namespace) -> int:
    from hcmaic.evaluation.kis import (
        evaluate_kis_runtime,
        load_kis_qrels,
        run_kis_ablation,
    )

    try:
        runtime = _kis_runtime(args)
        questions_path = Path(args.questions)
        questions, queries = _queries_to_kis(questions_path, top_k=args.top_k)
        qrels = load_kis_qrels(Path(args.qrels), source=args.qrels_source) if args.qrels else None
        baseline, per_query = evaluate_kis_runtime(
            runtime,
            questions,
            qrels,
            top_k=args.top_k,
            query_root=questions_path.parent,
            frame_tolerance=args.frame_tolerance,
        )
        ablation = run_kis_ablation(
            runtime,
            questions,
            qrels,
            top_k=args.top_k,
            query_root=questions_path.parent,
        )
        report = {
            "baseline": baseline,
            "ablation": ablation,
            "provider": runtime.provider.info(),
            "index": runtime.index.index_manifest,
            "channels": runtime.channel_status,
        }
        if args.out:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "kis_benchmark.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_kis_results(
                runtime.search_queries(queries), out_dir / "kis_benchmark_results.jsonl"
            )
            (out_dir / "kis_per_query.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_query),
                encoding="utf-8",
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _cmd_scale_benchmark(args: argparse.Namespace) -> int:
    from hcmaic.indexing.scale_benchmark import (
        ScaleBenchmarkConfig,
        run_scale_benchmark,
    )

    config = ScaleBenchmarkConfig(
        vector_count=args.vectors,
        dimension=args.dimension,
        query_count=args.queries,
        top_k=args.top_k,
        seed=args.seed,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    report = run_scale_benchmark(config)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from hcmaic.benchmark.runner import run_benchmark

    outputs = run_benchmark(Path(args.config), Path(args.out))
    print("Benchmark complete (proxy/plumbing evidence unless real inputs are supplied).")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from hcmaic.contracts.models import SearchRequest
    from hcmaic.retrieval.service import load_service

    service = load_service(
        Path(args.index),
        dataset_root=Path(args.data_root) if args.data_root else None,
        index_provider=args.index_provider,
    )
    filters = {}
    if args.video_id:
        filters["video_ids"] = args.video_id
    request = SearchRequest(
        query_id=args.query_id, text=args.query, top_k=args.top_k, filters=filters
    )
    results = service.search(request)
    if args.json:
        print(json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False))
    else:
        print(f"Query: {args.query!r}  top_k={args.top_k}  index={service.index_version}")
        if not results:
            print("(no results)")
        for r in results:
            print(
                f"  #{r.rank:<3} {r.frame_id:<20} score={r.final_score:.4f} "
                f"t={r.timestamp_ms}ms idx={r.frame_idx}"
            )
    return 0


def _dual_service_args(args: argparse.Namespace):
    from hcmaic.retrieval.dual_visual import load_dual_visual_service

    visual_indexes = None
    if getattr(args, "visual_indexes", None) is not None:
        visual_indexes = tuple(
            value.strip() for value in str(args.visual_indexes).split(",") if value.strip()
        )
    return load_dual_visual_service(
        Path(args.artifacts),
        visual_indexes=visual_indexes,
        image_root=Path(args.image_root) if args.image_root else None,
        video_root=Path(args.video_root) if args.video_root else None,
        siglip_device=args.siglip_device,
        qwen_device=args.qwen_device,
        local_files_only=not args.allow_network,
        siglip_model_path=Path(args.siglip_model_path) if args.siglip_model_path else None,
        model_path=Path(args.qwen_model_path) if args.qwen_model_path else None,
        object_sidecar=Path(args.object_sidecar) if args.object_sidecar else None,
        object_enabled=bool(args.object_enabled),
        allow_engineering_proxy=bool(args.allow_engineering_proxy),
        ocr_es_url=args.ocr_es_url,
        ocr_es_index=args.ocr_es_index,
        ocr_es_manifest=Path(args.ocr_es_manifest) if args.ocr_es_manifest else None,
        ocr_es_include_low_conf=not bool(args.ocr_es_exclude_low_conf),
        ocr_es_api_key_env=args.ocr_es_api_key_env,
        ocr_es_username_env=args.ocr_es_username_env,
        ocr_es_password_env=args.ocr_es_password_env,
    )


def _cmd_preflight_dual(args: argparse.Namespace) -> int:
    from hcmaic.retrieval.dual_visual import (
        emit_local_runtime_manifest,
        load_dual_visual_artifacts,
    )

    artifacts = load_dual_visual_artifacts(Path(args.artifacts))
    output = Path(args.out) if args.out else Path(args.artifacts) / "local_runtime_manifest.json"
    payload = emit_local_runtime_manifest(artifacts, output, include_file_hashes=not args.no_hash)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Local runtime manifest: {output}")
    return 0


def _cmd_search_dual(args: argparse.Namespace) -> int:
    service = _dual_service_args(args)
    video_ids = (
        [value.strip() for value in args.video_id.split(",") if value.strip()]
        if args.video_id
        else None
    )
    if args.image:
        results = service.search_image(
            args.query_id,
            Path(args.image),
            top_k=args.top_k,
            video_ids=video_ids,
        )
    else:
        results = service.search_text(
            args.query_id,
            args.query,
            top_k=args.top_k,
            video_ids=video_ids,
            object_query=args.object_query,
        )
    if args.json:
        print(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))
        return 0
    print(
        f"Query={args.query or args.image!r} top_k={args.top_k} "
        f"identity=frame_uid quality=UNVALIDATED"
    )
    for item in results:
        print(
            f"  #{item.rank:<3} {item.frame_uid:<24} score={item.fused_score:.6f} "
            f"video={item.video_id} source_frame_idx={item.source_frame_idx} "
            f"channels={','.join(item.executed_channels)}"
        )
    return 0


def _cmd_serve_dual(args: argparse.Namespace) -> int:
    import uvicorn

    from hcmaic.api.dual_app import create_dual_app
    from hcmaic.retrieval.asr_elasticsearch import ASRElasticsearchConfig

    asr_config = None
    asr_configured = any(
        getattr(args, name, None) is not None
        for name in (
            "asr_es_url",
            "asr_es_index",
            "asr_es_edges",
            "asr_es_lookup",
            "asr_es_timeout",
            "asr_es_top_n",
            "asr_es_mode",
            "asr_es_fuzziness",
            "asr_es_rank_constant",
            "asr_es_api_key_env",
            "asr_es_username_env",
            "asr_es_password_env",
        )
    ) or bool(args.asr_es_enabled or args.asr_es_policy_enabled)
    if asr_configured:
        asr_kwargs: dict[str, object] = {
            "url": args.asr_es_url,
            "index": args.asr_es_index or "aic26_transcripts_v1",
            "enabled": bool(args.asr_es_enabled),
            "policy_enabled": bool(args.asr_es_policy_enabled),
        }
        optional_asr_args = {
            "edges_path": args.asr_es_edges,
            "lookup_path": args.asr_es_lookup,
            "timeout_s": args.asr_es_timeout,
            "segment_top_n": args.asr_es_top_n,
            "mode": args.asr_es_mode,
            "fuzziness": args.asr_es_fuzziness,
            "rank_constant": args.asr_es_rank_constant,
            "api_key_env": args.asr_es_api_key_env,
            "username_env": args.asr_es_username_env,
            "password_env": args.asr_es_password_env,
        }
        for key, value in optional_asr_args.items():
            if value is not None:
                asr_kwargs[key] = (
                    Path(value) if key in {"edges_path", "lookup_path"} else value
                )
        asr_config = ASRElasticsearchConfig(**asr_kwargs)

    app = create_dual_app(
        Path(args.artifacts),
        visual_indexes=tuple(
            value.strip() for value in str(args.visual_indexes).split(",") if value.strip()
        )
        if args.visual_indexes is not None
        else None,
        asr_only=bool(args.asr_only),
        image_root=Path(args.image_root) if args.image_root else None,
        video_root=Path(args.video_root) if args.video_root else None,
        siglip_device=args.siglip_device,
        qwen_device=args.qwen_device,
        local_files_only=not args.allow_network,
        siglip_model_path=Path(args.siglip_model_path) if args.siglip_model_path else None,
        qwen_model_path=Path(args.qwen_model_path) if args.qwen_model_path else None,
        media_manifest=Path(args.media_manifest) if args.media_manifest else None,
        media_cache_root=Path(args.media_cache_root) if args.media_cache_root else None,
        media_http_hosts=set(args.media_http_host or []),
        object_sidecar=Path(args.object_sidecar) if args.object_sidecar else None,
        object_enabled=bool(args.object_enabled),
        allow_engineering_proxy=bool(args.allow_engineering_proxy),
        asr_config=asr_config,
        ocr_es_url=args.ocr_es_url,
        ocr_es_index=args.ocr_es_index,
        ocr_es_manifest=Path(args.ocr_es_manifest) if args.ocr_es_manifest else None,
        ocr_es_include_low_conf=not bool(args.ocr_es_exclude_low_conf),
        ocr_es_api_key_env=args.ocr_es_api_key_env,
        ocr_es_username_env=args.ocr_es_username_env,
        ocr_es_password_env=args.ocr_es_password_env,
        ui_dir=Path(args.ui_dir) if args.ui_dir else None,
    )
    print(f"Serving dual retrieval on http://{args.host}:{args.port}/ (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from hcmaic.api.app import create_app

    app = create_app(
        Path(args.index),
        dataset_root=Path(args.data_root) if args.data_root else None,
        index_provider=args.index_provider,
    )
    print(f"Serving on http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_prepare_groundtruth_review(args: argparse.Namespace) -> int:
    from hcmaic.groundtruth.review import ReviewBundleError, prepare_review_bundle

    def progress(message: str) -> None:
        print(f"[review] {message}", flush=True)

    try:
        manifest = prepare_review_bundle(
            Path(args.proposals),
            Path(args.inventory),
            Path(args.pts_dir),
            Path(args.output),
            fps=args.fps,
            window_before_s=args.window_before_s,
            window_after_s=args.window_after_s,
            raw_root=Path(args.raw_root) if args.raw_root else None,
            materialize=args.materialize,
            jpeg_quality=args.jpeg_quality,
            progress=progress,
        )
    except ReviewBundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(
        f"Review bundle: {args.output} ({manifest['item_count']} items, "
        f"{manifest['frame_count']} selected frames, "
        f"materialized={bool(manifest['unique_materialized_frame_count'])})"
    )
    return 0


def _cmd_serve_groundtruth_review(args: argparse.Namespace) -> int:
    import uvicorn

    from hcmaic.api.groundtruth_review import create_groundtruth_review_app

    app = create_groundtruth_review_app(Path(args.review_root))
    print(f"Serving review UI on http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from hcmaic.evaluation.evaluator import (
        evaluate,
        format_summary,
        load_qrels,
        load_queries,
        write_reports,
    )
    from hcmaic.retrieval.service import load_service

    service = load_service(
        Path(args.index),
        dataset_root=Path(args.data_root) if args.data_root else None,
    )
    queries = load_queries(Path(args.queries))
    qrels = load_qrels(Path(args.qrels))
    report, per_query = evaluate(service, queries, qrels, top_k=args.top_k)
    out_dir = Path(args.out) if args.out else Path(args.index) / "evaluation"
    report_path, per_query_path = write_reports(report, per_query, out_dir)
    print(format_summary(report))
    print(f"Reports: {report_path} , {per_query_path}")
    return 0


def _add_kis_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", required=True)
    parser.add_argument(
        "--provider", choices=["auto", "siglip2", "clip", "jina-clip-v2"], default="auto"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--ocr-artifact")
    parser.add_argument(
        "--ocr-es-url",
        help="Explicit Elasticsearch URL; credentials must come from environment variables",
    )
    parser.add_argument("--ocr-es-index")
    parser.add_argument(
        "--ocr-es-manifest",
        help="Merged ocr_manifest.json (or its artifact directory) for provenance validation",
    )
    parser.add_argument("--ocr-es-api-key-env", default="ELASTIC_API_KEY")
    parser.add_argument("--ocr-es-username-env")
    parser.add_argument("--ocr-es-password-env")
    parser.add_argument(
        "--ocr-es-exclude-low-conf",
        dest="ocr_es_include_low_conf",
        action="store_false",
        help="Exclude OCR rows marked LOW_CONF or EMPTY from ES retrieval",
    )
    parser.set_defaults(ocr_es_include_low_conf=True)
    parser.add_argument("--object-artifact")
    parser.add_argument("--object-sidecar")
    parser.add_argument("--asr-artifact")
    parser.add_argument("--asr-window-artifact")
    parser.add_argument("--asr-keyframe-manifest-hash")
    parser.add_argument("--asr-enabled", action="store_true")
    parser.add_argument("--allow-engineering-proxy", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hcmaic", description="HCMAIC keyframe-search MVP")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "ingest-video",
        help="Extract keyframes from raw videos (MP4/MKV/AVI/MOV) into a dataset",
    )
    p.add_argument("--input", required=True, help="Video file or directory of videos")
    p.add_argument("--output", required=True, help="Dataset directory to create/extend")
    p.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Sampling interval in seconds (default: 2.0)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum keyframes per video (default: 500)",
    )
    p.add_argument("--video-id", help="Override the video id (single-file ingest only)")
    p.add_argument(
        "--force",
        action="store_true",
        help="Replace keyframes/mapping if the video was already ingested",
    )
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser(
        "ingest-raw",
        help="Extract dense source-frame-stride images from raw videos",
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stride-frames", type=int, choices=[10, 12], default=10)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_ingest_raw)

    p = sub.add_parser("validate-data", help="Validate a dataset directory")
    p.add_argument("--input", required=True)
    p.add_argument("--report", help="Path for validation_report.json")
    p.add_argument("--skip-image-check", action="store_true")
    p.add_argument("--max-warnings", type=int, default=10)
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("build-index", help="Validate, embed, and write artifacts")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--provider",
        choices=["mock", "clip", "siglip2", "jina-clip-v2"],
        default="mock",
    )
    p.add_argument(
        "--index",
        choices=["exact-numpy", "faiss", "faiss-hnsw"],
        default="exact-numpy",
    )
    p.set_defaults(func=_cmd_build_index)

    p = sub.add_parser(
        "merge-ocr",
        help="Validate and stream-merge full-shard DeepSolo/PARSeq OCR artifacts",
    )
    p.add_argument(
        "--shard",
        dest="shards",
        action="append",
        required=True,
        help="Shard output directory; repeat once per shard",
    )
    p.add_argument("--output", required=True, help="New merged OCR artifact directory")
    p.add_argument("--batch-size", type=int, default=50_000)
    p.add_argument(
        "--allow-legacy-postflight",
        action="store_true",
        help="Allow a missing postflight status; use only for explicitly audited legacy output",
    )
    p.add_argument(
        "--allow-report-failures",
        action="store_true",
        help="Allow source COMPLETE_REPORT_FAILED artifacts and mark the merge partial",
    )
    p.set_defaults(func=_cmd_merge_ocr)

    p = sub.add_parser(
        "index-ocr-es",
        help="Create/update one versioned Elasticsearch OCR index from a merged artifact",
    )
    p.add_argument("--artifact", required=True)
    p.add_argument(
        "--url",
        required=True,
        help="Elasticsearch URL without credentials/query strings",
    )
    p.add_argument("--index", required=True)
    p.add_argument("--api-key-env", default="ELASTIC_API_KEY")
    p.add_argument("--username-env")
    p.add_argument("--password-env")
    p.add_argument(
        "--allow-anonymous-local",
        action="store_true",
        help="Allow unauthenticated HTTP only for localhost/127.0.0.1; never use for remote ES",
    )
    p.add_argument("--batch-size", type=int, default=1_000)
    p.add_argument("--replace", action="store_true", help="Delete/recreate exactly this index")
    p.add_argument("--no-refresh", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Read/convert rows without ES mutation")
    p.set_defaults(func=_cmd_index_ocr_es)

    p = sub.add_parser(
        "search-ocr-es", help="Search Elasticsearch OCR and collapse crops to frames"
    )
    p.add_argument(
        "--url",
        required=True,
        help="Elasticsearch URL without credentials/query strings",
    )
    p.add_argument("--index", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--video-id", help="Restrict to comma-separated video ids")
    p.add_argument("--exclude-low-conf", action="store_true")
    p.add_argument("--api-key-env", default="ELASTIC_API_KEY")
    p.add_argument("--username-env")
    p.add_argument("--password-env")
    p.add_argument(
        "--allow-anonymous-local",
        action="store_true",
        help="Allow unauthenticated HTTP only for localhost/127.0.0.1; never use for remote ES",
    )
    p.set_defaults(func=_cmd_search_ocr_es)

    p = sub.add_parser(
        "build-skillpixel-index",
        help="Build a real-provider SkillPixel FAISS FlatIP index from raw frames",
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--provider", choices=["siglip2", "clip", "jina-clip-v2"], default="siglip2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--model-path",
        help="Explicit local path or model ID for the selected provider",
    )
    p.add_argument(
        "--allow-network",
        action="store_true",
        help="Permit model fetches; default is local-files-only",
    )
    p.add_argument(
        "--strict-provider",
        action="store_true",
        help="Fail if the requested provider is unavailable; never use a fallback",
    )
    p.set_defaults(func=_cmd_build_skillpixel_index)

    p = sub.add_parser(
        "retrieve-skillpixel",
        help="Run batched TKIS/VKIS retrieval from a persisted SkillPixel index",
    )
    p.add_argument("--index", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--results", required=True)
    p.add_argument(
        "--provider",
        choices=["auto", "siglip2", "clip", "jina-clip-v2"],
        default="auto",
    )
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--revision")
    p.add_argument(
        "--model-path",
        help="Explicit local path or model ID matching the persisted index provider",
    )
    p.add_argument("--allow-network", action="store_true")
    p.add_argument(
        "--strict-provider",
        action="store_true",
        help="Fail if the requested provider is unavailable; never use a fallback",
    )
    p.set_defaults(func=_cmd_retrieve_skillpixel)

    p = sub.add_parser(
        "search-kis",
        help="Search the hybrid raw-video-first KIS runtime for one TKIS/VKIS query",
    )
    _add_kis_runtime_args(p)
    p.add_argument("--task", choices=["TKIS", "VKIS"], required=True)
    query_group = p.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query")
    query_group.add_argument("--image")
    p.add_argument("--query-id", default="cli-kis")
    p.add_argument("--top-k", type=int, default=100)
    p.set_defaults(func=_cmd_search_kis)

    p = sub.add_parser(
        "retrieve-kis",
        help="Run hybrid TKIS/VKIS retrieval from questions.csv and write JSONL",
    )
    _add_kis_runtime_args(p)
    p.add_argument("--questions", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--top-k", type=int, default=100)
    p.set_defaults(func=_cmd_retrieve_kis)

    p = sub.add_parser(
        "export-kis",
        help="Run hybrid KIS and export/validate exactly 100 CSV answers per query",
    )
    _add_kis_runtime_args(p)
    p.add_argument("--questions", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=_cmd_export_kis)

    p = sub.add_parser("serve-kis", help="Serve the hybrid KIS API and operator UI")
    _add_kis_runtime_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve_kis)

    p = sub.add_parser(
        "benchmark-kis",
        help="Benchmark KIS with optional official qrels and channel ablations",
    )
    _add_kis_runtime_args(p)
    p.add_argument("--questions", required=True)
    p.add_argument("--qrels")
    p.add_argument("--qrels-source", default="unknown")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--frame-tolerance", type=int, default=12)
    p.add_argument("--out")
    p.set_defaults(func=_cmd_benchmark_kis)

    p = sub.add_parser(
        "benchmark-skillpixel",
        help="Run the SkillPixel V0/V1/V2 provider and channel benchmark matrix",
    )
    p.add_argument("--raw", required=True, help="Generated raw-video dataset root")
    p.add_argument("--index", required=True, help="Existing V0 visual index")
    p.add_argument("--questions", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--providers", default="clip,siglip2,jina-clip-v2")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--clip-model")
    p.add_argument("--siglip2-model")
    p.add_argument("--jina-model")
    p.add_argument("--allow-network", action="store_true")
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--qrels")
    p.add_argument("--qrels-source", default="unknown")
    p.add_argument("--ocr-artifact")
    p.add_argument("--object-artifact")
    p.set_defaults(func=_cmd_benchmark_skillpixel)

    p = sub.add_parser(
        "package-kaggle-skillpixel",
        help="Write a metadata-only Kaggle recipe without raw/model/generated artifacts",
    )
    p.add_argument("--raw-input", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--index")
    p.add_argument("--out", required=True)
    p.add_argument("--max-file-bytes", type=int, default=5 * 1024 * 1024)
    p.set_defaults(func=_cmd_package_kaggle_skillpixel)

    p = sub.add_parser(
        "export-skillpixel",
        help="Validate and export exactly 100 answers per SkillPixel query",
    )
    p.add_argument("--queries", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=_cmd_export_skillpixel)

    p = sub.add_parser(
        "provider-doctor",
        help="Report provider dependencies without downloading model weights",
    )
    p.add_argument(
        "--provider",
        choices=["mock", "clip", "siglip2", "jina-clip-v2"],
        required=True,
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--revision")
    p.set_defaults(func=_cmd_provider_doctor)

    p = sub.add_parser(
        "scale-benchmark",
        help="Run a synthetic exact-versus-HNSW engineering benchmark",
    )
    p.add_argument("--vectors", type=int, default=10_000)
    p.add_argument("--dimension", type=int, default=512)
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--hnsw-m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, default=128)
    p.add_argument("--out")
    p.set_defaults(func=_cmd_scale_benchmark)

    p = sub.add_parser(
        "benchmark",
        help="Run a frozen reproducible benchmark from a YAML config",
    )
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="artifacts/benchmark/competitive-v1")
    p.set_defaults(func=_cmd_benchmark)

    p = sub.add_parser("search", help="Search an index from the command line")
    p.add_argument("--index", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--video-id", help="Restrict to comma-separated video ids")
    p.add_argument("--query-id", default="cli")
    p.add_argument("--data-root", help="Override dataset root for image paths")
    p.add_argument("--index-provider", choices=["exact-numpy", "faiss", "faiss-hnsw"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_search)

    p = sub.add_parser(
        "preflight-dual",
        help="Validate the merged SigLIP2/Qwen artifact and write a local manifest",
    )
    p.add_argument("--artifacts", required=True)
    p.add_argument("--out")
    p.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip file hashing for a quick schema-only check",
    )
    p.set_defaults(func=_cmd_preflight_dual)

    p = sub.add_parser(
        "search-dual",
        help="Search the validated local SigLIP2 + Qwen dual index",
    )
    p.add_argument("--artifacts", required=True)
    query_group = p.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query")
    query_group.add_argument("--image")
    p.add_argument("--query-id", default="cli-dual")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--video-id", help="Restrict to comma-separated video ids")
    p.add_argument("--object-query", help="Exact raw object label to late-fuse")
    p.add_argument("--image-root")
    p.add_argument("--video-root")
    p.add_argument("--object-sidecar")
    p.add_argument("--object-enabled", action="store_true")
    p.add_argument("--allow-engineering-proxy", action="store_true")
    p.add_argument("--ocr-es-url")
    p.add_argument("--ocr-es-index")
    p.add_argument("--ocr-es-manifest")
    p.add_argument("--ocr-es-api-key-env", default="ELASTIC_API_KEY")
    p.add_argument("--ocr-es-username-env")
    p.add_argument("--ocr-es-password-env")
    p.add_argument("--ocr-es-exclude-low-conf", action="store_true")
    p.add_argument("--siglip-device", default="cpu")
    p.add_argument("--qwen-device", default="cpu")
    p.add_argument("--siglip-model-path", help="Optional local SigLIP checkpoint directory")
    p.add_argument("--qwen-model-path", help="Optional local Qwen checkpoint directory")
    p.add_argument(
        "--visual-indexes",
        default="siglip2",
        help=(
            "Comma-separated visual indexes; RAM-safe default is siglip2. "
            "Use siglip2,qwen to opt into dual loading."
        ),
    )
    p.add_argument("--allow-network", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_search_dual)

    p = sub.add_parser("serve-dual", help="Serve the local dual visual retrieval API")
    p.add_argument("--artifacts", required=True)
    p.add_argument("--image-root")
    p.add_argument("--video-root")
    p.add_argument("--object-sidecar")
    p.add_argument("--object-enabled", action="store_true")
    p.add_argument("--allow-engineering-proxy", action="store_true")
    p.add_argument("--asr-es-url")
    p.add_argument("--asr-es-index")
    p.add_argument("--asr-es-edges")
    p.add_argument("--asr-es-lookup")
    p.add_argument("--asr-es-timeout", type=float)
    p.add_argument("--asr-es-top-n", type=int)
    p.add_argument("--asr-es-mode", choices=["pho", "whisper_v3", "rrf"])
    p.add_argument("--asr-es-fuzziness")
    p.add_argument("--asr-es-rank-constant", type=int)
    p.add_argument("--asr-es-enabled", action="store_true")
    p.add_argument("--asr-es-policy-enabled", action="store_true")
    p.add_argument("--asr-es-api-key-env")
    p.add_argument("--asr-es-username-env")
    p.add_argument("--asr-es-password-env")
    p.add_argument("--ocr-es-url")
    p.add_argument("--ocr-es-index")
    p.add_argument("--ocr-es-manifest")
    p.add_argument("--ocr-es-api-key-env", default="ELASTIC_API_KEY")
    p.add_argument("--ocr-es-username-env")
    p.add_argument("--ocr-es-password-env")
    p.add_argument("--ocr-es-exclude-low-conf", action="store_true")
    p.add_argument("--siglip-device", default="cpu")
    p.add_argument("--qwen-device", default="cpu")
    p.add_argument("--siglip-model-path", help="Optional local SigLIP checkpoint directory")
    p.add_argument("--qwen-model-path", help="Optional local Qwen checkpoint directory")
    p.add_argument(
        "--visual-indexes",
        default="siglip2",
        help=(
            "Comma-separated visual indexes; RAM-safe default is siglip2. "
            "Use siglip2,qwen to opt into dual loading."
        ),
    )
    p.add_argument(
        "--asr-only",
        action="store_true",
        help="Start without loading visual providers; requires a ready ASR Elasticsearch config",
    )
    p.add_argument(
        "--media-manifest",
        help=(
            "Optional versioned JSONL media manifest keyed by frame_uid/video_id; "
            "use the pinned Hugging Face artifact for runtime video streaming"
        ),
    )
    p.add_argument(
        "--media-cache-root",
        help="Cache directory for verified cloud JPEG/MP4 files",
    )
    p.add_argument(
        "--media-http-host",
        action="append",
        help="Allowlisted HTTPS host for HTTP media manifest entries (repeatable)",
    )
    p.add_argument(
        "--ui-dir",
        help="Optional built BFE dist directory; falls back to local static UI",
    )
    p.add_argument("--allow-network", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve_dual)

    p = sub.add_parser("serve", help="Serve the API and operator UI")
    p.add_argument("--index", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data-root", help="Override dataset root for image paths")
    p.add_argument("--index-provider", choices=["exact-numpy", "faiss", "faiss-hnsw"])
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("evaluate", help="Run the evaluator")
    p.add_argument("--index", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--qrels", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out", help="Output directory for reports")
    p.add_argument("--data-root")
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser(
        "prepare-groundtruth-review",
        help="Build a 3 FPS anchor-centred range-review bundle",
    )
    p.add_argument("--proposals", required=True, help="range proposals JSONL")
    p.add_argument("--inventory", required=True, help="video_inventory JSONL")
    p.add_argument("--pts-dir", required=True, help="Directory containing per-video PTS JSONL")
    p.add_argument("--output", required=True, help="Review bundle output directory")
    p.add_argument("--raw-root", help="Raw video root; required with --materialize")
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--window-before-s", type=float, default=20.0)
    p.add_argument("--window-after-s", type=float, default=20.0)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument(
        "--materialize",
        action="store_true",
        help="Decode raw videos and write deduplicated JPEGs for the UI",
    )
    p.set_defaults(func=_cmd_prepare_groundtruth_review)

    p = sub.add_parser(
        "serve-groundtruth-review",
        help="Serve the ground-truth range review UI",
    )
    p.add_argument("--review-root", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve_groundtruth_review)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

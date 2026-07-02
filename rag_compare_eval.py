#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml
from qdrant_client import QdrantClient


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "rag_indexer_tool" / "config.yaml"
DEFAULT_QUESTIONS = ROOT / "docs" / "rag_test_content" / "rag_eval_questions.json"
DEFAULT_REPORT = ROOT / "rag_eval_report.json"
DEFAULT_API_BASE = "http://localhost:8080"
DEFAULT_API_PATH = "/v1/chat/completions"


def normalize_api_url(value: str) -> str:
    url = value.strip().rstrip("/") or DEFAULT_API_BASE
    if "://" not in url:
        url = f"http://{url}"
    if not url.endswith(DEFAULT_API_PATH):
        url = url.rstrip("/") + DEFAULT_API_PATH
    return url


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return loaded if isinstance(loaded, dict) else {}


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, list):
        raise ValueError("Questions file must contain a JSON array")
    return loaded


def embed_query(ollama_url: str, model: str, question: str) -> list[float]:
    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/embed",
        json={"model": model, "input": [question]},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    embeddings = data.get("embeddings")
    if embeddings is None and "embedding" in data:
        embeddings = [data["embedding"]]
    if not embeddings or not isinstance(embeddings[0], list):
        raise RuntimeError(f"Unexpected Ollama embedding response: {data.keys()}")
    return embeddings[0]


def search_chunks(
    qdrant_url: str,
    collection: str,
    query_vector: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    client = QdrantClient(url=qdrant_url)
    result = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    hits = []
    for point in result.points:
        payload = point.payload or {}
        text = str(payload.get("text", "")).strip()
        if not text:
            continue
        hits.append(
            {
                "score": float(point.score or 0.0),
                "text": text,
                "source_path": str(
                    payload.get("source_path")
                    or payload.get("file_name")
                    or "unknown"
                ),
                "chunk_index": payload.get("chunk_index"),
                "heading_path": payload.get("heading_path") or [],
            }
        )
    return hits


def format_sources(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_path": hit["source_path"],
            "chunk_index": hit["chunk_index"],
            "score": hit["score"],
            "rerank_score": hit.get("rerank_score"),
            "lexical_overlap": hit.get("lexical_overlap"),
        }
        for hit in hits
    ]


def build_rag_context(hits: list[dict[str, Any]], max_chars: int) -> str:
    blocks = []
    used = 0
    for index, hit in enumerate(hits, 1):
        remaining = max_chars - used
        if remaining <= 0:
            break
        text = hit["text"]
        if len(text) > remaining:
            text = text[:remaining].rstrip()
        heading = hit["heading_path"]
        heading_text = (
            " > ".join(str(item) for item in heading)
            if isinstance(heading, list)
            else str(heading)
        )
        location = hit["source_path"]
        if hit["chunk_index"] is not None:
            location += f", chunk {hit['chunk_index']}"
        if heading_text:
            location += f", {heading_text}"
        blocks.append(
            f"[{index}] score={hit['score']:.3f}; source={location}\n{text}"
        )
        used += len(text)
    return "\n\n".join(blocks)


def ask_llm(api_url: str, messages: list[dict[str, str]], temperature: float) -> str:
    response = requests.post(
        api_url,
        json={"messages": messages, "temperature": temperature, "max_tokens": 2000},
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def rewrite_query(api_url: str, question: str) -> str:
    response = requests.post(
        api_url,
        json={
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's question into a concise search query "
                        "for a local vector knowledge base. Preserve exact names, "
                        "codes, identifiers, numbers, and quoted phrases. "
                        "Return only the rewritten query."
                    ),
                },
                {"role": "user", "content": question},
            ],
            "temperature": 0.0,
            "max_tokens": 120,
        },
        timeout=30,
    )
    response.raise_for_status()
    rewritten = str(response.json()["choices"][0]["message"]["content"]).strip()
    return rewritten or question


def tokenize_for_rerank(text: str) -> set[str]:
    import re

    return {
        token.casefold()
        for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9_.:-]+", text)
        if len(token) > 2
    }


def filter_and_rerank(
    hits: list[dict[str, Any]],
    original_query: str,
    rewritten_query: str,
    threshold: float,
    post_top_k: int,
) -> list[dict[str, Any]]:
    query_tokens = tokenize_for_rerank(f"{original_query} {rewritten_query}")
    filtered = [dict(hit) for hit in hits if hit["score"] >= threshold]
    for hit in filtered:
        text_tokens = tokenize_for_rerank(f"{hit['source_path']} {hit['text']}")
        overlap = len(query_tokens & text_tokens) / max(1, len(query_tokens))
        hit["lexical_overlap"] = overlap
        hit["rerank_score"] = (0.85 * hit["score"]) + (0.15 * overlap)
    filtered.sort(key=lambda item: item["rerank_score"], reverse=True)
    return filtered[:post_top_k]


def build_answer_from_context(
    api_url: str,
    question: str,
    hits: list[dict[str, Any]],
    max_context_chars: int,
    temperature: float,
) -> str:
    if not hits:
        return ""
    rag_context = build_rag_context(hits, max_context_chars)
    return ask_llm(
        api_url,
        [
            {
                "role": "system",
                "content": (
                    "Answer only from the provided RAG context. "
                    "If the context is insufficient, say exactly what is missing.\n\n"
                    f"RAG context:\n{rag_context}"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature,
    )


def contains_all(answer: str, expected: list[str]) -> bool:
    normalized = answer.casefold()
    return all(item.casefold() in normalized for item in expected)


def sources_match(hits: list[dict[str, Any]], expected_sources: list[str]) -> bool:
    if not expected_sources:
        return True
    source_text = "\n".join(hit["source_path"] for hit in hits).casefold()
    return all(expected.casefold() in source_text for expected in expected_sources)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(Path(args.config))
    embedding_cfg = config.get("embedding", {})
    qdrant_cfg = config.get("qdrant", {})

    api_url = normalize_api_url(
        args.api_url
        or args.server
        or os.environ.get("AI_ADVENT_API_URL")
        or os.environ.get("AI_ADVENT_API_BASE")
        or DEFAULT_API_BASE
    )
    ollama_url = (
        args.ollama_url
        or os.environ.get("AI_ADVENT_RAG_OLLAMA_URL")
        or embedding_cfg.get("ollama_url")
        or "http://localhost:11434"
    )
    embedding_model = (
        args.embedding_model
        or os.environ.get("AI_ADVENT_RAG_EMBEDDING_MODEL")
        or embedding_cfg.get("embedding_model")
        or "nomic-embed-text"
    )
    qdrant_url = (
        args.qdrant_url
        or os.environ.get("AI_ADVENT_RAG_QDRANT_URL")
        or qdrant_cfg.get("url")
        or "http://localhost:6333"
    )
    collection = (
        args.collection
        or os.environ.get("AI_ADVENT_RAG_COLLECTION")
        or qdrant_cfg.get("collection_name")
        or "local_knowledge_base"
    )

    questions = load_questions(Path(args.questions))
    results = []
    for item in questions:
        question = str(item["question"])
        expected_answer = [str(value) for value in item.get("expected_answer_contains", [])]
        expected_sources = [str(value) for value in item.get("expected_sources_contains", [])]

        basic_vector = embed_query(ollama_url, embedding_model, question)
        basic_hits = search_chunks(qdrant_url, collection, basic_vector, args.post_top_k)
        basic_best_score = max((hit["score"] for hit in basic_hits), default=0.0)

        rewritten_query = (
            question
            if args.no_query_rewrite or args.retrieval_only
            else rewrite_query(api_url, question)
        )
        improved_vector = embed_query(ollama_url, embedding_model, rewritten_query)
        raw_improved_hits = search_chunks(
            qdrant_url, collection, improved_vector, args.pre_top_k
        )
        improved_hits = filter_and_rerank(
            raw_improved_hits,
            question,
            rewritten_query,
            args.threshold,
            args.post_top_k,
        )
        fallback_to_original_query = False
        if not improved_hits and rewritten_query != question:
            fallback_to_original_query = True
            improved_vector = embed_query(ollama_url, embedding_model, question)
            raw_improved_hits = search_chunks(
                qdrant_url, collection, improved_vector, args.pre_top_k
            )
            improved_hits = filter_and_rerank(
                raw_improved_hits,
                question,
                question,
                args.threshold,
                args.post_top_k,
            )
        improved_best_score = max(
            (hit["score"] for hit in raw_improved_hits), default=0.0
        )
        improved_rag_used = bool(improved_hits)

        if args.retrieval_only:
            no_rag_answer = ""
            basic_rag_answer = ""
            improved_rag_answer = ""
        else:
            no_rag_answer = ask_llm(
                api_url,
                [
                    {
                        "role": "system",
                        "content": (
                            "Answer the user directly. Do not use any external "
                            "documents or hidden context."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                args.temperature,
            )
            basic_rag_answer = build_answer_from_context(
                api_url,
                question,
                basic_hits,
                args.max_context_chars,
                args.temperature,
            )
            improved_rag_answer = build_answer_from_context(
                api_url,
                question,
                improved_hits,
                args.max_context_chars,
                args.temperature,
            )

        basic_answer_ok = contains_all(basic_rag_answer, expected_answer)
        improved_answer_ok = contains_all(improved_rag_answer, expected_answer)
        basic_sources_ok = sources_match(basic_hits, expected_sources)
        improved_sources_ok = sources_match(improved_hits, expected_sources)

        results.append(
            {
                "id": item.get("id"),
                "question": question,
                "expected_answer_contains": expected_answer,
                "expected_sources_contains": expected_sources,
                "no_rag_answer": no_rag_answer,
                "basic_rag_answer": basic_rag_answer,
                "improved_rag_answer": improved_rag_answer,
                "rag_answer": improved_rag_answer,
                "rewritten_query": rewritten_query,
                "fallback_to_original_query": fallback_to_original_query,
                "basic_rag_used": bool(basic_hits),
                "improved_rag_used": improved_rag_used,
                "rag_used": improved_rag_used,
                "basic_best_score": basic_best_score,
                "improved_best_score": improved_best_score,
                "best_score": improved_best_score,
                "basic_sources": format_sources(basic_hits),
                "improved_sources": format_sources(improved_hits),
                "raw_improved_sources": format_sources(raw_improved_hits),
                "sources": format_sources(improved_hits),
                "top_k_before_filter": args.pre_top_k,
                "top_k_after_filter": args.post_top_k,
                "similarity_threshold": args.threshold,
                "checks": {
                    "basic_rag_answer_contains_expected": (
                        basic_answer_ok if not args.retrieval_only else False
                    ),
                    "improved_rag_answer_contains_expected": (
                        improved_answer_ok if not args.retrieval_only else False
                    ),
                    "rag_answer_contains_expected": (
                        improved_answer_ok if not args.retrieval_only else False
                    ),
                    "basic_expected_sources_used": basic_sources_ok,
                    "improved_expected_sources_used": improved_sources_ok,
                    "expected_sources_used": improved_sources_ok,
                },
            }
        )

    basic_passed = sum(
        1
        for item in results
        if item["checks"]["basic_rag_answer_contains_expected"]
        and item["checks"]["basic_expected_sources_used"]
    )
    improved_passed = sum(
        1
        for item in results
        if item["checks"]["improved_rag_answer_contains_expected"]
        and item["checks"]["improved_expected_sources_used"]
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "api_url": api_url,
        "ollama_url": ollama_url,
        "embedding_model": embedding_model,
        "qdrant_url": qdrant_url,
        "collection": collection,
        "top_k_before_filter": args.pre_top_k,
        "top_k_after_filter": args.post_top_k,
        "threshold": args.threshold,
        "query_rewrite": not args.no_query_rewrite,
        "retrieval_only": args.retrieval_only,
        "basic_passed": basic_passed,
        "improved_passed": improved_passed,
        "passed": improved_passed,
        "total": len(results),
        "results": results,
    }


def print_summary(report: dict[str, Any]) -> None:
    if report.get("retrieval_only"):
        source_passed = sum(
            1
            for item in report["results"]
            if item["checks"]["improved_expected_sources_used"]
        )
        routed = sum(1 for item in report["results"] if item["improved_rag_used"])
        print(
            f"RAG retrieval eval: sources {source_passed}/{report['total']}, "
            f"above threshold {routed}/{report['total']}"
        )
        for item in report["results"]:
            status = "PASS" if item["checks"]["improved_expected_sources_used"] else "FAIL"
            print(
                f"- {status} {item['id']}: basic={item['basic_best_score']:.3f}, "
                f"improved={item['improved_best_score']:.3f}, "
                f"kept={len(item['improved_sources'])}/"
                f"{len(item['raw_improved_sources'])}"
            )
        return

    print(
        "RAG eval: "
        f"basic {report['basic_passed']}/{report['total']}, "
        f"improved {report['improved_passed']}/{report['total']} passed"
    )
    for item in report["results"]:
        checks = item["checks"]
        basic_status = (
            "PASS"
            if checks["basic_rag_answer_contains_expected"]
            and checks["basic_expected_sources_used"]
            else "FAIL"
        )
        improved_status = (
            "PASS"
            if checks["improved_rag_answer_contains_expected"]
            and checks["improved_expected_sources_used"]
            else "FAIL"
        )
        print(
            f"- {item['id']}: basic={basic_status} "
            f"score={item['basic_best_score']:.3f}; "
            f"improved={improved_status} score={item['improved_best_score']:.3f}, "
            f"kept={len(item['improved_sources'])}/"
            f"{len(item['raw_improved_sources'])}, "
            f"fallback={item['fallback_to_original_query']}"
        )


def print_answers(report: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("RAG COMPARISON ANSWERS")
    print("=" * 80)
    for index, item in enumerate(report["results"], 1):
        print(f"\n[{index}] {item['id']}")
        print(f"QUESTION:\n{item['question']}")
        print("\nWITHOUT RAG:")
        print(item["no_rag_answer"] or "<not requested>")
        print("\nWITH RAG:")
        print("BASIC RAG:")
        print(item["basic_rag_answer"] or "<not requested>")
        print("\nIMPROVED RAG:")
        print(item["improved_rag_answer"] or "<RAG was not used>")
        print(f"\nQUERY REWRITE:\n{item['rewritten_query']}")
        print(f"FALLBACK TO ORIGINAL QUERY: {item['fallback_to_original_query']}")
        print("\nBASIC SOURCES:")
        if item["basic_sources"]:
            for source in item["basic_sources"]:
                print(
                    "- "
                    f"{source['source_path']}, "
                    f"chunk={source['chunk_index']}, "
                    f"score={source['score']:.3f}"
                )
        else:
            print("<no sources>")
        print("\nIMPROVED SOURCES:")
        if item["improved_sources"]:
            for source in item["improved_sources"]:
                rerank = source.get("rerank_score")
                rerank_text = f", rerank={rerank:.3f}" if rerank is not None else ""
                print(
                    "- "
                    f"{source['source_path']}, "
                    f"chunk={source['chunk_index']}, "
                    f"score={source['score']:.3f}"
                    f"{rerank_text}"
                )
        else:
            print("<no sources after filtering>")
        print("-" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare answers without RAG and with RAG for control questions."
    )
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--server", help="LLM server host[:port] or base URL")
    parser.add_argument("--api-url", help="Full OpenAI-compatible chat completions URL")
    parser.add_argument("--ollama-url")
    parser.add_argument("--embedding-model")
    parser.add_argument("--qdrant-url")
    parser.add_argument("--collection")
    parser.add_argument(
        "--pre-top-k",
        type=int,
        default=10,
        help="How many chunks to retrieve before filtering/reranking.",
    )
    parser.add_argument(
        "--post-top-k",
        "--top-k",
        dest="post_top_k",
        type=int,
        default=5,
        help="How many chunks to keep after filtering/reranking.",
    )
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Check only embedding search, scores, and expected sources.",
    )
    parser.add_argument(
        "--print-answers",
        action="store_true",
        help="Print every question with no-RAG answer, RAG answer, and sources.",
    )
    parser.add_argument(
        "--no-query-rewrite",
        action="store_true",
        help="Disable LLM query rewrite for the improved RAG mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = evaluate(args)
    except Exception as error:
        print(f"RAG eval failed: {error}", file=sys.stderr)
        return 1

    output = Path(args.output)
    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print_summary(report)
    if args.print_answers:
        print_answers(report)
    print(f"Full report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

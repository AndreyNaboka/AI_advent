from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from chunkers.base import ChunkingConfig
from chunkers.code_chunker import LANG_BY_EXT, chunk_code
from chunkers.docx_chunker import chunk_docx
from chunkers.html_chunker import chunk_html
from chunkers.json_chunker import chunk_json_or_yaml
from chunkers.markdown_chunker import chunk_markdown
from chunkers.pdf_chunker import chunk_pdf
from chunkers.table_chunker import chunk_table
from chunkers.text_chunker import chunk_txt
from utils.file_walker import iter_files
from utils.hashing import sha256_file, sha256_text
from utils.logging import setup_logging


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_chunk_config(raw: dict[str, Any], args: argparse.Namespace | None = None) -> ChunkingConfig:
    data = dict(raw.get("chunking", {}))
    if args is not None:
        for key in ["target_chunk_tokens", "min_chunk_tokens", "max_chunk_tokens", "overlap_tokens"]:
            value = getattr(args, key, None)
            if value is not None:
                data[key] = value
    return ChunkingConfig(**data)


def chunk_file(path: Path, config: ChunkingConfig):
    ext = path.suffix.lower()
    if ext == ".txt":
        return chunk_txt(path, config)
    if ext in {".md", ".markdown"}:
        return chunk_markdown(path, config)
    if ext == ".pdf":
        return chunk_pdf(path, config)
    if ext == ".docx":
        return chunk_docx(path, config)
    if ext in {".html", ".htm"}:
        return chunk_html(path, config)
    if ext in {".csv", ".tsv"}:
        return chunk_table(path, config)
    if ext in {".json", ".yaml", ".yml"}:
        return chunk_json_or_yaml(path, config)
    if ext in LANG_BY_EXT:
        return chunk_code(path, config)
    return []


def run_chunk(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    logger = setup_logging(args.log or cfg.get("output", {}).get("log_file", "rag_indexer.log"))
    chunk_config = build_chunk_config(cfg, args)
    input_root = Path(args.input).expanduser().resolve()
    output_path = Path(args.output or cfg.get("output", {}).get("chunks_file", "chunks.jsonl")).expanduser()

    if not input_root.exists() or not input_root.is_dir():
        logger.error("Input path is not a directory: %s", input_root)
        return 2

    found = processed = skipped = errors = chunk_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()

    with output_path.open("w", encoding="utf-8") as out:
        for path, supported in iter_files(input_root):
            found += 1
            rel_path = path.relative_to(input_root).as_posix()
            if not supported:
                skipped += 1
                logger.info("Skipped unsupported file: %s", rel_path)
                continue
            try:
                content_hash = sha256_file(path)
                document_id = sha256_text(rel_path)[:16]
                stat = path.stat()
                modified_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
                raw_chunks = chunk_file(path, chunk_config)
                for index, raw_chunk in enumerate(raw_chunks):
                    metadata = {
                        "source_path": rel_path,
                        "file_name": path.name,
                        "file_extension": path.suffix.lower(),
                        "content_type": raw_chunk.metadata.get("content_type"),
                        "chunk_index": index,
                        "heading_path": raw_chunk.metadata.get("heading_path", []),
                        "page_start": raw_chunk.metadata.get("page_start"),
                        "page_end": raw_chunk.metadata.get("page_end"),
                        "language": raw_chunk.metadata.get("language"),
                        "symbol_type": raw_chunk.metadata.get("symbol_type"),
                        "symbol_name": raw_chunk.metadata.get("symbol_name"),
                        "created_at": created_at,
                        "modified_at": modified_at,
                        "content_hash": content_hash,
                        "chunk_hash": raw_chunk.metadata.get("chunk_hash") or sha256_text(raw_chunk.text),
                    }
                    for extra_key in ["json_path", "yaml_path", "table_name", "columns"]:
                        if extra_key in raw_chunk.metadata:
                            metadata[extra_key] = raw_chunk.metadata[extra_key]
                    chunk = {
                        "id": f"{document_id}:{index:04d}",
                        "document_id": document_id,
                        "text": raw_chunk.text,
                        "metadata": metadata,
                    }
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    chunk_count += 1
                processed += 1
            except Exception:
                errors += 1
                logger.exception("Failed to process file: %s", rel_path)

    logger.info(
        "Chunking finished: found=%s processed=%s skipped=%s errors=%s chunks=%s output=%s",
        found,
        processed,
        skipped,
        errors,
        chunk_count,
        output_path,
    )
    return 0 if errors == 0 else 1


def load_chunks(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_embed(args: argparse.Namespace) -> int:
    from embeddings.ollama_embedder import OllamaEmbedder
    from vector_db.qdrant_store import QdrantStore

    cfg = load_config(Path(args.config))
    logger = setup_logging(args.log or cfg.get("output", {}).get("log_file", "rag_indexer.log"))
    chunks_path = Path(args.chunks or cfg.get("output", {}).get("chunks_file", "chunks.jsonl")).expanduser()
    embedding_cfg = cfg.get("embedding", {})
    qdrant_cfg = cfg.get("qdrant", {})
    batch_size = int(args.batch_size or embedding_cfg.get("batch_size", 16))

    chunks = list(load_chunks(chunks_path))
    if not chunks:
        logger.error("No chunks found in %s", chunks_path)
        return 2

    embedder = OllamaEmbedder(
        base_url=args.ollama_url or embedding_cfg.get("ollama_url", "http://localhost:11434"),
        model=args.model or embedding_cfg.get("embedding_model", "nomic-embed-text"),
    )
    store = QdrantStore(
        url=args.qdrant_url or qdrant_cfg.get("url", "http://localhost:6333"),
        collection_name=args.collection or qdrant_cfg.get("collection_name", "local_knowledge_base"),
    )

    try:
        first_vector = embedder.embed_one(chunks[0]["text"])
        store.ensure_collection(len(first_vector))
        for document_id in sorted({chunk["document_id"] for chunk in chunks}):
            store.delete_document(document_id)

        all_vectors = [first_vector]
        remaining = chunks[1:]
        for start in range(0, len(remaining), batch_size):
            batch = remaining[start : start + batch_size]
            all_vectors.extend(embedder.embed_many(chunk["text"] for chunk in batch))
            logger.info("Embedded %s/%s chunks", min(len(all_vectors), len(chunks)), len(chunks))

        for start in range(0, len(chunks), batch_size):
            store.upsert_chunks(chunks[start : start + batch_size], all_vectors[start : start + batch_size])
        logger.info("Embedding finished: chunks=%s collection=%s", len(chunks), store.collection_name)
        return 0
    except Exception:
        logger.exception("Embedding or Qdrant upload failed")
        return 1


def interactive_args() -> list[str]:
    folder = input("Введите путь к папке:\n> ").strip()
    print("Выберите действие:")
    print("1. Проанализировать папку и подпапки, создать chunks")
    print("2. Построить embeddings и сохранить их в локальную БД")
    choice = input("> ").strip()
    if choice == "1":
        return ["chunk", "--input", folder]
    if choice == "2":
        chunks = input("Путь к chunks JSONL [chunks.jsonl]:\n> ").strip() or "chunks.jsonl"
        return ["embed", "--chunks", chunks]
    print("Неизвестное действие", file=sys.stderr)
    return ["--help"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build chunks and local embeddings for a folder.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    subparsers = parser.add_subparsers(dest="command")

    chunk = subparsers.add_parser("chunk", help="Analyze folder and write chunks JSONL")
    chunk.add_argument("--input", required=True, help="Input folder")
    chunk.add_argument("--output", help="Output JSONL file")
    chunk.add_argument("--log", help="Log file")
    chunk.add_argument("--target-chunk-tokens", type=int)
    chunk.add_argument("--min-chunk-tokens", type=int)
    chunk.add_argument("--max-chunk-tokens", type=int)
    chunk.add_argument("--overlap-tokens", type=int)
    chunk.set_defaults(func=run_chunk)

    embed = subparsers.add_parser("embed", help="Build embeddings and upload to Qdrant")
    embed.add_argument("--chunks", help="Chunks JSONL file")
    embed.add_argument("--log", help="Log file")
    embed.add_argument("--ollama-url", help="Ollama base URL")
    embed.add_argument("--model", help="Ollama embedding model")
    embed.add_argument("--qdrant-url", help="Qdrant URL")
    embed.add_argument("--collection", help="Qdrant collection name")
    embed.add_argument("--batch-size", type=int)
    embed.set_defaults(func=run_embed)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        argv = interactive_args()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

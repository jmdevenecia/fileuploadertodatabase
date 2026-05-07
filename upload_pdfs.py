#!/usr/bin/env python3
"""
upload_pdfs.py — Upload PDF research files into Supabase pgvector.
Supports individual uploads, batch uploads, and metadata-file-driven ingestion.

Setup (one time):
    pip install pypdf langchain langchain-community langchain-postgres \\
                langchain-ollama langchain-core supabase psycopg2-binary \\
                pgvector python-dotenv

    Copy .env.example to .env in the same folder as this script and fill in values.
    Run setup_pdf_log.sql once in your Supabase SQL Editor.

─────────────────────────────────────────────
USAGE — INDIVIDUAL UPLOAD
─────────────────────────────────────────────
    python upload_pdfs.py paper.pdf --title "My Paper Title"
    python upload_pdfs.py paper.pdf --title "My Paper" --category Agriculture --year 2024 --authors "Juan dela Cruz"
    python upload_pdfs.py paper.pdf --interactive          # prompts for each metadata field
    python upload_pdfs.py paper.pdf --dry-run              # preview without writing

─────────────────────────────────────────────
USAGE — BATCH UPLOAD (metadata file)
─────────────────────────────────────────────
    python upload_pdfs.py --meta metadata.txt
    python upload_pdfs.py *.pdf --meta metadata.txt
    python upload_pdfs.py --folder ./research/ --meta metadata.txt
    python upload_pdfs.py --meta metadata.txt --dry-run

─────────────────────────────────────────────
DELETE
─────────────────────────────────────────────
    python upload_pdfs.py --delete paper.pdf
    python upload_pdfs.py --delete paper.pdf health_survey.pdf
    python upload_pdfs.py --delete --category Agriculture
    python upload_pdfs.py --delete --all
    python upload_pdfs.py --delete paper.pdf --dry-run     # preview without deleting

─────────────────────────────────────────────
OTHER COMMANDS
─────────────────────────────────────────────
    python upload_pdfs.py --list                           # list all uploaded PDFs
    python upload_pdfs.py paper.pdf --no-skip              # force re-upload

─────────────────────────────────────────────
METADATA FILE FORMAT  (metadata.txt)
─────────────────────────────────────────────
Blank-line-separated blocks. Only `file` is required per entry.
Title is NEVER auto-filled from the filename — leave it blank
to store NULL.

    file:     rice_yields_2023.pdf
    title:    Effect of Drought on Rice Yields in Luzon
    category: Agriculture
    year:     2023
    authors:  Juan dela Cruz, Maria Santos

    file:     health_survey.pdf
    title:    Rural Health Survey Results
    category: Health

Lines starting with # are comments and are ignored.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

SUPABASE_URL              = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
DATABASE_URL              = os.environ["DATABASE_URL"]
OLLAMA_BASE_URL           = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL        = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")
VECTOR_COLLECTION         = os.environ.get("VECTOR_COLLECTION", "research_embeddings")


# ══════════════════════════════════════════════
#  Metadata file parser
# ══════════════════════════════════════════════

def parse_metadata_file(meta_path: Path) -> dict:
    """
    Parse a metadata .txt file into a dict keyed by bare filename.

    Returns:
        { "paper.pdf": {"title": ..., "category": ..., "year": ..., "authors": ...}, ... }
    """
    if not meta_path.exists():
        print(f"❌  Metadata file not found: {meta_path}")
        sys.exit(1)

    entries: dict = {}
    current: dict = {}

    def flush(entry: dict):
        fname = entry.get("file", "").strip()
        if fname:
            key = Path(fname).name
            year_raw = entry.get("year", "").strip()
            entries[key] = {
                "title":    entry.get("title",    "").strip() or None,
                "category": entry.get("category", "").strip() or None,
                "year":     int(year_raw) if year_raw.isdigit() else None,
                "authors":  entry.get("authors",  "").strip() or None,
            }

    for raw_line in meta_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line:                # blank line → end of block
            if current:
                flush(current)
                current = {}
            continue

        if line.startswith("#"):    # comment
            continue

        if ":" in line:
            key, _, value = line.partition(":")
            current[key.strip().lower()] = value.strip()

    if current:                     # flush last block (no trailing blank line needed)
        flush(current)

    return entries


def lookup_meta(filename: str, meta_map: dict):
    """Return the metadata dict for a given filename, or None if not found."""
    return meta_map.get(Path(filename).name)


# ══════════════════════════════════════════════
#  Interactive metadata prompt
# ══════════════════════════════════════════════

def prompt_meta(pdf_path: Path) -> dict:
    """Interactively prompt the user for metadata fields for a single PDF."""
    print(f"\n  📝 Enter metadata for: {pdf_path.name}")
    print(f"     (Press Enter to leave a field blank / store as NULL)\n")

    def ask(label, cast=str):
        while True:
            raw = input(f"     {label}: ").strip()
            if not raw:
                return None
            try:
                return cast(raw)
            except ValueError:
                print(f"     ⚠️  Invalid value. Try again.")

    return {
        "title":    ask("Title"),
        "category": ask("Category  (e.g. Agriculture, Health, Education)"),
        "year":     ask("Year      (e.g. 2024)", cast=int),
        "authors":  ask("Authors   (comma-separated)"),
    }


# ══════════════════════════════════════════════
#  Core helpers
# ══════════════════════════════════════════════

def extract_text(pdf_path: Path) -> str:
    import pypdf
    reader = pypdf.PdfReader(str(pdf_path))
    pages = [p.extract_text() for p in reader.pages]
    return "\n\n".join(t.strip() for t in pages if t and t.strip())


def file_md5(pdf_path: Path) -> str:
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk_bytes in iter(lambda: f.read(8192), b""):
            h.update(chunk_bytes)
    return h.hexdigest()


def already_uploaded(checksum: str, sb) -> bool:
    try:
        r = sb.table("pdf_ingestion_log").select("id").eq("checksum", checksum).execute()
        return len(r.data) > 0
    except Exception:
        return False


def log_upload(pdf_path: Path, checksum: str, chunk_count: int, meta: dict, sb):
    try:
        sb.table("pdf_ingestion_log").insert({
            "filename":    pdf_path.name,
            "filepath":    str(pdf_path.resolve()),
            "checksum":    checksum,
            "chunk_count": chunk_count,
            "title":       meta.get("title"),
            "category":    meta.get("category"),
            "year":        meta.get("year"),
            "authors":     meta.get("authors"),
        }).execute()
    except Exception as e:
        print(f"  ⚠️  Log write failed: {e}")


# ══════════════════════════════════════════════
#  Ingestion
# ══════════════════════════════════════════════

def ingest_pdf(
    pdf_path: Path,
    meta: dict,
    skip_dupes: bool,
    dry_run: bool,
    chunk_size: int,
    chunk_overlap: int,
    sb,
    store,
) -> int:
    """
    Ingest one PDF. Returns number of chunks stored (0 = skipped or failed).
    Title is taken strictly from `meta`; never auto-derived from filename.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    print(f"\n  📄 {pdf_path.name}")

    if not pdf_path.exists():
        print(f"     ❌ File not found")
        return 0
    if pdf_path.suffix.lower() != ".pdf":
        print(f"     ⚠️  Not a PDF — skipping")
        return 0

    # Title: ONLY from metadata, never from filename
    title = meta.get("title") or None
    if title is None:
        print(f"     ⚠️  No title provided — will be stored as NULL")

    checksum = file_md5(pdf_path)
    if skip_dupes and already_uploaded(checksum, sb):
        print(f"     ⏭️  Already uploaded — skipping (use --no-skip to force)")
        return 0

    print(f"     📖 Extracting text...", end=" ", flush=True)
    try:
        text = extract_text(pdf_path)
    except Exception as e:
        print(f"❌\n     Error: {e}")
        return 0

    if not text.strip():
        print("❌\n     No text found — PDF may be image-based/scanned. Use OCR first.")
        return 0
    print(f"{len(text):,} chars")

    print(
        f"     🏷️  title={title!r}  "
        f"category={meta.get('category')!r}  "
        f"year={meta.get('year')}  "
        f"authors={meta.get('authors')!r}"
    )

    # Build document metadata for the vector store
    doc_meta: dict = {
        "source":       pdf_path.name,
        "source_table": "pdf_upload",
        "file_type":    "pdf",
        "checksum":     checksum,
    }
    if title:                   doc_meta["title"]    = title
    if meta.get("category"):    doc_meta["category"] = meta["category"]
    if meta.get("year"):        doc_meta["year"]     = str(meta["year"])
    if meta.get("authors"):     doc_meta["authors"]  = meta["authors"]

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_documents([Document(page_content=text, metadata=doc_meta)])

    print(f"     ✂️  {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")

    if dry_run:
        print(f"     🔍 DRY RUN — nothing stored or logged")
        print(f"     Preview: {chunks[0].page_content[:140].strip()}...")
        return len(chunks)

    stored = 0
    for i in range(0, len(chunks), 10):
        store.add_documents(chunks[i:i + 10])
        stored += len(chunks[i:i + 10])
        pct = int(stored / len(chunks) * 100)
        print(f"\r     🧠 Embedding... {pct}% ({stored}/{len(chunks)})", end="", flush=True)
    print(f"\r     ✅ {len(chunks)} chunks embedded and stored" + " " * 20)

    log_upload(pdf_path, checksum, len(chunks), {**meta, "title": title}, sb)
    return len(chunks)


# ══════════════════════════════════════════════
#  List uploaded PDFs
# ══════════════════════════════════════════════

def list_uploads(sb):
    try:
        rows = (
            sb.table("pdf_ingestion_log")
            .select("filename, title, category, year, authors, chunk_count, created_at")
            .order("created_at", desc=True)
            .execute()
            .data
        )
    except Exception as e:
        print(f"❌ Could not fetch log: {e}")
        return

    if not rows:
        print("No PDFs uploaded yet.")
        return

    col = "{:<35} {:<30} {:<14} {:<6} {:<8} {}"
    print("\n" + col.format("FILENAME", "TITLE", "CATEGORY", "YEAR", "CHUNKS", "DATE"))
    print("─" * 110)
    for r in rows:
        print(col.format(
            (r.get("filename") or "")[:34],
            (r.get("title")    or "—")[:29],
            (r.get("category") or "")[:13],
            str(r.get("year")        or ""),
            str(r.get("chunk_count") or ""),
            (r.get("created_at") or "")[:10],
        ))
    print(f"\nTotal: {len(rows)} file(s)")


# ══════════════════════════════════════════════
#  Delete
# ══════════════════════════════════════════════

def delete_pdfs(
    filenames: list[str],
    category: str | None,
    all_files: bool,
    dry_run: bool,
    sb,
):
    """
    Delete PDFs from both pdf_ingestion_log and langchain_pg_embedding.
    Uses only the Supabase REST API — no direct DB connection needed.
    Accepts a list of filenames, a --category filter, or --all.
    """

    # ── 1. Resolve which log rows to delete ──
    try:
        if all_files:
            rows = sb.table("pdf_ingestion_log") \
                     .select("filename, chunk_count") \
                     .execute().data
        elif category:
            rows = sb.table("pdf_ingestion_log") \
                     .select("filename, chunk_count") \
                     .eq("category", category).execute().data
        else:
            rows = []
            for fname in filenames:
                r = sb.table("pdf_ingestion_log") \
                      .select("filename, chunk_count") \
                      .eq("filename", fname).execute().data
                if r:
                    rows.extend(r)
                else:
                    print(f"  ⚠️  '{fname}' not found in log — skipping")
    except Exception as e:
        print(f"❌ Could not query log: {e}")
        return

    if not rows:
        print("  ℹ️  Nothing matched — no files deleted.")
        return

    # ── 2. Preview & confirm ──
    total_chunks = sum(r.get("chunk_count") or 0 for r in rows)
    print(f"\n  The following {len(rows)} file(s) will be permanently deleted:")
    for r in rows:
        print(f"    • {r['filename']}  ({r.get('chunk_count') or '?'} chunks)")
    print(f"\n  Total vectors to remove: ~{total_chunks}")

    if dry_run:
        print("\n  🔍 DRY RUN — nothing deleted.")
        return

    confirm = input(f"\n  ⚠️  Confirm delete? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    # ── 3. Delete vectors via Supabase REST (langchain_pg_embedding) ──
    #    The `source` field in cmetadata matches the PDF filename.
    deleted_vectors = 0
    vector_errors = []
    for r in rows:
        fname = r["filename"]
        try:
            # langchain_pg_embedding stores metadata as JSONB in `cmetadata`
            result = sb.table("langchain_pg_embedding") \
                       .delete() \
                       .eq("cmetadata->>source", fname) \
                       .execute()
            count = len(result.data) if result.data else 0
            deleted_vectors += count
            print(f"  🗑️  Vectors removed for '{fname}': {count}")
        except Exception as e:
            vector_errors.append(fname)
            print(f"  ⚠️  Could not delete vectors for '{fname}': {e}")

    if vector_errors:
        print(f"\n  ⚠️  Vector deletion failed for {len(vector_errors)} file(s): {vector_errors}")
        print(f"     Log entries will still be removed.")

    # ── 4. Delete log entries ──
    deleted_log = 0
    try:
        if all_files:
            sb.table("pdf_ingestion_log") \
              .delete() \
              .neq("id", "00000000-0000-0000-0000-000000000000") \
              .execute()
            deleted_log = len(rows)
        elif category:
            sb.table("pdf_ingestion_log").delete().eq("category", category).execute()
            deleted_log = len(rows)
        else:
            for r in rows:
                sb.table("pdf_ingestion_log").delete().eq("filename", r["filename"]).execute()
                deleted_log += 1
    except Exception as e:
        print(f"  ❌ Log deletion failed: {e}")
        return

    print(f"\n  ✅ Deleted {deleted_log} log entry/entries and ~{deleted_vectors} vector chunk(s).")


# ══════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Upload PDF research files to Supabase pgvector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
INDIVIDUAL UPLOAD
  python upload_pdfs.py paper.pdf --title "My Paper" --category Agriculture --year 2024
  python upload_pdfs.py paper.pdf --interactive
  python upload_pdfs.py paper.pdf --dry-run

BATCH UPLOAD (metadata file)
  python upload_pdfs.py --meta metadata.txt
  python upload_pdfs.py --folder ./research/ --meta metadata.txt
  python upload_pdfs.py *.pdf --meta metadata.txt

DELETE
  python upload_pdfs.py --delete paper.pdf
  python upload_pdfs.py --delete paper.pdf health_survey.pdf
  python upload_pdfs.py --delete --category Agriculture
  python upload_pdfs.py --delete --all
  python upload_pdfs.py --delete paper.pdf --dry-run

OTHER
  python upload_pdfs.py --list
  python upload_pdfs.py paper.pdf --no-skip
        """
    )

    # ── Input sources ──
    parser.add_argument("files",    nargs="*", metavar="FILE",
                        help="One or more PDF files (or glob e.g. *.pdf)")
    parser.add_argument("--folder", metavar="DIR",
                        help="Scan a folder recursively for PDFs")
    parser.add_argument("--meta",   metavar="FILE",
                        help="Path to a metadata .txt file for batch uploads")

    # ── Per-file metadata ──
    meta_group = parser.add_argument_group(
        "individual metadata",
        "Used for single-file uploads, multiple files without --meta, "
        "or as a fallback when a file is missing from the metadata file."
    )
    meta_group.add_argument("--title",       help="Paper title (never auto-filled from filename)")
    meta_group.add_argument("--category",    help="e.g. Agriculture, Health, Education")
    meta_group.add_argument("--year",        type=int, help="Publication year e.g. 2024")
    meta_group.add_argument("--authors",     help="e.g. 'Juan dela Cruz, Maria Santos'")
    meta_group.add_argument(
        "--interactive", action="store_true",
        help="Prompt for metadata interactively for each file (cannot combine with --meta)"
    )

    # ── Chunking ──
    chunk_group = parser.add_argument_group("chunking options")
    chunk_group.add_argument("--chunk-size",    type=int, default=800,
                             help="Characters per chunk (default: 800)")
    chunk_group.add_argument("--chunk-overlap", type=int, default=100,
                             help="Overlap between chunks (default: 100)")

    # ── Delete ──
    delete_group = parser.add_argument_group(
        "delete",
        "Remove files from both the ingestion log and the vector store. "
        "Always prompts for confirmation unless --dry-run."
    )
    delete_group.add_argument(
        "--delete", action="store_true",
        help="Delete mode — removes named files, a category, or all files"
    )
    delete_group.add_argument(
        "--all", action="store_true",
        help="Used with --delete: delete every file in the database"
    )

    # ── Behaviour flags ──
    parser.add_argument("--no-skip",  action="store_true",
                        help="Re-ingest even if already in the log")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Preview only — nothing stored, logged, or deleted")
    parser.add_argument("--list",     action="store_true",
                        help="List all uploaded PDFs and exit")

    args = parser.parse_args()

    if args.interactive and args.meta:
        print("❌  --interactive and --meta cannot be used together. Pick one.")
        sys.exit(1)

    if args.delete and args.interactive:
        print("❌  --delete and --interactive cannot be used together.")
        sys.exit(1)

    if args.delete and not args.files and not args.category and not args.all:
        print("❌  --delete requires at least one filename, --category, or --all.")
        print("   Examples:")
        print("     python upload_pdfs.py --delete paper.pdf")
        print("     python upload_pdfs.py --delete --category Agriculture")
        print("     python upload_pdfs.py --delete --all")
        sys.exit(1)

    print("=" * 62)
    print("📚  PDF Research Uploader")
    print(f"    Supabase  : {SUPABASE_URL[:48]}...")
    print(f"    Model     : {OLLAMA_EMBED_MODEL}")
    print(f"    Collection: {VECTOR_COLLECTION}")
    if args.dry_run:     print("    ⚠️  DRY RUN — nothing will be written")
    if args.interactive: print("    💬 INTERACTIVE MODE — will prompt for each file")
    if args.delete:      print("    🗑️  DELETE MODE")
    print("=" * 62)

    from supabase import create_client
    from langchain_ollama import OllamaEmbeddings
    from langchain_postgres import PGVector

    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    if args.list:
        list_uploads(sb)
        return

    if args.delete:
        delete_pdfs(
            filenames=[Path(f).name for f in args.files],
            category=args.category,
            all_files=args.all,
            dry_run=args.dry_run,
            sb=sb,
        )
        return

    store = PGVector(
        embeddings=OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL),
        collection_name=VECTOR_COLLECTION,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    # ── Load metadata file ──
    meta_map: dict = {}
    if args.meta:
        meta_path = Path(args.meta)
        meta_map = parse_metadata_file(meta_path)
        print(f"\n📋 Loaded metadata for {len(meta_map)} file(s) from {meta_path.name}:")
        for fname, m in meta_map.items():
            print(f"   • {fname}")
            print(f"     title={m['title']!r}, category={m['category']!r}, "
                  f"year={m['year']}, authors={m['authors']!r}")

    # ── Collect PDF paths ──
    pdf_paths: list[Path] = []

    if args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"❌ Folder not found: {folder}")
            sys.exit(1)
        found = sorted(folder.rglob("*.pdf")) + sorted(folder.rglob("*.PDF"))
        pdf_paths.extend(found)
        print(f"\n📁 Found {len(found)} PDF(s) in {folder}")

    for f in args.files:
        p = Path(f)
        if p.is_dir():
            pdf_paths.extend(sorted(p.rglob("*.pdf")))
        else:
            pdf_paths.append(p)

    # If only --meta given (no explicit files/folder), use filenames from the metadata
    if not pdf_paths and meta_map:
        pdf_paths = [Path(fname) for fname in meta_map.keys()]
        print(f"\n📂 Using {len(pdf_paths)} file(s) listed in metadata file")

    if not pdf_paths:
        print("❌ No PDF files specified.\n")
        parser.print_usage()
        sys.exit(1)

    # ── Warn on individual upload with no title ──
    if not args.meta and not args.interactive and not args.title:
        if len(pdf_paths) == 1:
            print(f"\n  ⚠️  No --title provided. Title will be stored as NULL.")
            print(f"     Tip: use --title \"Your Title\" or --interactive to set it.")
        else:
            print(f"\n  ⚠️  No --title or --meta provided. Titles will be stored as NULL.")
            print(f"     Tip: use --meta metadata.txt for batch uploads with individual titles.")

    # ── CLI fallback meta (when a file is not in --meta) ──
    cli_meta = {
        "title":    args.title,
        "category": args.category,
        "year":     args.year,
        "authors":  args.authors,
    }

    print(f"\nProcessing {len(pdf_paths)} file(s)...")
    total, success, skipped, failed = 0, 0, 0, 0

    for pdf_path in pdf_paths:
        # ── Resolve metadata for this file (priority: --interactive > --meta > CLI flags) ──
        if args.interactive:
            file_meta = prompt_meta(pdf_path)

        elif meta_map:
            file_meta = lookup_meta(pdf_path.name, meta_map)
            if file_meta is None:
                print(f"\n  ⚠️  '{pdf_path.name}' not listed in the metadata file.")
                if any(v is not None for v in cli_meta.values()):
                    print(f"      Falling back to CLI flags.")
                    file_meta = cli_meta
                else:
                    print(f"      No CLI flags set — title/category/year/authors will be NULL.")
                    file_meta = {}

        else:
            file_meta = cli_meta

        try:
            n = ingest_pdf(
                pdf_path=pdf_path,
                meta=file_meta,
                skip_dupes=not args.no_skip,
                dry_run=args.dry_run,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                sb=sb,
                store=store,
            )
            if n > 0:
                total += n
                success += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"     ❌ Unexpected error: {e}")
            failed += 1

    print("\n" + "=" * 62)
    print(f"✅  Done!  uploaded={success}  skipped={skipped}  failed={failed}  chunks={total}")
    print("=" * 62)


if __name__ == "__main__":
    main()
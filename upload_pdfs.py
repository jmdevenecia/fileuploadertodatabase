#!/usr/bin/env python3
"""
upload_pdfs.py — Standalone CLI to upload PDF research files into Supabase pgvector.
Completely independent from the chatbot codebase. Run from anywhere.

Setup (one time):
    pip install pypdf langchain langchain-community langchain-postgres \
                langchain-ollama langchain-core supabase psycopg2-binary \
                pgvector python-dotenv

    Copy .env.example to .env in the same folder as this script and fill in values.

Usage:
    python upload_pdfs.py paper.pdf
    python upload_pdfs.py *.pdf --category Agriculture --year 2024
    python upload_pdfs.py --folder ./research/
    python upload_pdfs.py paper.pdf --dry-run
    python upload_pdfs.py --list
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


def extract_text(pdf_path: Path) -> str:
    import pypdf
    reader = pypdf.PdfReader(str(pdf_path))
    pages = [p.extract_text() for p in reader.pages]
    return "\n\n".join(t.strip() for t in pages if t and t.strip())


def file_md5(pdf_path: Path) -> str:
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
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
            "checksum":    checksum,
            "chunk_count": chunk_count,
            "title":       meta.get("title"),
            "category":    meta.get("category"),
            "year":        meta.get("year"),
            "authors":     meta.get("authors"),
        }).execute()
    except Exception as e:
        print(f"  ⚠️  Log write failed: {e}")


def ingest_pdf(pdf_path, meta, skip_dupes, dry_run, chunk_size, chunk_overlap, sb, store) -> int:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    print(f"\n  📄 {pdf_path.name}")

    if not pdf_path.exists():
        print(f"     ❌ File not found"); return 0
    if pdf_path.suffix.lower() != ".pdf":
        print(f"     ⚠️  Not a PDF — skipping"); return 0

    checksum = file_md5(pdf_path)
    if skip_dupes and already_uploaded(checksum, sb):
        print(f"     ⏭️  Already uploaded — skipping (use --no-skip to force)"); return 0

    print(f"     📖 Extracting text...", end=" ", flush=True)
    try:
        text = extract_text(pdf_path)
    except Exception as e:
        print(f"❌\n     {e}"); return 0

    if not text.strip():
        print("❌\n     No text found — PDF may be image/scanned. Use OCR first.")
        return 0
    print(f"{len(text):,} chars")

    title = meta.get("title") or pdf_path.stem.replace("_", " ").replace("-", " ").title()
    doc_meta = {"source": pdf_path.name, "title": title,
                "source_table": "pdf_upload", "file_type": "pdf", "checksum": checksum}
    if meta.get("category"): doc_meta["category"] = meta["category"]
    if meta.get("year"):     doc_meta["year"]     = str(meta["year"])
    if meta.get("authors"):  doc_meta["authors"]  = meta["authors"]

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_documents([Document(page_content=text, metadata=doc_meta)])
    print(f"     ✂️  {len(chunks)} chunks")

    if dry_run:
        print(f"     🔍 DRY RUN — skipping embed")
        print(f"     Preview: {chunks[0].page_content[:120].strip()}...")
        return len(chunks)

    stored = 0
    for i in range(0, len(chunks), 10):
        store.add_documents(chunks[i:i + 10])
        stored += len(chunks[i:i + 10])
        print(f"\r     🧠 Embedding... {int(stored/len(chunks)*100)}% ({stored}/{len(chunks)})", end="", flush=True)
    print(f"\r     ✅ {len(chunks)} chunks stored" + " " * 30)

    log_upload(pdf_path, checksum, len(chunks), {**meta, "title": title}, sb)
    return len(chunks)


def list_uploads(sb):
    try:
        rows = sb.table("pdf_ingestion_log") \
            .select("filename, title, category, year, chunk_count, created_at") \
            .order("created_at", desc=True).execute().data
    except Exception as e:
        print(f"❌ {e}"); return

    if not rows:
        print("No PDFs uploaded yet."); return

    print(f"\n{'FILENAME':<35} {'TITLE':<28} {'CATEGORY':<14} {'YEAR':<6} {'CHUNKS':<8} DATE")
    print("─" * 105)
    for r in rows:
        print(f"{r.get('filename',''):<35} {(r.get('title') or '')[:26]:<28} "
              f"{(r.get('category') or ''):<14} {str(r.get('year') or ''):<6} "
              f"{str(r.get('chunk_count') or ''):<8} {(r.get('created_at') or '')[:10]}")
    print(f"\nTotal: {len(rows)} file(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Upload PDF research files to Supabase pgvector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upload_pdfs.py paper.pdf
  python upload_pdfs.py *.pdf --category Agriculture --year 2024 --authors "Juan dela Cruz"
  python upload_pdfs.py --folder ./research/ --category Health
  python upload_pdfs.py paper.pdf --dry-run
  python upload_pdfs.py --list
  python upload_pdfs.py paper.pdf --no-skip
        """
    )
    parser.add_argument("files",          nargs="*", metavar="FILE")
    parser.add_argument("--folder",       metavar="DIR",  help="Scan folder recursively for PDFs")
    parser.add_argument("--title",                        help="Title (auto-detected if omitted)")
    parser.add_argument("--category",                     help="e.g. Agriculture, Health, Education")
    parser.add_argument("--year",         type=int,       help="Publication year e.g. 2024")
    parser.add_argument("--authors",                      help="e.g. 'Juan dela Cruz, Maria Santos'")
    parser.add_argument("--chunk-size",   type=int, default=800)
    parser.add_argument("--chunk-overlap",type=int, default=100)
    parser.add_argument("--no-skip",      action="store_true", help="Re-ingest even if already uploaded")
    parser.add_argument("--dry-run",      action="store_true", help="Preview only — nothing stored")
    parser.add_argument("--list",         action="store_true", help="List all uploaded PDFs")
    args = parser.parse_args()

    print("=" * 60)
    print("📚  PDF Research Uploader")
    print(f"    Supabase  : {SUPABASE_URL[:45]}...")
    print(f"    Model     : {OLLAMA_EMBED_MODEL}")
    print(f"    Collection: {VECTOR_COLLECTION}")
    if args.dry_run: print("    ⚠️  DRY RUN")
    print("=" * 60)

    from supabase import create_client
    from langchain_ollama import OllamaEmbeddings
    from langchain_postgres import PGVector

    sb    = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    store = PGVector(
        embeddings=OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL),
        collection_name=VECTOR_COLLECTION,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    if args.list:
        list_uploads(sb); return

    pdf_paths = []
    if args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"❌ Folder not found: {folder}"); sys.exit(1)
        pdf_paths = sorted(folder.rglob("*.pdf")) + sorted(folder.rglob("*.PDF"))
        print(f"📁 Found {len(pdf_paths)} PDF(s) in {folder}")

    for f in args.files:
        p = Path(f)
        pdf_paths.extend(sorted(p.rglob("*.pdf")) if p.is_dir() else [p])

    if not pdf_paths:
        print("❌ No PDF files specified.\n")
        parser.print_usage(); sys.exit(1)

    meta = {"title": args.title, "category": args.category,
            "year": args.year, "authors": args.authors}

    print(f"\nProcessing {len(pdf_paths)} file(s)...")
    total, success, skipped, failed = 0, 0, 0, 0

    for pdf_path in pdf_paths:
        try:
            n = ingest_pdf(
                pdf_path, meta,
                skip_dupes=not args.no_skip,
                dry_run=args.dry_run,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                sb=sb, store=store,
            )
            if n > 0: total += n; success += 1
            else: skipped += 1
        except Exception as e:
            print(f"     ❌ {e}"); failed += 1

    print("\n" + "=" * 60)
    print(f"✅  Done!  uploaded={success}  skipped={skipped}  failed={failed}  chunks={total}")
    print("=" * 60)


if __name__ == "__main__":
    main()

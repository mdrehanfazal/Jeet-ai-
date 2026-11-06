"""Tiny TF-IDF semantic index over the project — zero heavy deps, fully offline.

The index maps every file to a normalized term vector; search returns the
top-k files by cosine similarity plus the lines that matched, highlighted.
Persisted to .jeet/index.json so it survives restarts.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

from ..config import ROOT
from .parser import IGNORED_DIRS, LANGUAGE_BY_EXT, MAX_FILE_SIZE, TEXT_EXTS

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
MAX_TOKENS_PER_FILE = 20_000
MAX_VOCAB = 30_000


def index_path() -> Path:
    env = os.environ.get("JEET_INDEX_PATH")
    path = Path(env) if env else ROOT / ".jeet" / "index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text):
        tokens.append(match.group(0).lower())
        if len(tokens) >= MAX_TOKENS_PER_FILE:
            break
    return tokens


def _walk_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".git")]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() not in TEXT_EXTS and path.name.lower() != "dockerfile":
                continue
            try:
                if path.stat().st_size > MAX_FILE_SIZE:
                    continue
                yield path
            except OSError:
                continue


def build_index(root: Path | str) -> dict:
    root = Path(root).resolve()
    started = time.time()
    documents: dict[str, dict[str, int]] = {}
    doc_count = 0
    for path in _walk_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        tokens = _tokenize(text + " " + path.as_posix().replace("/", " "))
        if not tokens:
            continue
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        documents[path.relative_to(root).as_posix()] = tf
        doc_count += 1

    df: dict[str, int] = {}
    for tf in documents.values():
        for term in tf:
            df[term] = df.get(term, 0) + 1

    # Keep the most informative vocab terms if the corpus is huge
    if len(df) > MAX_VOCAB:
        keep = sorted(df.items(), key=lambda kv: kv[1], reverse=True)[:MAX_VOCAB]
        allowed = {t for t, _ in keep}
    else:
        allowed = set(df)

    idf = {term: math.log(1 + doc_count / (df[term] + 1)) + 1.0 for term in allowed}

    docs_out: dict[str, dict[str, float]] = {}
    for rel, tf in documents.items():
        vec: dict[str, float] = {}
        for term, freq in tf.items():
            if term in idf:
                vec[term] = freq * idf[term]
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        docs_out[rel] = {term: round(w / norm, 6) for term, w in vec.items()}

    index = {
        "root": str(root),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "doc_count": doc_count,
        "build_seconds": round(time.time() - started, 3),
        "idf": {t: round(v, 6) for t, v in idf.items()},
        "docs": docs_out,
    }
    with index_path().open("w", encoding="utf-8") as fh:
        json.dump(index, fh)
    return {"root": index["root"], "doc_count": doc_count, "terms": len(idf),
            "build_seconds": index["build_seconds"], "built_at": index["built_at"]}


def _load_index(root: Path) -> dict | None:
    path = index_path()
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            index = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if Path(index.get("root", "")).resolve() != root.resolve():
        return None
    return index


def index_status(root: Path | str) -> dict:
    path = index_path()
    exists = path.exists()
    index = _load_index(Path(root)) if exists else None
    return {
        "indexed": index is not None,
        "path": str(path),
        "root": (index or {}).get("root"),
        "doc_count": (index or {}).get("doc_count", 0),
        "built_at": (index or {}).get("built_at"),
    }


def _snippet(root: Path, rel: str, query_tokens: set[str], max_lines: int = 3) -> list[dict]:
    try:
        lines = (root / rel).read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    hits = []
    for i, line in enumerate(lines):
        low = line.lower()
        if any(tok in low for tok in query_tokens):
            hits.append({"n": i + 1, "text": line.rstrip()[:220]})
        if len(hits) >= max_lines:
            break
    if not hits:
        hits = [{"n": i + 1, "text": lines[i].rstrip()[:220]} for i in range(min(max_lines, len(lines)))]
    return hits


def search(query: str, root: Path | str, k: int = 6, auto_build: bool = True) -> list[dict]:
    root = Path(root).resolve()
    index = _load_index(root)
    if index is None:
        if not auto_build:
            return []
        build_index(root)
        index = _load_index(root)
        if index is None:
            return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    qtf: dict[str, int] = {}
    for tok in query_tokens:
        if tok in index["idf"]:
            qtf[tok] = qtf.get(tok, 0) + 1
    if not qtf:
        return []
    qvec = {t: f * index["idf"][t] for t, f in qtf.items()}
    qnorm = math.sqrt(sum(w * w for w in qvec.values())) or 1.0
    qvec = {t: w / qnorm for t, w in qvec.items()}

    scores: list[tuple[str, float]] = []
    for rel, doc_vec in index["docs"].items():
        score = 0.0
        # iterate the smaller side
        if len(qvec) <= len(doc_vec):
            for term, qw in qvec.items():
                dw = doc_vec.get(term)
                if dw:
                    score += qw * dw
        else:
            for term, dw in doc_vec.items():
                qw = qvec.get(term)
                if qw:
                    score += qw * dw
        if score > 0.02:
            scores.append((rel, score))
    scores.sort(key=lambda item: item[1], reverse=True)

    token_set = set(query_tokens)
    results = []
    for rel, score in scores[:k]:
        results.append({
            "path": rel,
            "score": round(score, 4),
            "snippet": _snippet(root, rel, token_set),
            "language": LANGUAGE_BY_EXT.get(Path(rel).suffix.lower(), "text"),
        })
    return results

"""Extract page text and build a document-only dense index.

python -m rag.build --extract-only
python -m rag.build
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np

from .clients import Embeddings, ROOT


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def extract_pages():
    output = ROOT / 'artifacts'
    output.mkdir(exist_ok=True)
    pages = []
    empty = []
    for doc in read_jsonl(ROOT / 'data/documents.jsonl'):
        pdf = ROOT / 'data/reports' / doc['path']
        with fitz.open(pdf) as source:
            if len(source) != doc['pages']:
                raise ValueError('Page count differs for ' + doc['doc_id'])
            for number, page in enumerate(source, 1):
                text = page.get_text('text', sort=True).strip()
                method = 'text'
                if len(text) < 60:
                    # OCR is a separate resumable step: python -m rag.ocr.
                    empty.append({'doc_id':doc['doc_id'], 'page':number, 'error':'OCR required'})
                pages.append({'doc_id':doc['doc_id'], 'page':number,
                              'organization':doc['organization'], 'text':text, 'method':method})
        print('Extracted', doc['doc_id'], len(source) if not source.is_closed else doc['pages'], flush=True)
    (output / 'pages.jsonl').write_text('\n'.join(json.dumps(p, ensure_ascii=False) for p in pages)+'\n')
    (output / 'extraction_issues.json').write_text(json.dumps(empty, ensure_ascii=False, indent=2))
    print('Pages:',len(pages), 'OCR issues:',len(empty), flush=True)
    return pages


def make_chunks(pages):
    chunks = []
    current = None
    for idx, page in enumerate(pages):
        if not page['text'].strip():
            continue
        # Never cross documents. Up to three pages retain nearby table headers.
        prefix = '%s; %s; страница PDF %d\n' % (page['organization'], page['doc_id'], page['page'])
        parts = [page['text'][offset:offset+15000] for offset in range(0,len(page['text']),15000)]
        for part in parts:
            text = prefix + part
            if (current is None or current['doc_id'] != page['doc_id'] or
                    len(current['page_indices']) >= 3 or len(current['text']) + len(text) + 2 > 18000):
                current = {'doc_id':page['doc_id'], 'page_indices':[], 'text':''}
                chunks.append(current)
            current['page_indices'].append(idx)
            current['text'] += ('\n\n' if current['text'] else '') + text
    return chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--extract-only', action='store_true')
    ap.add_argument('--reextract', action='store_true')
    args = ap.parse_args()
    path = ROOT / 'artifacts/pages.jsonl'
    pages = extract_pages() if args.reextract or not path.exists() else read_jsonl(path)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.extract_only:
        return
    issues = path.parent/'extraction_issues.json'
    if issues.exists() and json.loads(issues.read_text()):
        raise ValueError('Run python -m rag.ocr before building the index')
    chunks = make_chunks(pages)
    print('Embedding chunks:', len(chunks), flush=True)
    api = Embeddings()
    vectors = [None] * len(chunks)
    failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(api.embed,c['text']):i for i,c in enumerate(chunks)}
        for n,f in enumerate(as_completed(futures),1):
            try:
                vectors[futures[f]] = f.result()
            except Exception as exc:
                failures.append(futures[f])
                print('Chunk %d failed: %s' % (futures[f],type(exc).__name__),flush=True)
            if n % 25 == 0 or n == len(chunks):
                print('Processed chunks %d/%d' % (n,len(chunks)),flush=True)
    if failures:
        raise RuntimeError('%d chunks failed; rerun to retry only missing embeddings' % len(failures))
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('Page extraction changed during embedding. Rerun; embeddings are cached.')
    np.save(ROOT/'artifacts/vectors.npy', np.stack(vectors))
    (ROOT/'artifacts/chunks.json').write_text(json.dumps(chunks,ensure_ascii=False))
    (ROOT/'artifacts/index_manifest.json').write_text(json.dumps({
        'pages_sha256':source_hash,
        'pages':len(pages), 'chunks':len(chunks), 'dimensions':1024,
        'source':'PDF reports only; no questions or gold answers',
    },ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

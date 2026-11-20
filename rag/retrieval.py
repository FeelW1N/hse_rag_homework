"""Hybrid semantic/BM25 retrieval; no access to train/test annotations."""
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi
import snowballstemmer

from .build import read_jsonl
from .clients import Embeddings, ROOT

STEMMER = snowballstemmer.stemmer('russian')
STOP = set('как каков какой каковы какие сколько согласно составил составляет составили год года году годы отчет отчетность консолидированный финансовый банк группа по на в и за из с к у от для ли это об его ее а или'.split())
BANKS = {
    'sber': r'сбер', 'vtb': r'втб|\bvtb\b', 'tbank':r'т[ -]?банк|тинькофф|t[ -]?bank',
    'domrf':r'дом\s*[. -]?\s*рф', 'mkb':r'\bмкб\b|московск\w* кредитн',
    'rshb':r'россельхоз|\bрсхб\b', 'sovkom':r'совком',
    'alfa':r'альфа|\balfa\b', 'gpb':r'газпромбанк|\bнпф\b|\bgpb\b',
}


@lru_cache(maxsize=150000)
def stem(word):
    return STEMMER.stemWord(word)


def tokenize(text):
    return [stem(w) for w in re.findall(r'[а-яa-z]+|20\d{2}', text.lower().replace('ё','е')) if w not in STOP]


QUERY_STOP = {stem(w) for w in (STOP | set('общий объем объём состояние декабрь июнь обобщенный промежуточный консолидированный финансовый отчетность организация российский рубль ао пао зао'.split()))}


def query_tokens(text):
    # Organization names are metadata filters, not indicators to rank by frequency.
    for pattern in BANKS.values():
        text=re.sub(pattern,' ',text,flags=re.IGNORECASE)
    return [w for w in tokenize(text) if w not in QUERY_STOP and not re.fullmatch(r'20\d{2}',w)]


def mentioned_banks(question):
    return [key for key, pattern in BANKS.items() if re.search(pattern, question.lower())]


class Retriever:
    def __init__(self, dense=True):
        self.pages = read_jsonl(ROOT/'artifacts/pages.jsonl')
        self.bm25 = BM25Okapi([tokenize(p['text']) for p in self.pages])
        self.dense = dense
        if dense:
            manifest=json.loads((ROOT/'artifacts/index_manifest.json').read_text())
            if manifest['pages_sha256']!=hashlib.sha256((ROOT/'artifacts/pages.jsonl').read_bytes()).hexdigest():
                raise ValueError('Pages changed after indexing; rebuild index')
            self.chunks = json.loads((ROOT/'artifacts/chunks.json').read_text())
            self.vectors = np.load(ROOT/'artifacts/vectors.npy')
            if self.vectors.shape != (len(self.chunks), 1024):
                raise ValueError('Index shape mismatch; rebuild index')
            self.embeddings = Embeddings()

    def search(self, question, k=8):
        banks = mentioned_banks(question)
        allowed = np.array([not banks or p['doc_id'].split('_')[0] in banks for p in self.pages])
        # If a naming convention changes, never silently retrieve zero documents.
        if not allowed.any():
            raise ValueError('Requested organization has no indexed documents')
        years = set(re.findall(r'20\d{2}', question))
        # Prefer the explicitly named report year over dates in comparative columns.
        report_years = re.findall(r'отч[её]тност[а-я]*[^.!?]{0,45}?за\s+(20\d{2})',question.lower())
        wanted_years = set(report_years) if report_years else years
        if wanted_years:
            narrowed = np.array([allowed[i] and p['doc_id'].split('_')[1] in wanted_years for i,p in enumerate(self.pages)])
            if narrowed.any():
                allowed = narrowed
        half_year = bool(re.search(r'30\s+июня|шесть\s+месяц|6\s+месяц|полугоди',question.lower()))
        annual = bool(re.search(r'31\s+декабря|годов',question.lower()))
        if half_year and not annual:
            narrowed = np.array([allowed[i] and p['doc_id'].endswith('_6m') for i,p in enumerate(self.pages)])
            # Keep banks that have only annual documents in the collection.
            half_banks={p['doc_id'].split('_')[0] for i,p in enumerate(self.pages) if narrowed[i]}
            if narrowed.any():
                allowed=np.array([allowed[i] and (p['doc_id'].endswith('_6m') or p['doc_id'].split('_')[0] not in half_banks) for i,p in enumerate(self.pages)])
        if annual and not half_year and 'промежуточ' not in question.lower():
            narrowed=np.array([allowed[i] and p['doc_id'].endswith('_annual') for i,p in enumerate(self.pages)])
            if narrowed.any():
                allowed=narrowed
        lexical = np.asarray(self.bm25.get_scores(query_tokens(question)))
        lexical[~allowed] = -np.inf
        scores = np.zeros(len(self.pages))
        ranked = np.argsort(-lexical, kind='stable')[:80]
        for rank, i in enumerate(ranked):
            if allowed[i] and lexical[i] > 0:
                scores[i] += 1 / (20 + rank)
        if self.dense:
            # Keep timing comparable between first and repeated evaluations.
            vector = self.embeddings.embed(question, use_cache=False)
            # Avoid platform-specific BLAS warnings on macOS; 1.4M multiplies is small.
            similarity = np.einsum('ij,j->i',self.vectors,vector,optimize=False)
            for rank, i in enumerate(sorted(
                (i for i,c in enumerate(self.chunks) if any(allowed[j] for j in c['page_indices'])),
                key=lambda i: -float(similarity[i]),
            )[:60]):
                for page_index in self.chunks[i]['page_indices']:
                    scores[page_index] += 0.55 / (20 + rank)
        # A year is a soft preference: reports often contain comparative years.
        for i,p in enumerate(self.pages):
            if p['doc_id'].split('_')[1] in years:
                scores[i] *= 1.25
        scores[~allowed] = -np.inf
        ranked = [int(i) for i in np.argsort(-scores, kind='stable') if scores[i] > 0]
        chosen = []
        if len(banks) > 1:
            for bank in banks:
                candidate = next((i for i in ranked if self.pages[i]['doc_id'].startswith(bank+'_')), None)
                if candidate is not None:
                    chosen.append(candidate)
        chosen += [i for i in ranked if i not in chosen][:max(0,k-len(chosen))]
        return [{'index':i, **self.pages[i], 'retrieval_score':float(scores[i])} for i in chosen]


def context_from_pages(pages):
    docs = defaultdict(set)
    for p in pages:
        docs[p['doc_id']].add(p['page'])
    return [{'doc_id':d, 'pages':sorted(ps)} for d,ps in sorted(docs.items())]

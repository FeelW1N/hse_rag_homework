"""Course APIs with bounded retries, rate limiting and an embedding cache."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time

import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')


class Limiter:
    def __init__(self, rpm):
        self.interval = 60 / rpm
        self.next = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            reserved = max(now, self.next)
            self.next = reserved + self.interval
        time.sleep(max(0, reserved - now))


EMB_LIMIT = Limiter(55)
LLM_LIMIT = Limiter(float(os.getenv('LLM_RPM', '9')))


def post(url, *, headers, payload, limiter, attempts=6):
    for attempt in range(attempts):
        limiter.wait()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=(15, 120))
            if response.status_code not in (408, 429, 500, 502, 503, 504, 529):
                response.raise_for_status()
                return response.json()
            delay = float(response.headers.get('Retry-After', min(60, 2 ** attempt)))
            if delay > 120:
                raise RuntimeError('API limit: Retry-After=%ss; resume later' % delay)
        except (requests.Timeout, requests.ConnectionError):
            if attempt == attempts - 1:
                raise
            delay = min(60, 2 ** attempt)
        if attempt == attempts - 1:
            raise RuntimeError('API did not recover after %d attempts' % attempts)
        time.sleep(max(1, delay))
    raise RuntimeError('Unreachable retry state')


class Embeddings:
    def __init__(self, cache=None):
        self.path = Path(cache or ROOT / 'artifacts' / 'embeddings.sqlite')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)')

    def embed(self, text, use_cache=True):
        if not text.strip() or len(text) > 20000:
            raise ValueError('Embedding input must contain 1..20000 characters')
        key = hashlib.sha256(text.encode()).hexdigest()
        with sqlite3.connect(self.path, timeout=30) as db:
            record = db.execute('SELECT vector FROM embeddings WHERE key=?', (key,)).fetchone()
        if record and use_cache:
            return np.array(json.loads(record[0]), dtype=np.float32)
        value = post(
            'https://embeddings.parsers360.ru:10443/embed',
            headers={'Authorization': 'Bearer ' + os.environ['EMB_KEY']},
            payload={'text': text}, limiter=EMB_LIMIT,
        )
        vector = np.asarray(value['embedding'], dtype=np.float32)
        if vector.shape != (1024,) or not np.isfinite(vector).all():
            raise ValueError('Invalid embedding from provider')
        norm = float(np.linalg.norm(vector))
        if norm < 1e-10:
            raise ValueError('Zero embedding')
        vector /= norm
        if use_cache:
            with sqlite3.connect(self.path, timeout=30) as db:
                db.execute('INSERT OR REPLACE INTO embeddings VALUES (?,?)', (key, json.dumps(vector.tolist())))
        return vector


def generate(system, question, max_tokens=1400):
    result = post(
        'https://claude.gigaparsers.ru/v1/messages',
        headers={'x-api-key': os.environ['LLM_API_KEY'], 'anthropic-version': '2023-06-01'},
        payload={'model': 'deepseek-flash', 'temperature': 0,
                 'thinking': {'type': 'disabled'}, 'max_tokens': max_tokens,
                 'system': system, 'messages': [{'role': 'user', 'content': question}]},
        limiter=LLM_LIMIT,
    )
    if result.get('stop_reason') == 'max_tokens':
        raise ValueError('Generation was truncated')
    text = ''.join(x['text'] for x in result['content'] if x['type'] == 'text').strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    return json.loads(text)

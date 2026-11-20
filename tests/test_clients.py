import numpy as np
import pytest

from rag import clients


def test_document_embeddings_cached_but_question_timing_uses_fresh_request(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setenv('EMB_KEY','test-key')
    def fake_post(*args,**kwargs):
        calls.append(kwargs['payload']['text'])
        return {'embedding':[1.0]*1024}
    monkeypatch.setattr(clients,'post',fake_post)
    api=clients.Embeddings(tmp_path/'cache.sqlite')
    first=api.embed('Document')
    assert np.linalg.norm(first)==pytest.approx(1.0)
    np.testing.assert_array_equal(api.embed('Document'),first)
    assert calls==['Document']
    api.embed('Document',use_cache=False)
    assert calls==['Document','Document']
    with pytest.raises(ValueError,match='1..20000'):
        api.embed('x'*20001)


def test_invalid_provider_vector_is_not_cached(tmp_path,monkeypatch):
    monkeypatch.setenv('EMB_KEY','test-key')
    monkeypatch.setattr(clients,'post',lambda *args,**kwargs:{'embedding':[0.0]*1024})
    api=clients.Embeddings(tmp_path/'cache.sqlite')
    with pytest.raises(ValueError,match='Zero'):
        api.embed('Document')

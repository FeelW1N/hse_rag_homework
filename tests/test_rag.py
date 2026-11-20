import pytest
from rank_bm25 import BM25Okapi

from rag.build import make_chunks
from rag.retrieval import Retriever, context_from_pages, mentioned_banks, tokenize
from rag.system import RAG


def page(doc='sber_2024_annual', number=1, text='Содержание страницы'):
    return {'doc_id':doc,'page':number,'text':text,'organization':'Сбербанк'}


def test_chunks_keep_document_boundaries_and_api_size_limit():
    pages = [page(number=i,text='А'*9000) for i in range(1,6)]
    pages += [page('vtb_2024_annual',1,'Б'*25000)]
    chunks = make_chunks(pages)
    assert {i for c in chunks for i in c['page_indices']} == set(range(6))
    assert all(len(c['text']) <= 20000 for c in chunks)
    assert all(len({pages[i]['doc_id'] for i in c['page_indices']})==1 for c in chunks)


def test_metadata_and_stemming():
    assert mentioned_banks('Сравни ВТБ, Совкомбанк и Т-Банк') == ['vtb','tbank','sovkom']
    assert mentioned_banks('АО «Банк ДОМ.РФ»') == ['domrf']
    assert tokenize('кредиты') == tokenize('кредитов')
    assert context_from_pages([page(number=2),page(number=1),page(number=2)]) == [
        {'doc_id':'sber_2024_annual','pages':[1,2]}]


def test_report_year_and_half_year_filters_keep_correct_documents():
    retrieval=Retriever.__new__(Retriever)
    retrieval.dense=False
    retrieval.pages=[page(doc,1,'Комиссионные доходы') for doc in (
        'domrf_2024_annual','domrf_2025_6m','domrf_2025_annual','sber_2025_6m')]
    retrieval.bm25=BM25Okapi([tokenize(p['text']) for p in retrieval.pages]+[['other']]*10)
    # Supply scores for the four real pages without changing the filter logic.
    import numpy as np
    retrieval.bm25.get_scores=lambda query:np.array([1.,1.,1.,1.])
    result=retrieval.search('Комиссионные доходы ДОМ.РФ за шесть месяцев 2025 года?')
    assert [p['doc_id'] for p in result]==['domrf_2025_6m']
    result=retrieval.search('Доходы ДОМ.РФ за 2024 год согласно отчётности за 2025 год на 31 декабря?')
    assert [p['doc_id'] for p in result]==['domrf_2025_annual']


class FakeRetriever:
    def __init__(self):
        self.queries=[]

    def search(self,question,k):
        self.queries.append(question)
        return [page(number=len(self.queries))]


def test_agent_runs_second_search_only_when_requested_and_returns_used_pages():
    retrieval=FakeRetriever()
    answers=iter([
        {'answer':'','sources':[],'search_query':'недостающая сумма'},
        {'answer':'10 млн рублей','sources':[1,2],'search_query':''},
    ])
    rag=RAG(retriever=retrieval,llm=lambda *args:next(answers))
    result=rag.rag('Прибыль Сбербанка за 2024 год?')
    assert len(retrieval.queries)==2
    assert result['context']==[{'doc_id':'sber_2024_annual','pages':[1,2]}]
    assert rag.last_trace['additional_search']


def test_agent_does_not_search_when_evidence_is_sufficient():
    retrieval=FakeRetriever()
    rag=RAG(retriever=retrieval,llm=lambda *args:{'answer':'10','sources':[1],'search_query':''})
    assert rag.rag('Сумма?')['answer']=='10'
    assert len(retrieval.queries)==1


def test_model_cannot_invent_context_or_request_unbounded_search():
    rag=RAG(retriever=FakeRetriever(),llm=lambda *args:{'answer':'10','sources':[99]})
    with pytest.raises(ValueError,match='outside'):
        rag.rag('Сумма?')
    retrieval=FakeRetriever()
    rag=RAG(retriever=retrieval,llm=lambda *args:{'answer':'Нет данных','sources':[],'search_query':'ещё'})
    rag.rag('Сумма?')
    assert len(retrieval.queries)==2

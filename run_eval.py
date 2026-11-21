"""Evaluate RAG with the unchanged instructor judge; checkpoint every question."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import time

import requests
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from eval.eval_functions import score_answer, score_context
from rag.build import read_jsonl
from rag.clients import LLM_LIMIT, ROOT
from rag.system import RAG

METRICS = ['context_recall','context_precision','answer_correctness','time']
COLUMNS = ['question_id','question','gold_context','gold_answer','context','answer',*METRICS]


def fingerprint():
    paths = sorted((ROOT/'rag').glob('*.py')) + [ROOT/'eval/eval_functions.py',ROOT/'artifacts/index_manifest.json']
    h=hashlib.sha256()
    for p in paths:
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def judge(gold,answer):
    # Do not modify the instructor's model, prompt or scoring function.
    for attempt in range(6):
        LLM_LIMIT.wait()
        try:
            return score_answer(gold,answer)
        except requests.HTTPError as exc:
            response=exc.response
            if response.status_code not in (408,429,500,502,503,504,529) or attempt==5:
                raise
            delay=float(response.headers.get('Retry-After',min(60,2**attempt)))
            if delay>120:
                raise RuntimeError('Judge API limit: Retry-After=%ss; resume later' % delay) from exc
            time.sleep(max(1,delay))
        except (requests.Timeout,requests.ConnectionError,json.JSONDecodeError,ValueError):
            if attempt==5:
                raise
            time.sleep(min(60,2**attempt))


def export(rows,path):
    """Keep the instructor's two-sheet workbook and ten required columns."""
    if not rows:
        raise ValueError('No evaluated rows')
    report=Workbook()
    summary=report.active
    summary.title='Итоги'
    summary.append(METRICS)
    summary.append([mean(r[k] for r in rows) for k in METRICS])
    summary['A4']='Оценено вопросов'
    summary['B4']=len(rows)
    detail=report.create_sheet('По вопросам')
    detail.append(COLUMNS)
    for row in rows:
        values=[json.dumps(row[k],ensure_ascii=False) if isinstance(row[k],(dict,list)) else row[k] for k in COLUMNS]
        detail.append(values)
        for cell in detail[detail.max_row]:
            if isinstance(cell.value,str):
                cell.data_type='s'
    for sheet in report:
        sheet.freeze_panes='A2'
        sheet.auto_filter.ref=sheet.dimensions
        for cell in sheet[1]:
            cell.font=Font(bold=True,color='FFFFFF')
            cell.fill=PatternFill('solid',fgColor='24485B')
            cell.alignment=Alignment(wrap_text=True,vertical='center')
        sheet.row_dimensions[1].height=32
    for col,width in {'A':25,'B':65,'C':38,'D':65,'E':38,'F':65,'G':20,'H':20,'I':22,'J':16}.items():
        detail.column_dimensions[col].width=width
    for row in detail.iter_rows(min_row=2):
        for cell in row:
            cell.alignment=Alignment(vertical='top',wrap_text=True)
        # Show full source texts; the worksheet remains filterable.
        lengths=[len(str(c.value or '')) for c in row[:6]]
        detail.row_dimensions[row[0].row].height=min(409,max(45,15*(1+max(lengths[1]//60,lengths[3]//60,lengths[5]//60))))
        for cell in row[6:8]:
            cell.number_format='0.00%'
        row[8].number_format='0'
        row[9].number_format='0.00'
    for col in 'ABCD':
        summary.column_dimensions[col].width=24
    for cell in summary[2][:3]:
        cell.number_format='0.00%'
    summary['D2'].number_format='0.00'
    detail.freeze_panes='B2'
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    report.save(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',default='data/test.jsonl')
    ap.add_argument('--output',default='test_results.xlsx')
    ap.add_argument('--checkpoint',default='runs/test.jsonl')
    ap.add_argument('--limit',type=int)
    ap.add_argument('--export-only',action='store_true')
    args=ap.parse_args()
    dataset=read_jsonl(args.dataset)
    if args.limit:
        dataset=dataset[:args.limit]
    if len({item['question_id'] for item in dataset})!=len(dataset):
        raise ValueError('Duplicate dataset IDs')
    path=Path(args.checkpoint)
    prior=read_jsonl(path) if path.exists() else []
    version=fingerprint()
    if any(r.get('fingerprint')!=version for r in prior):
        raise ValueError('Checkpoint is from another RAG version; choose a new checkpoint path')
    by_id={r['question_id']:r for r in prior}
    if len(by_id)!=len(prior):
        raise ValueError('Duplicate checkpoint IDs')
    for item in dataset:
        if item['question_id'] in by_id and any(by_id[item['question_id']][k]!=item[k]
                                               for k in ('question','gold_context','gold_answer')):
            raise ValueError('Dataset changed since checkpoint')
    prediction_path=path.with_suffix('.predictions.jsonl')
    saved=read_jsonl(prediction_path) if prediction_path.exists() else []
    if any(r.get('fingerprint')!=version for r in saved):
        raise ValueError('Saved predictions are from another RAG version; choose a new checkpoint path')
    predictions={r['question_id']:r for r in saved}
    if len(predictions)!=len(saved):
        raise ValueError('Duplicate saved prediction IDs')
    if not args.export_only:
        rag=RAG()
        path.parent.mkdir(parents=True,exist_ok=True)
        for item in dataset:
            if item['question_id'] in by_id:
                if by_id[item['question_id']]['question']!=item['question']:
                    raise ValueError('Dataset changed since checkpoint')
                continue
            saved_prediction=predictions.get(item['question_id'])
            if saved_prediction is None:
                started=time.perf_counter()
                prediction=rag.rag(item['question'])
                elapsed=time.perf_counter()-started
                saved_prediction={'question_id':item['question_id'],'question':item['question'],
                                  **prediction,'time':elapsed,'trace':rag.last_trace,'fingerprint':version}
                with prediction_path.open('a') as f:
                    f.write(json.dumps(saved_prediction,ensure_ascii=False)+'\n')
            if saved_prediction['question']!=item['question']:
                raise ValueError('Dataset changed since prediction')
            prediction={k:saved_prediction[k] for k in ('answer','context')}
            elapsed=saved_prediction['time']
            context=score_context(item['gold_context'],prediction['context'])
            decision=judge(item['gold_answer'],prediction['answer'])
            row={k:item[k] for k in ('question_id','question','gold_context','gold_answer')}
            row.update(prediction)
            row.update(context)
            row.update(answer_correctness=decision['answer_correctness'],time=elapsed,
                       judge_reason=decision['reason'],trace=saved_prediction['trace'],fingerprint=version)
            with path.open('a') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            by_id[item['question_id']]=row
            print('%d/%d %s recall=%.3f precision=%.3f answer=%d time=%.2fs' %
                  (len(by_id),len(dataset),item['question_id'],context['context_recall'],
                   context['context_precision'],decision['answer_correctness'],elapsed),flush=True)
    if any(item['question_id'] not in by_id for item in dataset):
        raise ValueError('Checkpoint is incomplete; finish the evaluation before exporting')
    rows=[by_id[item['question_id']] for item in dataset]
    export(rows,args.output)
    print(json.dumps({k:mean(r[k] for r in rows) for k in METRICS},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

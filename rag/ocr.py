"""Resume OCR only for pages without a usable text layer."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np

from .build import read_jsonl
from .clients import ROOT


def recognize(task):
    import fitz
    cv2.setNumThreads(1)
    path, doc_id, number, dpi = task
    with fitz.open(path) as pdf:
        page=pdf[number-1]
        pix=page.get_pixmap(dpi=dpi,colorspace=fitz.csGRAY)
        img=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width).copy()
    binary=cv2.threshold(img,180,255,cv2.THRESH_BINARY_INV)[1]
    thumbnail=cv2.resize(binary,None,fx=0.25,fy=0.25,interpolation=cv2.INTER_NEAREST)
    lines=cv2.HoughLinesP(thumbnail,1,np.pi/1800,threshold=40,minLineLength=100,maxLineGap=5)
    angles=[np.degrees(np.arctan2(y2-y1,x2-x1)) for [[x1,y1,x2,y2]] in lines
            if abs(y2-y1)<0.05*abs(x2-x1)] if lines is not None else []
    angle=float(np.median(angles)) if angles else 0
    img=cv2.warpAffine(img,cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),angle,1),
                       (img.shape[1],img.shape[0]),borderValue=255)
    binary=cv2.threshold(img,180,255,cv2.THRESH_BINARY_INV)[1]
    horizontal=cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(60,1)))
    vertical=cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,60)))
    img[cv2.dilate(horizontal|vertical,np.ones((3,3),np.uint8))>0]=255
    result=subprocess.run(['tesseract','stdin','stdout','-l','rus+eng','--psm','6'],
                          input=cv2.imencode('.png',img)[1].tobytes(),capture_output=True,check=True)
    return {'doc_id':doc_id,'page':number,'text':result.stdout.decode().strip(),'method':'ocr_clean'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--dpi',type=int,default=300)
    args=ap.parse_args()
    os.environ.setdefault('OMP_THREAD_LIMIT','1')
    mac_data=Path('/opt/homebrew/share/tessdata')
    if mac_data.exists():
        os.environ.setdefault('TESSDATA_PREFIX',str(mac_data))
    path=ROOT/'artifacts/pages.jsonl'
    pages=read_jsonl(path)
    docs={d['doc_id']:ROOT/'data/reports'/d['path'] for d in read_jsonl(ROOT/'data/documents.jsonl')}
    cache=ROOT/('artifacts/ocr_clean_v3_%d.jsonl' % args.dpi)
    records=read_jsonl(cache) if cache.exists() else []
    done={(p['doc_id'],p['page']):p for p in records}
    tasks=[(str(docs[p['doc_id']]),p['doc_id'],p['page'],args.dpi) for p in pages
           if (len(p['text'])<60 or p['method']=='ocr') and (p['doc_id'],p['page']) not in done]
    print('OCR tasks:',len(tasks),'cached:',len(done),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(recognize,task) for task in tasks]
        for n,future in enumerate(as_completed(futures),1):
            row=future.result()
            done[row['doc_id'],row['page']]=row
            with cache.open('a') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if n%10==0 or n==len(tasks):
                print('OCR %d/%d' % (n,len(tasks)),flush=True)
    for page in pages:
        replacement=done.get((page['doc_id'],page['page']))
        if replacement:
            page.update(replacement)
    temporary=path.with_suffix('.tmp')
    temporary.write_text('\n'.join(json.dumps(p,ensure_ascii=False) for p in pages)+'\n')
    temporary.replace(path)
    (ROOT/'artifacts/extraction_issues.json').write_text('[]\n')
    print('OCR complete; remaining pages under 60 chars:',sum(len(p['text'])<60 for p in pages),flush=True)


if __name__=='__main__':
    main()

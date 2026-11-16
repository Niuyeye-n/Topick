"""Reconstruct historical frozen inputs from public official datasets and ID lists.
Run on Linux in the GPU environment. This command does not train or load a Reader.
"""
from pathlib import Path
import argparse,json,shutil,sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'code'))
from eval_common import write,file_sha,validate_rows,content_key

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();out=a.run_dir.resolve()
    if out.exists() and any(out.iterdir()):p.error('Use an empty new run directory; existing results are never replaced')
    out.mkdir(parents=True,exist_ok=True)
    shutil.copytree(REPO/'code',out/'code',ignore=shutil.ignore_patterns('__pycache__'))
    ids=json.loads((REPO/'configs/frozen_ids.json').read_text(encoding='utf-8'))
    protocol=json.loads((REPO/'configs/protocol.json').read_text(encoding='utf-8'))
    from datasets import load_dataset
    def normalize(task,row,labeled=True):
        labels=row['choices']['label'];options=row['choices']['text']
        if len(options)!=4 or row['answerKey'] not in labels:return None
        r={'id':str(row['id']),'question':row['question'] if task=='arc_c' else row['question_stem'],'options':options}
        if labeled:r['gold']=labels.index(row['answerKey'])
        if task=='openbookqa':r['fact']=row['fact1']
        return r
    for task,repo,config in [('arc_c','allenai/ai2_arc','ARC-Challenge'),('openbookqa','allenai/openbookqa','additional')]:
        ds=load_dataset(repo,config)
        (out/'staged_assets').mkdir(exist_ok=True)
        ds['train'].to_parquet(str(out/'staged_assets'/(task+'_official_train.parquet')))
        values={}
        for role,split in [('pool','train'),('dev','validation'),('queries','test')]:
            index={str(r['id']):r for r in ds[split]}
            rows=[normalize(task,index[i],role!='queries') for i in ids[task][role]]
            validate_rows(rows,labeled=role!='queries',task=task)
            values[role]=rows
            write(out/f'{task}_{role}.json',rows)
            expected=protocol['tasks'][task]['files'][role]['sha256']
            if file_sha(out/f'{task}_{role}.json')!=expected:
                raise ValueError(f'{task}/{role}: official source/serialization differs from historical frozen input; do not silently replace hashes')
        gold=[{'id':i,'gold':index[i]['choices']['label'].index(index[i]['answerKey'])} for i in ids[task]['queries']]
        write(out/f'{task}_gold.json',gold)
        if file_sha(out/f'{task}_gold.json')!=protocol['tasks'][task]['files']['gold']['sha256']:raise ValueError('Gold hash mismatch')
        for a,b in [('pool','dev'),('pool','queries'),('dev','queries')]:
            if {content_key(r) for r in values[a]} & {content_key(r) for r in values[b]}:raise ValueError('Content leakage')
    write(out/'protocol.json',protocol)
    shutil.copy2(REPO/'configs/reader_reference.json',out/'reader_reference.json')
    print(json.dumps({'status':'inputs_ready','run_dir':str(out),'gpu_started':False}))

if __name__=='__main__':main()

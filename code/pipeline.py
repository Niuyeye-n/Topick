import datetime,json,pathlib,platform,subprocess,sys,time,traceback,signal
from eval_common import write,file_sha
from runtime import ROOT,ASSETS,event,stage,status,now,bj,seed_all,read,code_manifest

def wait_ready():
    deadline=time.monotonic()+2400
    while not (ROOT/'prepare.complete.json').exists() or not (ROOT.parent/'reader_recovery.complete.json').exists():
        if (ROOT.parent/'reader_recovery.failed.json').exists():raise RuntimeError('Exact reader recovery failed; inspect preserved evidence')
        state=read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
        if state.get('status')=='failed':raise RuntimeError('Preparation failed; inspect preserved evidence')
        if time.monotonic()>deadline:raise TimeoutError('Preparation/recovery wait exceeded 40 minutes')
        time.sleep(5)

def main():
    def expired(*args):raise TimeoutError('Approved six-hour pipeline wall-clock budget reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(6*3600)
    write(ROOT/'execution_manifest.json',{'status':'running','gpu_work_started':True,'started_beijing':bj(),'authorization':'user accepted local annotation + max300 training/task; no paid API','code':code_manifest()})
    event('pipeline_started',python=sys.version,platform=platform.platform(),code=code_manifest(),paid_api_allowed=False)
    seed_all()
    if not (ROOT/'prepare.complete.json').exists():
        from prepare import prepare_data,download_assets,encode_initial
        prepare_data();download_assets();encode_initial()
        write(ROOT/'prepare.complete.json',{'status':'complete'})
    if not all((ROOT/'training'/t/'soft_targets.npy').exists() for t in ['arc_c','openbookqa']):
        from annotate_train import annotate,build_targets_and_embeddings
        annotate();build_targets_and_embeddings()
    from train_repair import train_repaired
    from evaluate import evaluate_all
    try:
        train_repaired();evaluate_all()
        tags=['qwen_arc','qwen_obqa','llama_arc','llama_obqa'];rows=[]
        for tag in tags:
            path=ROOT/'table_runs'/tag/'result.json';marker=read(path.with_suffix('.complete.json'))
            if marker['result_sha256']!=file_sha(path):raise ValueError('Result seal mismatch')
            result=read(path);rows.append({'tag':tag,'task':result['task'],'reader':tag.split('_')[0],**result['methods']['topick']['metrics']})
        write(ROOT/'summary.json',{'status':'complete','method':'topick','rows':rows,'n_total':2000,'completed_beijing':bj(),'result_files':[{'path':str(ROOT/'table_runs'/t/'result.json'),'sha256':file_sha(ROOT/'table_runs'/t/'result.json')} for t in tags]})
        write(ROOT/'status.json',{'status':'complete','phase':'all_four_complete','updated_beijing':bj(),'groups':4,'predictions':2000})
        event('pipeline_complete',summary=rows)
    except BaseException as e:
        write(ROOT/'status.json',{'status':'failed','updated_beijing':bj(),'error':f'{type(e).__name__}: {e}'})
        event('pipeline_failed',error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':main()

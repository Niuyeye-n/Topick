import contextlib,datetime,json,os,pathlib,time,hashlib,random
import numpy as np
from eval_common import write,file_sha,digest

ROOT=pathlib.Path(os.environ.get('TOPICK_RUN_DIR', pathlib.Path(__file__).resolve().parents[1])).resolve()
ASSETS=pathlib.Path(os.environ.get('TOPICK_ASSETS_DIR', ROOT.parent/'assets')).resolve()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def bj():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def event(kind,**fields):
    row={'at_utc':now(),'at_beijing':bj(),'event':kind,**fields}
    with (ROOT/'timeline.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)
def status(phase,**fields):write(ROOT/'status.json',{'status':'running','phase':phase,'updated_beijing':bj(),**fields})
@contextlib.contextmanager
def stage(name):
    status(name);start=time.monotonic();event('stage_started',stage=name)
    try:yield
    except BaseException as e:
        event('stage_failed',stage=name,seconds=time.monotonic()-start,error=f'{type(e).__name__}: {e}')
        write(ROOT/'status.json',{'status':'failed','phase':name,'updated_beijing':bj(),'error':f'{type(e).__name__}: {e}'})
        raise
    else:event('stage_completed',stage=name,seconds=time.monotonic()-start)
def seed_all(seed=42):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def code_manifest():return [{'path':str(p),'sha256':file_sha(p)} for p in sorted((ROOT/'code').glob('*.py'))]
def protocol():return read(ROOT/'protocol.json')
def data(task,role):
    if role=='gold':raise PermissionError('Use explicit scoring-side gold loader after selection')
    spec=protocol()['tasks'][task]['files'][role];p=ROOT/spec['path']
    if file_sha(p)!=spec['sha256']:raise ValueError('Frozen input changed')
    return read(p)

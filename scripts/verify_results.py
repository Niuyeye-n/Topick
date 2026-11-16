"""CPU-only verification of an extracted result ZIP (Python standard library)."""
from pathlib import Path
import argparse,hashlib,json

def verify(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
    for r in manifest['files']:
        p=root/r['path']
        if not p.resolve().is_relative_to(root):raise ValueError('Path outside package')
        if p.stat().st_size!=r['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Changed file: '+r['path'])
    checks=[]
    for p in sorted((root/'runs').glob('run*/table_runs/*/result.json')):
        d=json.loads(p.read_text(encoding='utf-8'));rows=d['methods']['topick']['records']
        ids=[r['id'] for r in rows]
        expected=json.loads((p.parent/'expected_ids.json').read_text(encoding='utf-8'))
        assert ids==expected and len(set(ids))==500 and d['status']=='complete'
        assert all(r['pred'] in range(4) and r['gold'] in range(4) and len(set(r['demo_ids']))==len(r['demo_ids'])==4 for r in rows)
        cm=[[0]*4 for _ in range(4)]
        for r in rows:cm[r['gold']][r['pred']]+=1
        supports=[sum(row) for row in cm]
        f=[]
        for c in range(4):
            den=supports[c]+sum(row[c] for row in cm)
            f.append(2*cm[c][c]/den if den else 0.)
        m={'accuracy':100*sum(cm[c][c] for c in range(4))/500,'f1_macro':100*sum(f)/4,'f1_weighted':100*sum(s*f1 for s,f1 in zip(supports,f))/500}
        for k,v in m.items():assert abs(v-d['methods']['topick']['metrics'][k])<1e-9
        checks.append({'run':p.parents[2].name,'tag':p.parent.name,'n':500,**m})
    assert len(checks)==8
    return {'status':'verified','files':len(manifest['files']),'groups':checks}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('extracted_result_directory',type=Path);a=p.parse_args()
    print(json.dumps(verify(a.extracted_result_directory),ensure_ascii=False,indent=2))

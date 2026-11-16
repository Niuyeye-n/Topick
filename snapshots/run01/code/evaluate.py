"""Reader-specific training knowledge, dev-only selection, then sealed test evaluation."""
import json,pathlib,time
import numpy as np
import torch
from eval_common import digest,file_sha,metrics,write
from runtime import ROOT,data,event,stage,status,read,protocol,code_manifest,now
from topic_core import TopicPredictor,select
from reader import FrozenReader

TASKS=['arc_c','openbookqa']
def predictor(task):
    folder=ROOT/'training'/task;best=read(folder/'best_model.json')
    model=TopicPredictor(np.load(folder/'topic_embeddings.npy')).to('cuda')
    model.load_state_dict(torch.load(best['selected']['path'],map_location='cuda',weights_only=True));model.eval()
    return model

def knowledge(lm,task,model):
    folder=ROOT/'training'/task;rows=read(folder/'rows.json');emb=np.load(folder/'training_emb.npy')
    out=ROOT/'knowledge'/lm.name;out.mkdir(parents=True,exist_ok=True);path=out/(task+'_zero_shot.json')
    records=read(path) if path.exists() else []
    for r in rows[len(records):]:
        pred=lm.answer({k:v for k,v in r.items() if k!='gold'},[],{},save_prompt=(not records));pred['gold']=r['gold'];records.append(pred)
        if len(records)%25==0 or len(records)==len(rows):write(path,records);event('knowledge_progress',reader=lm.name,task=task,n=len(records),total=len(rows))
    if [r['id'] for r in records]!=[r['id'] for r in rows]:raise ValueError('Resumed knowledge row order changed')
    with torch.inference_mode():p=torch.sigmoid(model(torch.tensor(emb,device='cuda'))).float().cpu().numpy()
    correct=np.asarray([r['pred']==r['gold'] for r in records],dtype=np.float64)
    denominator=p.sum(0,dtype=np.float64);know=(p.astype(np.float64)*correct[:,None]).sum(0)/np.maximum(denominator,1e-12)
    if not np.isfinite(know).all():raise FloatingPointError('Invalid model knowledge')
    np.save(out/(task+'_topic_knowledge.npy'),know)
    write(out/(task+'_manifest.json'),{'definition':'paper Eq5: topic-weighted empirical zero-shot correctness','records_sha256':file_sha(path),'reader_model_files':lm.manifest['model_files'],'n':len(rows),'zero_shot_metrics':metrics(records),'predictor_checkpoint_sha256':read(folder/'best_model.json')['best_checkpoint_sha256'],'test_labels_used':False})
    return know

def choose_rows(task,model,knowledge,role,weight):
    folder=ROOT/'training'/task;pool=data(task,'pool');queries=data(task,role)
    p=np.load(folder/'pool_emb.npy');q=np.load(folder/(role+'_emb.npy'))
    rows=[]
    for row,vector in zip(queries,q):
        chosen,trace=select(model,vector,p,knowledge,weight_rel=weight)
        rows.append({'query_id':row['id'],'demo_ids':[pool[i]['id'] for i in chosen],'trace':trace})
    if len(rows)!=len(queries) or len({r['query_id'] for r in rows})!=len(queries):raise ValueError('Incomplete selection')
    return rows

def tune_and_test(lm,task,model,knowledge):
    tag=lm.name+'_'+('arc' if task=='arc_c' else 'obqa');out=ROOT/'table_runs'/tag;out.mkdir(parents=True,exist_ok=True)
    pool={r['id']:r for r in data(task,'pool')};dev=data(task,'dev');tuning=[]
    with stage(tag+'_dev_smoke'):
        chosen=choose_rows(task,model,knowledge,'dev',0.1)
        smoke=[]
        for q,s in zip(dev[:3],chosen[:3]):
            r=lm.answer(q,s['demo_ids'],pool,save_prompt=True);r['gold']=q['gold'];smoke.append(r)
        write(out/'dev_smoke.json',{'reader_manifest':lm.manifest,'records':smoke,'not_official_test':True})
    with stage(tag+'_dev_tuning'):
        for weight in [0.1,0.5,1.0]:
            rows=choose_rows(task,model,knowledge,'dev',weight);records=[]
            for q,s in zip(dev,rows):
                r=lm.answer(q,s['demo_ids'],pool);r['gold']=q['gold'];records.append(r)
            measure=metrics(records);tuning.append({'lambda':weight,'metrics':measure})
            write(out/f'dev_lambda_{weight:g}.json',{'lambda':weight,'records':records,'metrics':measure,'selection':rows})
            event('dev_lambda_finished',reader=lm.name,task=task,lambda_value=weight,accuracy=measure['accuracy'])
        best=min(tuning,key=lambda r:(-r['metrics']['accuracy'],r['lambda']))
        write(out/'locked_config.json',{'selected_lambda':best['lambda'],'selection_rule':'highest dev50 accuracy, ties smaller lambda','trials':tuning,'test_results_consulted':False,'locked_utc':now()})
    with stage(tag+'_test_selection'):
        rows=choose_rows(task,model,knowledge,'queries',best['lambda']);queries=data(task,'queries')
        selector_manifest={'test_labels_used':False,'tuning_splits':['train','dev'],'implementation_files':code_manifest(),'training_rows_sha256':file_sha(ROOT/'training'/task/'rows.json'),'topic_predictor_sha256':read(ROOT/'training'/task/'best_model.json')['best_checkpoint_sha256'],'topic_knowledge_sha256':file_sha(ROOT/'knowledge'/lm.name/(task+'_topic_knowledge.npy')),'lambda':best['lambda'],'implementation':'pinned_public_TopicK_rank_and_pooling_with_approved_local_core_topic_supervision','upstream_commit':'0d5d0ba727d7cd072f8dc328ceaaee9ee949d440','adaptations':['predefined_CMS_topic_set','local_Llama_core_topics','max_300_train_questions','frozen_chat_letter_reader','50_candidates_4_shot']}
        selection={'method':'topick','protocol_sha256':file_sha(ROOT/'protocol.json'),'task_name':task,'pool_sha256':digest(list(pool.values())),'queries_sha256':digest(queries),'selector_manifest':selector_manifest,'rows':rows}
        write(out/'selection.json',selection)
        if len(rows)!=500 or any(len(r['demo_ids'])!=4 or len(set(r['demo_ids']))!=4 or not set(r['demo_ids'])<=pool.keys() for r in rows):raise ValueError('Invalid test selections')
    # Gold is opened ONLY after the entire selector output is serialized and validated.
    gold_spec=protocol()['tasks'][task]['files']['gold'];gold_path=ROOT/gold_spec['path']
    if file_sha(gold_path)!=gold_spec['sha256']:raise ValueError('Changed scorer gold file')
    gold={r['id']:r['gold'] for r in read(gold_path)}
    path=out/'result.json'
    if path.exists():raise FileExistsError('Never overwrite an existing formal test run')
    state={'schema_version':1,'status':'running','task':task,'protocol_sha256':file_sha(ROOT/'protocol.json'),'protocol':protocol(),'started_utc':now(),'execution_scope':'single_baseline','selection_sources':{'topick':{'path':str(out/'selection.json'),'sha256':file_sha(out/'selection.json'),'selector_manifest':selector_manifest}},'manifest':{**lm.manifest,'evaluator_sha256':file_sha(__file__),'common_sha256':file_sha(ROOT/'code/eval_common.py')},'methods':{'topick':{'records':[]}}}
    with stage(tag+'_formal_test'):
        write(path,state)
        try:
            for i,(q,s) in enumerate(zip(queries,rows)):
                r=lm.answer(q,s['demo_ids'],pool,save_prompt=(i==0));r['gold']=gold[q['id']];state['methods']['topick']['records'].append(r)
                if (i+1)%25==0 or i==499:write(path,state);event('formal_test_progress',reader=lm.name,task=task,n=i+1,total=500);status(tag+'_formal_test',n=i+1,total=500)
            records=state['methods']['topick']['records']
            if len(records)!=500 or {r['id'] for r in records}!=set(gold):raise ValueError('Final test membership mismatch')
            result=metrics(records);state['methods']['topick']['metrics']=result
            state['status']='complete';state['completed_utc']=now();state['manifest']['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated();state['manifest']['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved()
            write(path,state);write(out/'result.complete.json',{'status':'complete','n_per_method':500,'result_sha256':file_sha(path),'protocol_sha256':file_sha(ROOT/'protocol.json')})
            event('group_complete',reader=lm.name,task=task,metrics=result)
        except BaseException as e:
            state['status']='failed';state['error']=f'{type(e).__name__}: {e}';write(path,state);raise

def evaluate_all():
    for reader in ['qwen','llama']:
        with stage(reader+'_reader_load'):lm=FrozenReader(reader)
        for task in TASKS:
            model=predictor(task)
            with stage(reader+'_'+task+'_knowledge'):know=knowledge(lm,task,model)
            tune_and_test(lm,task,model,know)
            del model;torch.cuda.empty_cache()
        peak=lm.close();write(ROOT/(reader+'_peak_gpu.json'),peak)

if __name__=='__main__':evaluate_all()

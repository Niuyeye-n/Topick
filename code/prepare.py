import collections,hashlib,io,json,pathlib,pickle,re,time,urllib.request
import numpy as np
from eval_common import content_key,question_block,write,file_sha,digest,validate_rows
from runtime import ROOT,ASSETS,data,stage,event,read

class SafeListUnpickler(pickle.Unpickler):
    def find_class(self,module,name):raise pickle.UnpicklingError('No executable pickle globals permitted')

def normalize(task,row):
    choices=row['choices'];labels=choices['label'];opts=choices['text']
    if len(opts)!=4 or row['answerKey'] not in labels:return None
    if task=='openbookqa':
        q=row.get('question_stem',row.get('question',''));fact=row.get('fact1','')
        if not fact.strip():raise ValueError('Training source lacks required fact1')
    else:q=row['question'];fact=''
    result={'id':str(row['id']),'question':q,'options':opts,'gold':labels.index(row['answerKey'])}
    if fact:result['fact']=fact
    return result

def cached_training(task):
    from datasets import load_dataset,Dataset
    repo,config=('allenai/ai2_arc','ARC-Challenge') if task=='arc_c' else ('allenai/openbookqa','additional')
    staged=ROOT/'staged_assets'/(task+'_official_train.parquet')
    if staged.is_file():
        dataset=load_dataset('parquet',data_files={'train':[str(staged)]},split='train')
        return dataset,{'repo':repo,'config':config,'split':'train','files':[{'path':str(staged),'sha256':file_sha(staged)}],'transfer':'official Hugging Face file fetched locally, transferred by SSH because remote Hub is unreachable'}
    base=pathlib.Path('/root/.cache/huggingface/hub')/('datasets--'+repo.replace('/','--'))
    files=sorted(base.glob(f'snapshots/*/{config}/train*.parquet'))
    if files:
        dataset=load_dataset('parquet',data_files={'train':[str(p) for p in files]},split='train')
        return dataset,{'repo':repo,'config':config,'split':'train','files':[{'path':str(p),'sha256':file_sha(p)} for p in files]}
    raise FileNotFoundError('Stage official train parquet; do not repeat unreachable Hub requests')

def prepare_data():
    with stage('prepare_training_data'):
        for task in ['arc_c','openbookqa']:
            dest=ROOT/'training'/task;dest.mkdir(parents=True,exist_ok=True)
            pool=data(task,'pool');dev=data(task,'dev');queries=data(task,'queries')
            prohibited={content_key(r) for r in dev+queries}
            ds,source=cached_training(task)
            all_rows={};rejected=collections.Counter()
            for row in ds:
                r=normalize(task,row)
                if r is None:rejected['non_four_option_or_bad_label']+=1;continue
                key=content_key(r)
                if key in prohibited:rejected['overlap_with_dev_or_test_content']+=1;continue
                if key in all_rows:rejected['duplicate_content']+=1;continue
                all_rows[key]=r
            for r in pool:
                key=content_key(r)
                if key not in all_rows:raise ValueError('Frozen pool missing from current official train source')
                if all_rows[key]['gold']!=r['gold']:raise ValueError('Train source label differs from frozen candidate')
            chosen=list(pool);seen={content_key(r) for r in chosen}
            ordered=sorted(all_rows.items(),key=lambda kv:digest([42,task,'topick_topic_train',kv[1]['id']]))
            for key,r in ordered:
                if len(chosen)>=300:break
                if key not in seen:chosen.append(r);seen.add(key)
            validate_rows(chosen,labeled=True,task=task)
            order=sorted(range(len(chosen)),key=lambda i:digest([42,task,'topic_internal_split',chosen[i]['id']]))
            val=set(order[:max(1,round(len(chosen)*0.1))]);roles=['validation' if i in val else 'train' for i in range(len(chosen))]
            write(dest/'rows.json',chosen);write(dest/'split.json',roles)
            write(dest/'manifest.json',{'official_source':source,'original_rows':len(ds),'eligible_unique':len(all_rows),'rejections':dict(rejected),'used':len(chosen),'training':roles.count('train'),'validation':roles.count('validation'),'pool_included':len(pool),'rows_sha256':file_sha(dest/'rows.json'),'test_gold_read':False})
            event('training_data_prepared',task=task,used=len(chosen),train=roles.count('train'),validation=roles.count('validation'))

def download_assets():
    from huggingface_hub import snapshot_download
    with stage('download_topic_assets'):
        ASSETS.mkdir(parents=True,exist_ok=True)
        p=ASSETS/'official_cms_topic_list.pickle'
        if not p.exists():
            url='https://raw.githubusercontent.com/WonbinKweon/TopicK_EMNLP2025/0d5d0ba727d7cd072f8dc328ceaaee9ee949d440/data/cms/topic_list'
            with urllib.request.urlopen(url,timeout=60) as r:p.write_bytes(r.read())
        names=SafeListUnpickler(io.BytesIO(p.read_bytes())).load()
        if not isinstance(names,list) or len(names)!=3781 or not all(isinstance(x,str) for x in names):raise ValueError('Unexpected topic artifact')
        write(ASSETS/'initial_topics.json',names)
        snapshot_download('sentence-transformers/all-mpnet-base-v2',revision='e8c3b32edf5434bc2275fc9bab85f82640a19130',local_dir=str(ASSETS/'all-mpnet-base-v2'),allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','modules.json','sentence_bert_config.json','1_Pooling/config.json','2_Normalize/config.json'])
        if file_sha(ASSETS/'all-mpnet-base-v2/model.safetensors')!='78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658':raise ValueError('Encoder artifact differs from historical weights')
        write(ASSETS/'topic_asset_manifest.json',{'initial_topics_sha256':file_sha(p),'upstream_commit':'0d5d0ba727d7cd072f8dc328ceaaee9ee949d440','portability_note':'HF download; verify encoder against archived model fingerprint for historical comparison'})

def encode_texts(texts,model,tokenizer,batch=32):
    import torch
    from torch.nn import functional as F
    results=[]
    with torch.inference_mode():
        for i in range(0,len(texts),batch):
            tokens=tokenizer(texts[i:i+batch],padding=True,truncation=True,max_length=384,return_tensors='pt').to('cuda')
            output=model(**tokens).last_hidden_state;mask=tokens['attention_mask'].unsqueeze(-1).to(output.dtype)
            mean=(output*mask).sum(1)/mask.sum(1).clamp_min(1e-9)
            results.append(F.normalize(mean,p=2,dim=1).float().cpu().numpy())
    return np.concatenate(results)

def encode_initial():
    import torch
    from transformers import AutoTokenizer,AutoModel
    with stage('encode_initial_texts'):
        model=AutoModel.from_pretrained(ASSETS/'all-mpnet-base-v2',local_files_only=True).to('cuda').eval()
        tokenizer=AutoTokenizer.from_pretrained(ASSETS/'all-mpnet-base-v2',local_files_only=True)
        names=read(ASSETS/'initial_topics.json');np.save(ASSETS/'initial_topic_embeddings.npy',encode_texts(names,model,tokenizer))
        for task in ['arc_c','openbookqa']:
            folder=ROOT/'training'/task
            for role,rows in [('training',read(folder/'rows.json')),('pool',data(task,'pool')),('dev',data(task,'dev')),('queries',data(task,'queries'))]:
                np.save(folder/(role+'_emb.npy'),encode_texts([question_block(r) for r in rows],model,tokenizer))
            event('embedding_prepared',task=task,embedding='mean_pool_normalize',max_seq_length=384)
        del model;torch.cuda.empty_cache()

if __name__=='__main__':
    prepare_data();download_assets();encode_initial()
    write(ROOT/'prepare.complete.json',{'status':'complete','training_rows':{t:len(read(ROOT/'training'/t/'rows.json')) for t in ['arc_c','openbookqa']}})

import collections,json,math,pathlib,re,time
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from eval_common import question_block,write,file_sha,digest
from runtime import ROOT,ASSETS,event,stage,status,data,read,seed_all
from reader import FrozenReader
from prepare import encode_texts
from topic_core import TopicPredictor,paper_loss

def words(text):return re.findall(r'[a-z0-9]+',text.lower())
def clean_topic(t):return re.sub(r'\s+',' ',t.strip().strip('"\'`[]')).casefold()

def annotate():
    with stage('annotate_core_topics_local_llama'):
        lm=FrozenReader('llama');names=read(ASSETS/'initial_topics.json');te=np.load(ASSETS/'initial_topic_embeddings.npy')
        for task in ['arc_c','openbookqa']:
            pool={r['id']:r for r in data(task,'pool')};demo=list(pool)[:4];records=[]
            for r in data(task,'dev')[:3]:
                item=lm.answer(r,demo,pool,save_prompt=True);item['gold']=r['gold'];records.append(item)
            write(ROOT/'training'/task/'pretraining_dev_smoke.json',{'purpose':'frozen reader / rendering diagnostic; first four pool members, NOT TopicK selections','records':records,'manifest':lm.manifest})
        bm=BM25Okapi([words(n) for n in names]);name_index={clean_topic(n):i for i,n in enumerate(names)}
        for task in ['arc_c','openbookqa']:
            folder=ROOT/'training'/task;rows=read(folder/'rows.json');emb=np.load(folder/'training_emb.npy')
            path=folder/'annotations.jsonl';done={}
            if path.exists():done={r['id']:r for r in [json.loads(x) for x in path.read_text().splitlines() if x.strip()]}
            for i,r in enumerate(rows):
                if r['id'] in done:continue
                scores=bm.get_scores(words(question_block(r)));lex=np.argsort(-scores,kind='stable')[:10].tolist()
                sem=[j for j in np.argsort(-(emb[i]@te.T),kind='stable').tolist() if j not in lex][:10]
                candidates=[names[j] for j in lex+sem]
                demo=question_block(r)+'\nAnswer: '+'ABCD'[r['gold']]+'. '+r['options'][r['gold']]
                prompt=('You will receive a question-answer demonstration along with a candidate topic set. Your task is to output relevant topics of the demonstration. You may choose topics from the candidate topic set, or you can create new relevant topics. You must provide at least five topics. Do not include any explanation or numbers. Please just output the list of relevant topics, separated by commas. Demonstration: '+demo+', Candidate topic set: '+', '.join(candidates))
                attempts=[]
                for attempt in range(2):
                    msgs=[{'role':'user','content':prompt if attempt==0 else prompt+'\nReturn between five and ten concise English topics as one comma-separated line.'}]
                    text=lm.tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
                    tokens=lm.tokenizer(text,return_tensors='pt',add_special_tokens=False,truncation=False).to('cuda')
                    if tokens.input_ids.shape[1]>3072:raise ValueError('Annotation input too long')
                    with torch.inference_mode():
                        output=lm.model.generate(**tokens,do_sample=False,num_beams=1,max_new_tokens=128 if attempt==0 else 192,repetition_penalty=1.,pad_token_id=lm.tokenizer.pad_token_id,eos_token_id=lm.tokenizer.eos_token_id,return_dict_in_generate=True,output_logits=True)
                    if any(not torch.isfinite(step).all() for step in output.logits):raise FloatingPointError('Non-finite annotation logits')
                    raw=lm.tokenizer.decode(output.sequences[0,tokens.input_ids.shape[1]:],skip_special_tokens=True).strip()
                    topics=[]
                    for t in re.split(r'[,;\n]',raw):
                        t=clean_topic(re.sub(r'^\s*(?:\d+[.)]|[-*])\s*','',t))
                        if 2<=len(t)<=80 and len(words(t))<=10 and ':' not in t and t not in topics:topics.append(t)
                    attempts.append({'messages':msgs,'raw_output':raw,'topics':topics,'max_new_tokens':128 if attempt==0 else 192,'input_tokens':int(tokens.input_ids.shape[1])})
                    if len(topics)>=5:break
                if len(topics)<5:
                    write(folder/f'annotation_failed_{i}.json',{'id':r['id'],'attempts':attempts});raise ValueError('Core-topic annotation failed twice')
                item={'id':r['id'],'topics':topics,'candidates':candidates,'attempts':attempts,'source':'local_Llama-3-8B-Instruct_substitution_for_GPT-4o','reader_manifest_sha256':digest(lm.manifest),'test_labels_used':False}
                with path.open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
                if (i+1)%10==0 or i+1==len(rows):event('annotation_progress',task=task,done=i+1,total=len(rows));status('annotate_core_topics_local_llama',task=task,done=i+1,total=len(rows))
            write(folder/'annotator_manifest.json',lm.manifest)
        lm.close()

def build_targets_and_embeddings():
    from transformers import AutoModel,AutoTokenizer
    with stage('build_soft_labels'):
        encoder=AutoModel.from_pretrained(ASSETS/'all-mpnet-base-v2',local_files_only=True).to('cuda').eval()
        tokenizer=AutoTokenizer.from_pretrained(ASSETS/'all-mpnet-base-v2',local_files_only=True)
        for task in ['arc_c','openbookqa']:
            folder=ROOT/'training'/task;rows=read(folder/'rows.json')
            ann={r['id']:r for r in [json.loads(x) for x in (folder/'annotations.jsonl').read_text().splitlines()]}
            names=list(read(ASSETS/'initial_topics.json'));index={clean_topic(t):i for i,t in enumerate(names)}
            for row in rows:
                for t in ann[row['id']]['topics']:
                    if t not in index:index[t]=len(names);names.append(t)
            topics=encode_texts(names,encoder,tokenizer)
            np.save(folder/'topic_embeddings.npy',topics);write(folder/'topics.json',names)
            embeds=np.load(folder/'training_emb.npy');sim=embeds@embeds.T
            np.fill_diagonal(sim,-np.inf);near=np.argsort(-sim,axis=1,kind='stable')[:,:min(100,len(rows)-1)]
            docs=[words(question_block(r)+' '+r['options'][r['gold']]) for r in rows];bm=BM25Okapi(docs)
            target=np.zeros((len(rows),len(names)),dtype=np.float32)
            scorecache={}
            for i,r in enumerate(rows):
                ids=[index[t] for t in ann[r['id']]['topics']];dist=[]
                for tid in ids:
                    if tid not in scorecache:scorecache[tid]=np.asarray(bm.get_scores(words(names[tid])),dtype=np.float64)
                    s=scorecache[tid];den=np.logaddexp.reduce(np.concatenate([[0.],s[near[i]]]))
                    dist.append(s[i]-den)
                weights=np.exp(np.asarray(dist)-max(dist));target[i,ids]=weights
            np.save(folder/'soft_targets.npy',target)
            write(folder/'soft_target_manifest.json',{'topics':len(names),'seed_topics':3781,'neighbor_count':near.shape[1],'label_definition':'literal_paper_eq2_eq3','training_rows':len(rows),'topic_source':'predefined_official_CMS_plus_local_core_annotation_new_topics','annotation_sha256':file_sha(folder/'annotations.jsonl')})
            event('soft_targets_ready',task=task,topics=len(names),n=len(rows))
        del encoder;torch.cuda.empty_cache()

def train_predictors():
    with stage('train_topic_predictors'):
        torch.set_num_threads(8)
        for task in ['arc_c','openbookqa']:
            folder=ROOT/'training'/task;roles=read(folder/'split.json')
            x=torch.tensor(np.load(folder/'training_emb.npy'),device='cuda');y=torch.tensor(np.load(folder/'soft_targets.npy'),device='cuda');topics=np.load(folder/'topic_embeddings.npy')
            tr=torch.tensor([i for i,r in enumerate(roles) if r=='train'],device='cuda');va=torch.tensor([i for i,r in enumerate(roles) if r=='validation'],device='cuda')
            best_candidates=[]
            for lr in [1e-5,1e-4]:
                seed_all();model=TopicPredictor(topics).to('cuda');opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=lr)
                out=folder/f'lr_{lr:g}';out.mkdir(exist_ok=True);history=[];best=float('inf');updates=0;selected=None
                generator=torch.Generator().manual_seed(42)
                for epoch in range(1,31):
                    model.train();perm=tr[torch.randperm(len(tr),generator=generator).to('cuda')];losses=[]
                    for j in range(0,len(perm),64):
                        ids=perm[j:j+64];opt.zero_grad(set_to_none=True);loss=paper_loss(model(x[ids]),y[ids])
                        if not torch.isfinite(loss):raise FloatingPointError('Invalid topic training loss')
                        loss.backward();opt.step();updates+=1;losses.append((float(loss.detach()),len(ids)))
                    model.eval()
                    with torch.inference_mode():valid=float(paper_loss(model(x[va]),y[va]))
                    row={'epoch':epoch,'optimizer_updates':updates,'train_loss':sum(a*b for a,b in losses)/sum(b for a,b in losses),'validation_topic_loss':valid,'learning_rate':lr}
                    history.append(row)
                    if valid<best:
                        best=valid;selected=epoch;torch.save(model.state_dict(),out/'best.pt')
                    event('topic_epoch',task=task,**row)
                write(out/'history.json',history);best_candidates.append({'lr':lr,'epoch':selected,'validation_topic_loss':best,'path':str(out/'best.pt'),'updates_per_epoch':math.ceil(len(tr)/64),'total_updates':updates})
                del model;torch.cuda.empty_cache()
            selected=min(best_candidates,key=lambda r:(r['validation_topic_loss'],r['epoch'],r['lr']))
            write(folder/'best_model.json',{'selected':selected,'candidates':best_candidates,'test_metric_used':False,'selection':'minimum independent internal-validation topic loss','seed':42,'epochs':30,'batch_size':64,'optimizer':'AdamW','weight_decay':0.01,'warmup_steps':0,'loss':'literal paper Eq4, mean over batch/topics','best_checkpoint_sha256':file_sha(selected['path'])})

if __name__=='__main__':annotate();build_targets_and_embeddings();train_predictors()

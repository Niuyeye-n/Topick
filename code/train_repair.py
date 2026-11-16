"""One approved convergence repair. Same data/objective/architecture, larger cap.
The pre-test diagnostic gate is fixed here before this run starts.
"""
import math,pathlib,time
import numpy as np
import torch
from eval_common import write,file_sha
from runtime import ROOT,event,stage,read,seed_all
from topic_core import TopicPredictor,paper_loss

def recalls(model,x,y,ids):
    with torch.inference_mode():
        order=model(x[ids]).topk(20,dim=1).indices
        positive=y[ids]>0;den=positive.sum(1).clamp_min(1)
        return {f'recall_at_{k}':float((positive.gather(1,order[:,:k]).sum(1)/den).float().mean()) for k in [5,10,20]}

def train_repaired():
    with stage('train_topic_predictors_repair'):
        torch.set_num_threads(8);all_selected={}
        for task in ['arc_c','openbookqa']:
            folder=ROOT/'training'/task;roles=read(folder/'split.json')
            x=torch.tensor(np.load(folder/'training_emb.npy'),device='cuda');y=torch.tensor(np.load(folder/'soft_targets.npy'),device='cuda');topics=np.load(folder/'topic_embeddings.npy')
            tr=torch.tensor([i for i,r in enumerate(roles) if r=='train'],device='cuda');va=torch.tensor([i for i,r in enumerate(roles) if r=='validation'],device='cuda')
            candidates=[]
            for lr in [1e-5,1e-4]:
                seed_all();model=TopicPredictor(topics).to('cuda');opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=lr)
                out=folder/f'lr_{lr:g}';out.mkdir(exist_ok=False);history=[];best=float('inf');updates=0;selected=None;last_improved=0;generator=torch.Generator().manual_seed(42)
                for epoch in range(1,1001):
                    model.train();perm=tr[torch.randperm(len(tr),generator=generator).to('cuda')];losses=[]
                    for j in range(0,len(perm),64):
                        ids=perm[j:j+64];opt.zero_grad(set_to_none=True);loss=paper_loss(model(x[ids]),y[ids])
                        if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
                        loss.backward();opt.step();updates+=1;losses.append((float(loss.detach()),len(ids)))
                    model.eval()
                    with torch.inference_mode():valid=float(paper_loss(model(x[va]),y[va]))
                    vr=recalls(model,x,y,va);row={'epoch':epoch,'optimizer_updates':updates,'train_loss':sum(a*b for a,b in losses)/sum(b for a,b in losses),'validation_topic_loss':valid,'learning_rate':lr,'validation_core_recall':vr}
                    history.append(row)
                    if valid<best:
                        best=valid;selected=epoch;last_improved=epoch;torch.save(model.state_dict(),out/'best.pt')
                    if epoch%10==0 or epoch==1:event('repair_epoch',task=task,**row)
                    if epoch%25==0:write(out/'history.json',history)
                    if epoch>=50 and epoch-last_improved>=100:break
                write(out/'history.json',history)
                model.load_state_dict(torch.load(out/'best.pt',map_location='cuda',weights_only=True))
                candidate={'lr':lr,'epoch':selected,'validation_topic_loss':best,'path':str(out/'best.pt'),'updates_per_epoch':math.ceil(len(tr)/64),'total_updates':updates,'executed_epochs':epoch,'training_core_recall':recalls(model,x,y,tr),'validation_core_recall':recalls(model,x,y,va)}
                candidates.append(candidate);event('repair_trial_completed',task=task,**candidate)
                del model;torch.cuda.empty_cache()
            selected=min(candidates,key=lambda r:(r['validation_topic_loss'],r['epoch'],r['lr']))
            write(folder/'best_model.json',{'selected':selected,'candidates':candidates,'test_metric_used':False,'selection':'minimum independent internal-validation topic loss','seed':42,'max_epochs':1000,'patience':100,'minimum_epochs':50,'batch_size':64,'optimizer':'AdamW','weight_decay':0.01,'warmup_steps':0,'loss':'same literal paper Eq4, mean over batch/topics','best_checkpoint_sha256':file_sha(selected['path'])})
            all_selected[task]=selected
        gate={task:{'training_recall_at_10':r['training_core_recall']['recall_at_10'],'validation_recall_at_10':r['validation_core_recall']['recall_at_10'],'passed':r['training_core_recall']['recall_at_10']>=0.10 and r['validation_core_recall']['recall_at_10']>=0.05} for task,r in all_selected.items()}
        write(ROOT/'repair_diagnostic_gate.json',{'thresholds_fixed_before_training':{'train_recall_at_10':0.10,'validation_recall_at_10':0.05},'purpose':'minimal evidence of nontrivial topic learning, not a claim of optimal TopicK reproduction','tasks':gate,'all_passed':all(v['passed'] for v in gate.values()),'test_results_used':False})
        if not all(v['passed'] for v in gate.values()):raise RuntimeError('Second convergence failure: do not proceed to test or retry this training path')

if __name__=='__main__':train_repaired()

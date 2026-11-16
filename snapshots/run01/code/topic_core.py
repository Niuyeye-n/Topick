"""TopicK core: paper-supervised topic predictor and pinned public-code ranking.
No module in this file reads test gold or final reader predictions.
"""
import numpy as np
import torch
from torch import nn

class TopicPredictor(nn.Module):
    def __init__(self,topic_embeddings):
        super().__init__()
        self.topic_emb=nn.Parameter(torch.as_tensor(topic_embeddings,dtype=torch.float32),requires_grad=False)
        self.mlp=nn.Sequential(nn.Linear(768,768),nn.ReLU(),nn.Linear(768,768),nn.ReLU(),nn.Linear(768,768))
    def forward(self,x):return self.mlp(x)@self.topic_emb.T

def filtering(scores,quantile=0.1):
    threshold=torch.quantile(scores,quantile,dim=-1,keepdim=True)
    return torch.where(scores<=threshold,torch.zeros((),device=scores.device,dtype=scores.dtype),scores)

def rank_scores(scores,weight=0.53):
    order=torch.argsort(scores,dim=-1,descending=True)
    ranks=torch.zeros_like(scores,dtype=torch.long)
    ranks.scatter_(-1,order,torch.arange(scores.size(-1),device=scores.device).expand_as(order))
    return (1./(ranks+1).float())**weight

def zscore(x):return (x-x.mean())/(x.std(unbiased=False)+1e-8)

def select(model,query_embedding,pool_embeddings,knowledge,*,weight_rel=0.1,k=4):
    """Equivalent to public TopicK's pool-of-50 path; all indices are valid.
    Public code applies filtering/rank to query logits, not to their sigmoid.
    No silent correction of that upstream behavior is made here.
    """
    if not 0<k<=len(pool_embeddings):raise ValueError('Insufficient candidate pool')
    dev=next(model.parameters()).device
    q=torch.as_tensor(query_embedding,dtype=torch.float32,device=dev).reshape(1,-1)
    p=torch.as_tensor(pool_embeddings,dtype=torch.float32,device=dev)
    if p.shape[0]!=50:raise ValueError('Frozen protocol requires 50 candidates')
    with torch.inference_mode():
        # FAISS TopK orders the all-50 neighborhood by semantic relevance.
        order=torch.argsort((q@p.T).squeeze(0),descending=True)
        p=p[order]
        know=rank_scores(torch.as_tensor(np.clip(knowledge,0.1,0.9),dtype=torch.float32,device=dev))
        query=rank_scores(filtering(model(q)))
        rel=(q@p.T).squeeze(0)
        pool_prob=torch.sigmoid(model(p))
        candidates=rank_scores(filtering(pool_prob)/(know+1e-2))
        chosen=[];trace=[]
        for step in range(k):
            if chosen:
                current=p[chosen].mean(0,keepdim=True)
                combined=(p+len(chosen)*current)/(len(chosen)+1)
                candidates=rank_scores(filtering(torch.sigmoid(model(combined)))/(know+1e-2))
            topic=(query@candidates.T).squeeze(0)
            scores=zscore(topic)+weight_rel*zscore(rel)
            if not torch.isfinite(scores).all():raise FloatingPointError('Non-finite selector scores')
            scores[chosen]=-float('inf')
            idx=int(scores.argmax().item());chosen.append(idx)
            trace.append({'step':step+1,'pool_index':int(order[idx]),'topic_score':float(topic[idx]),'semantic_score':float(rel[idx]),'total_score':float(scores[idx])})
    if len(set(chosen))!=k:raise ValueError('Repeated demonstrations')
    return [int(order[i]) for i in chosen],trace

def paper_loss(logits,targets):
    # Literal Eq.(4): positive soft weight * log(p), absent-topic log(1-p).
    # No additional (1-y)*log(1-p) term on present topics.
    return -(targets*torch.nn.functional.logsigmoid(logits)+(targets==0).float()*torch.nn.functional.logsigmoid(-logits)).mean()

"""Frozen reader and prompt contract, shared by knowledge/dev/test evaluation."""
import datetime,json,pathlib,time,os
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig,LogitsProcessor
from eval_common import LETTERS,digest,file_sha,messages,write
from runtime import ROOT,ASSETS,event,protocol,read

class FrozenReader:
    def __init__(self,name):
        self.name=name;reference=read(ROOT/'reader_reference.json')['readers'][name]
        self.path=pathlib.Path(os.environ.get('TOPICK_'+name.upper()+'_MODEL', reference['model_path']))
        started=time.monotonic();files=[]
        for spec in reference['model_files']:
            p=self.path/spec['name']
            if not p.is_file() or p.stat().st_size!=spec['bytes']:raise ValueError('Reader artifact missing/wrong length: '+str(p))
            actual=file_sha(p)
            if actual!=spec['sha256']:raise ValueError('Reader hash mismatch: '+str(p))
            files.append({'name':spec['name'],'bytes':p.stat().st_size,'sha256':actual})
        event('reader_fingerprints_verified',reader=name,seconds=time.monotonic()-started)
        self.tokenizer=AutoTokenizer.from_pretrained(self.path,local_files_only=True)
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        self.tokenizer.padding_side='left'
        if not self.tokenizer.chat_template:raise ValueError('Missing frozen chat template')
        if digest(self.tokenizer.chat_template)!=reference['chat_template_sha256']:raise ValueError('Frozen chat template differs')
        ids=[self.tokenizer.encode(x,add_special_tokens=False) for x in LETTERS]
        if any(len(x)!=1 for x in ids):raise ValueError('Letter is not one token')
        self.allowed=[x[0] for x in ids]
        if dict(zip(LETTERS,self.allowed))!=reference['answer_tokens']:raise ValueError('Answer-token IDs differ')
        self.model=AutoModelForCausalLM.from_pretrained(self.path,local_files_only=True,torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa').eval()
        self.generation=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=1,pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id,bos_token_id=self.tokenizer.bos_token_id,repetition_penalty=1.,use_cache=True)
        self.manifest={'model_path':str(self.path),'model_files':files,'chat_template_sha256':digest(self.tokenizer.chat_template),'answer_tokens':dict(zip(LETTERS,self.allowed)),'attention_implementation':'sdpa','dtype':'float16','quantization':None,'generation_config':self.generation.to_dict(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
        import transformers
        self.manifest['transformers']=transformers.__version__
        torch.cuda.reset_peak_memory_stats()

    def prompt(self,query,demo_ids,pool):
        msgs=messages(query,demo_ids,pool,protocol()['evaluation']['system'])
        if len(msgs)!=2+2*len(demo_ids):raise ValueError('Wrong demonstration count in rendered messages')
        return msgs,self.tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)

    def answer(self,query,demo_ids,pool,*,save_prompt=False):
        msgs,text=self.prompt(query,demo_ids,pool)
        tokens=self.tokenizer(text,return_tensors='pt',add_special_tokens=False,truncation=False)
        length=int(tokens['input_ids'].shape[1])
        if length>protocol()['evaluation']['max_input_tokens']:raise ValueError(f'Overlong final input: {query["id"]}, {length}')
        allowed=self.allowed
        class Constraint(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if not torch.isfinite(scores).all():raise FloatingPointError('Non-finite pre-mask scores')
                mask=torch.full_like(scores,-float('inf'));mask[:,allowed]=scores[:,allowed]
                if not torch.isfinite(mask[:,allowed]).all():raise FloatingPointError('Non-finite allowed scores')
                return mask
        start=time.perf_counter()
        with torch.inference_mode():
            result=self.model.generate(**tokens.to('cuda'),generation_config=self.generation,logits_processor=[Constraint()],return_dict_in_generate=True,output_logits=True,remove_invalid_values=False)
        if len(result.logits)!=1 or not torch.isfinite(result.logits[0]).all():raise FloatingPointError('Non-finite full raw logits')
        generated=result.sequences[0,length:].tolist()
        if len(generated)!=1 or generated[0] not in allowed:raise ValueError('Invalid canonical output token')
        row={'id':query['id'],'pred':allowed.index(generated[0]),'demo_ids':list(demo_ids),'input_tokens':length,'prompt_sha256':digest(text),'generated_token_ids':generated,'raw_output':self.tokenizer.decode(generated),'letter_logits':result.logits[0][0,allowed].float().cpu().tolist(),'inference_seconds':time.perf_counter()-start}
        if save_prompt:row['messages']=msgs;row['rendered_prompt']=text
        return row

    def close(self):
        peak={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
        event('reader_closed',reader=self.name,peak_gpu_bytes=peak)
        del self.model;torch.cuda.empty_cache()
        return peak

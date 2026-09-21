from __future__ import annotations
import csv,os,random,tempfile
import numpy as np,torch,torch.distributed as dist
try:
 from torch.utils.tensorboard import SummaryWriter
except Exception:
 class SummaryWriter:
  def __init__(self,*a,**k): pass
  def add_scalar(self,*a,**k): pass
  def flush(self): pass
  def close(self): pass

def setup_distributed():
 ws=int(os.environ.get('WORLD_SIZE','1')); rank=int(os.environ.get('RANK','0')); lr=int(os.environ.get('LOCAL_RANK','0')); torch.cuda.set_device(lr); dev=torch.device('cuda',lr)
 if ws>1 and not dist.is_initialized(): dist.init_process_group(backend='nccl',init_method='env://')
 return rank,ws,lr,dev
def cleanup():
 if dist.is_initialized(): dist.destroy_process_group()
def barrier():
 if dist.is_initialized(): dist.barrier()
def reduce_sum(v,dev):
 t=torch.tensor([float(v)],dtype=torch.float64,device=dev); 
 if dist.is_initialized(): dist.all_reduce(t,op=dist.ReduceOp.SUM)
 return float(t.item())
def reduce_max(v,dev):
 t=torch.tensor([float(v)],dtype=torch.float64,device=dev)
 if dist.is_initialized(): dist.all_reduce(t,op=dist.ReduceOp.MAX)
 return float(t.item())
def set_seed(seed,rank):
 s=int(seed)+rank*100003; random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def append_csv(path,row):
 os.makedirs(os.path.dirname(path) or '.',exist_ok=True); new=not os.path.isfile(path)
 with open(path,'a',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=row.keys());
  if new: w.writeheader()
  w.writerow(row)
def atomic_save(obj,path):
 os.makedirs(os.path.dirname(path) or '.',exist_ok=True); tmp=path+'.tmp'; torch.save(obj,tmp); os.replace(tmp,path)
def make_optimizer(cfg,model): return torch.optim.AdamW(model.parameters(),lr=float(cfg.learning_rate),weight_decay=float(cfg.get('weight_decay',0.0)))
def make_scheduler(cfg,opt,total_steps):
 decay=max(1,int(total_steps*float(cfg.lr_decay_steps_fraction)))
 def lam(step):
  if step<decay: return 1.0
  progress=(step-decay)/max(total_steps-decay,1); ratio=float(cfg.min_learning_rate)/float(cfg.learning_rate); return max(ratio,ratio**progress)
 return torch.optim.lr_scheduler.LambdaLR(opt,lam),decay

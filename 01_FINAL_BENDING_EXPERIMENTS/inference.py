from __future__ import annotations
import argparse,json,os
import torch
from omegaconf import OmegaConf
from model_factory import create_model
from packed_dataset import load_normalization_stats
from rollout import evaluate_manifest
from target_space import load_target_stats

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',default='best'); ap.add_argument('--split',default='test',choices=['val','test']); args=ap.parse_args(); cfg=OmegaConf.load(args.config); device=torch.device('cuda:0')
 path=str(cfg.best_checkpoint) if args.checkpoint=='best' else (str(cfg.last_checkpoint) if args.checkpoint=='last' else args.checkpoint); path=os.path.abspath(path)
 model=create_model(cfg,device); ck=torch.load(path,map_location=device,weights_only=False); ts=load_target_stats(str(cfg.target_stats_path),str(cfg.prediction_mode));
 if str(ck.get('data_fingerprint','')) != str(ts['raw'].get('data_fingerprint','')): raise RuntimeError('checkpoint/data fingerprint mismatch; refusing stale evaluation')
 model.load_state_dict(ck['model'],strict=True); model.eval(); es=load_normalization_stats(str(cfg.stats_dir)); out=os.path.abspath(str(cfg.rollout_output_dir)); os.makedirs(out,exist_ok=True)
 res=evaluate_manifest(model,str(cfg.preprocess_output_dir),args.split,es['edge_mean'],es['edge_std'],str(cfg.target_stats_path),device,cfg,output_csv=os.path.join(out,f'{args.split}_metrics.csv'),collect_positions_dir=os.path.join(out,f'{args.split}_pred_npz'))
 summary={'checkpoint':path,'checkpoint_epoch':int(ck.get('epoch',-1)),'experiment_id':str(cfg.experiment_id),'prediction_mode':str(cfg.prediction_mode),'feature_mode':str(cfg.feature_mode),'processor_type':str(cfg.processor_type),'rollout':{k:v for k,v in res.items() if k!='rows'}}; sp=os.path.join(out,f'{args.split}_summary.json'); json.dump(summary,open(sp,'w'),indent=2); print(json.dumps(summary,indent=2)); print('summary:',sp)
if __name__=='__main__': main()

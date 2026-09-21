from pathlib import Path
from omegaconf import OmegaConf
from features import FEATURE_NAMES,num_input_features
from target_space import canonical_prediction_mode
EXPECTED={
'exp01_mgn_x10_direct_2step.yaml':('mgn','x10','direct',2,120,60,80),
'exp02_mgnt_x10_direct_2step.yaml':('mgn_t','x10','direct',2,120,60,128),
'exp03_global_anaresid_x10_2step.yaml':('mgn_t','x10','global_residual',2,120,60,128),
'exp04_local_anaresid_x10_2step.yaml':('mgn_t','x10','local_residual',2,120,60,128),
'exp05_local_anaresid_x3_2step.yaml':('mgn_t','x3','local_residual',2,120,60,128),
'exp06_local_anaresid_x6_2step.yaml':('mgn_t','x6','local_residual',2,120,60,128),
'exp07_local_x10_n040_2step.yaml':('mgn_t','x10','local_residual',2,40,180,128),
'exp08_local_x10_n060_2step.yaml':('mgn_t','x10','local_residual',2,60,120,128),
'exp09_local_x10_n080_2step.yaml':('mgn_t','x10','local_residual',2,80,90,128),
'exp10_local_x10_n120_1step.yaml':('mgn_t','x10','local_residual',1,120,60,128),
}
for fn,e in EXPECTED.items():
 c=OmegaConf.load(Path('conf')/fn); got=(str(c.processor_type),str(c.feature_mode),canonical_prediction_mode(c.prediction_mode),int(c.multistep_rollout_steps),int(c.expected_train_samples),int(c.max_epochs),int(c.hidden_dim_processor)); assert got==e,(fn,got,e); assert int(c.num_input_features)==num_input_features(c.feature_mode); assert tuple(c.node_feature_names)==FEATURE_NAMES[str(c.feature_mode)]; assert str(c.training_contract)=='final_bending_controlled_v1'; assert bool(c.resume_if_exists); assert int(c.target_optimizer_steps_per_rank)==322200
# Full-epoch 2-step schedules land exactly on 322200; 1-step stops mid final epoch at the same exact budget.
for fn in ('exp04_local_anaresid_x10_2step.yaml','exp07_local_x10_n040_2step.yaml','exp08_local_x10_n060_2step.yaml','exp09_local_x10_n080_2step.yaml'):
 c=OmegaConf.load(Path('conf')/fn); steps=(int(c.expected_train_samples)//4)*(181-2)*int(c.max_epochs); assert steps==322200,(fn,steps)
print('ALL FINAL EXPERIMENT CONTRACTS PASSED')

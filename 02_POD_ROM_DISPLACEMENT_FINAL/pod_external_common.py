from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import torch

from evaluate_rom import aggregate, metrics, predict
from pod_common import (
    canonical_representation, decode_scalar_string, fixed_to_native,
    load_case, representation_native_to_displacement,
)
from train_rom_mlp import ROMMLP


def sample_id_from_npz(path: Path) -> int:
    with np.load(path,allow_pickle=False) as z:
        key="Sample_ID" if "Sample_ID" in z.files else "sample_id"
        return int(round(float(np.asarray(z[key]).reshape(-1)[0])))


def map_npz_by_id(data_dir: Path) -> Dict[int,Path]:
    files=sorted(data_dir.glob("*.npz")); out={}
    for p in files:
        sid=sample_id_from_npz(p)
        if sid in out: raise RuntimeError(f"duplicate Sample_ID={sid} in {data_dir}")
        out[sid]=p
    return out


def load_frozen_rom(basis: Path,dataset: Path,checkpoint: Path,representation: str):
    rep=canonical_representation(representation)
    b=np.load(basis,allow_pickle=False); br=decode_scalar_string(b["representation"])
    d=np.load(dataset,allow_pickle=False); dr=decode_scalar_string(d["representation"])
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False); cr=str(ck.get("representation",""))
    if not (br==dr==cr==rep):
        raise RuntimeError(f"representation mismatch basis={br} dataset={dr} checkpoint={cr} requested={rep}")
    if str(ck.get("coefficient_output_space",""))!="normalized_by_global_train_max":
        raise RuntimeError("unexpected POD coefficient output space")
    mean=np.asarray(b["mean"],dtype=np.float32); modes=np.asarray(b["modes"],dtype=np.float32)
    n_s,n_phi=[int(v) for v in b["grid"]]; rank=int(ck["rank"]); modes_r=modes[:rank]
    gm=np.asarray(ck["geom_mean"],dtype=np.float32); gs=np.asarray(ck["geom_std"],dtype=np.float32)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ROMMLP(int(ck["n_inputs"]),rank,int(ck["width"]),int(ck["depth"])); model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    return {"rep":rep,"mean":mean,"modes_r":modes_r,"n_s":n_s,"n_phi":n_phi,"rank":rank,
            "gm":gm,"gs":gs,"device":device,"model":model,"ck":ck}


def eval_case(path: Path, frozen: Dict):
    rep=frozen["rep"]; mean=frozen["mean"]; modes_r=frozen["modes_r"]
    n_s=frozen["n_s"]; n_phi=frozen["n_phi"]; ck=frozen["ck"]
    c=load_case(path,n_s,n_phi,rep); T=c.U_true.shape[0]
    X=c.representation_fixed.reshape(T,-1); coeff_true=(X-mean[None,:])@modes_r.T
    coeff_pred_norm=predict(frozen["model"],c.geom,c.tau,frozen["gm"],frozen["gs"],frozen["device"])
    coeff_pred=coeff_pred_norm*np.float32(ck["y_scale"])
    rt_native=fixed_to_native(c.representation_fixed,c.grid_ids,c.s_src,c.phi_src)
    rt_U=representation_native_to_displacement(c,rt_native,rep)
    oracle_fixed=(mean[None,:]+coeff_true@modes_r).reshape(T,n_s,n_phi,3)
    oracle_native=fixed_to_native(oracle_fixed.astype(np.float32),c.grid_ids,c.s_src,c.phi_src)
    oracle_U=representation_native_to_displacement(c,oracle_native,rep)
    rom_fixed=(mean[None,:]+coeff_pred@modes_r).reshape(T,n_s,n_phi,3)
    rom_native=fixed_to_native(rom_fixed.astype(np.float32),c.grid_ids,c.s_src,c.phi_src)
    rom_U=representation_native_to_displacement(c,rom_native,rep)
    return c, {
        "roundtrip":metrics(rt_U,c.U_true,c.free_mask),
        "pod_oracle":metrics(oracle_U,c.U_true,c.free_mask),
        "rom":metrics(rom_U,c.U_true,c.free_mask),
    }

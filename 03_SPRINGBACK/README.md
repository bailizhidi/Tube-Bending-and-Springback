# MGN_springback_onestep_150_singlemat — HPC formal version

This is the formal **single-material Base_TC4 / 150-geometry** springback retraining package for the supercomputer.

Please read `README_HPC_CN.md` first.

Formal target:

```text
dU_springback = U_last - U_frame0
X_spring_pred = X_bend + dU_springback_pred
```

The server-side package does **not** read Abaqus ODB files. ODB -> NPZ extraction is completed on Windows/Abaqus first; only the 5400 pair NPZ files are uploaded to the supercomputer.

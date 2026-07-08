#!/usr/bin/env python3
# Per-tensor step-1 delta (trained - init) sanity for the 2-chip shim dump.
import numpy as np, sys, os
D,FFN,VOCAB,SEQ = 512,2048,256,256
# FMaster field order (name, count)
FIELDS = [("WQ",D*D),("WK",D*D),("WV",D*D),("WO",D*D),("WUP",FFN*D),("WDN",D*FFN),
          ("WHEAD",VOCAB*D),("EMB",VOCAB*D),("POS",SEQ*D),
          ("LN1G",D),("LN1B",D),("LN2G",D),("LN2B",D),("LNFG",D),("LNFB",D)]
INIT = sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(__file__),"..","s2_locinit")
SHIM = sys.argv[2] if len(sys.argv)>2 else os.path.join(os.path.dirname(__file__),"out")
def load(path):
    a=np.fromfile(path,dtype=np.float32); off=0; out={}
    for name,cnt in FIELDS: out[name]=a[off:off+cnt]; off+=cnt
    assert off==a.size, f"{path}: {off} != {a.size}"
    return out
for chip in (0,1):
    ini=load(os.path.join(INIT,f"master{chip}.f32"))
    shm=load(os.path.join(SHIM,f"shim_master{chip}.f32"))
    role = "HID+hostembed" if chip==0 else "OUT"
    print(f"\n=== chip{chip} ({role}) ===")
    print(f"{'tensor':7} {'nchanged':>9} {'frac':>7} {'maxabs_delta':>13} {'mean_abs_delta':>14} {'nan/inf':>8}")
    for name,cnt in FIELDS:
        d = shm[name]-ini[name]
        nz = np.count_nonzero(d)
        nan = int(np.isnan(d).sum()+np.isinf(d).sum())
        mad = float(np.abs(d).max()) if d.size else 0.0
        mnd = float(np.abs(d).mean()) if d.size else 0.0
        flag=""
        if nan: flag="<-NANINF!"
        print(f"{name:7} {nz:9d} {nz/max(1,cnt):7.3f} {mad:13.3e} {mnd:14.3e} {nan:8d} {flag}")

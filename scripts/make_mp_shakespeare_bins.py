#!/usr/bin/env python3
# make_mp_shakespeare_bins.py — build train.u16/val.u16 + itos.txt with the EXACT tokenization the bf16 golden uses
# (forge_pc_xformer_bf16_golden_test.cpp): first 64 distinct chars in order of appearance -> ids 0..63; rest -> 0.
# So the on-silicon run reproduces the golden's bar, and itos decodes sampled tokens back to text.
#   python3 scripts/make_mp_shakespeare_bins.py ~/.cache/tiur/tinyshakespeare.txt /tmp/mpbins
import sys, os, json, struct
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
txt = open(src,'rb').read(); mp={}; vocab=0
for c in txt:
    if c not in mp and vocab < 64: mp[c]=vocab; vocab+=1
data=[mp.get(c,0) for c in txt]; cut=int(len(data)*0.9)
w=lambda p,t: open(p,'wb').write(b''.join(struct.pack('<H',x) for x in t))
w(f'{out}/train.u16', data[:cut]); w(f'{out}/val.u16', data[cut:])
itos={v:k for k,v in mp.items()}; open(f'{out}/itos.txt','wb').write(bytes(itos[i] for i in range(vocab)))
json.dump({str(i):itos[i] for i in range(vocab)}, open(f'{out}/vocab.json','w'))
print(f'vocab={vocab} tokens={len(data)} train={cut} val={len(data)-cut}')

import json,subprocess,struct,collections
base="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/"
idx=json.load(open('/tmp/v41idx.json'))
files=sorted(set(idx['weight_map'].values()))
def rng(u,a,b):
    for _ in range(4):
        p=subprocess.run(['curl','-sL','--retry','3','--max-time','90','-r',f'{a}-{b}',u],capture_output=True)
        if p.returncode==0 and p.stdout: return p.stdout
    raise RuntimeError(u)
tot=collections.Counter(); cnt=collections.Counter(); dt=collections.Counter()
for i,f in enumerate(files,1):
    u=base+f
    n=struct.unpack('<Q',rng(u,0,7))[0]
    hdr=json.loads(rng(u,8,8+n-1))
    for k,v in hdr.items():
        if k=='__metadata__': continue
        sz=v['data_offsets'][1]-v['data_offsets'][0]
        if k.startswith('mtp'): c='mtp'
        elif '.experts.' in k: c='routed_experts'
        elif 'engram' in k.lower(): c='engram'
        elif 'shared_expert' in k: c='shared_expert'
        elif 'vision' in k.lower(): c='vision'
        elif 'embed' in k or 'lm_head' in k or 'head' in k: c='embed/head'
        else: c='attn/other'
        tot[c]+=sz; cnt[c]+=1; dt[v['dtype']]+=sz
    print(f'{i}/{len(files)} {f}',flush=True)
T=sum(tot.values())
print('=== BREAKDOWN')
for k,v in tot.most_common(): print(f'{k:16s} {v/1e9:8.1f} GB  {100*v/T:5.1f}%  ({cnt[k]} tensors)')
print('TOTAL',round(T/1e9,1),'GB')
print('=== DTYPE')
for k,v in dt.most_common(): print(f'{k:10s} {v/1e9:8.1f} GB')

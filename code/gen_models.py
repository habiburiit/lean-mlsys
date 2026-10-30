#!/usr/bin/env python3
"""gen_models.py - writes models/<name>/probs/*.yaml + layers.json for every model in the LEAN paper plus two LLMs.
Each layer: (tag, C, K, R, S, P, Q, stride, count).  Linear layers are 1x1 convs with P=Q=1 (decode: N=1 token).
LLM prefill rows use N = tokens so the weights are reused across the sequence."""
import json, os
def conv(tag,C,K,k,P,s=1,n=1): return (tag,C,K,k,k,P,P,s,n)
def fc(tag,C,K,n=1): return (tag,C,K,1,1,1,1,1,n)
M = {}
# ---- ImageNet ----
M["imagenet_resnet18"] = [conv("conv1",3,64,7,112,2),conv("l1",64,64,3,56,1,4),conv("l2a",64,128,3,28,2),conv("l2",128,128,3,28,1,3),conv("l2ds",64,128,1,28,2),
  conv("l3a",128,256,3,14,2),conv("l3",256,256,3,14,1,3),conv("l3ds",128,256,1,14,2),conv("l4a",256,512,3,7,2),conv("l4",512,512,3,7,1,3),conv("l4ds",256,512,1,7,2),fc("fc",512,1000)]
M["imagenet_vgg16"] = [conv("c1",3,64,3,224),conv("c2",64,64,3,224),conv("c3",64,128,3,112),conv("c4",128,128,3,112),conv("c5",128,256,3,56),conv("c6",256,256,3,56,1,2),
  conv("c8",256,512,3,28),conv("c9",512,512,3,28,1,2),conv("c11",512,512,3,14,1,3),fc("fc1",25088,4096),fc("fc2",4096,4096),fc("fc3",4096,1000)]
# ---- CIFAR (32x32 inputs, the usual CIFAR variants: 3x3 stem, no maxpool) ----
M["cifar_resnet18"] = [conv("conv1",3,64,3,32),conv("l1",64,64,3,32,1,4),conv("l2a",64,128,3,16,2),conv("l2",128,128,3,16,1,3),conv("l2ds",64,128,1,16,2),
  conv("l3a",128,256,3,8,2),conv("l3",256,256,3,8,1,3),conv("l3ds",128,256,1,8,2),conv("l4a",256,512,3,4,2),conv("l4",512,512,3,4,1,3),conv("l4ds",256,512,1,4,2),fc("fc",512,10)]
def bottleneck_stage(tag,cin,mid,P,s,n,L):
    L += [conv(f"{tag}_a1",cin,mid,1,P if s==1 else P*s),conv(f"{tag}_b1",mid,mid,3,P,s),conv(f"{tag}_c",mid,4*mid,1,P,1,n),conv(f"{tag}_ds",cin,4*mid,1,P,s)]
    if n>1: L += [conv(f"{tag}_a",4*mid,mid,1,P,1,n-1),conv(f"{tag}_b",mid,mid,3,P,1,n-1)]
L=[conv("conv1",3,64,3,32)]
for tag,cin,mid,P,s,n in (("s1",64,64,32,1,3),("s2",256,128,16,2,4),("s3",512,256,8,2,6),("s4",1024,512,4,2,3)): bottleneck_stage(tag,cin,mid,P,s,n,L)
M["cifar_resnet50"] = L+[fc("fc",2048,10)]
M["cifar_vgg16"] = [conv("c1",3,64,3,32),conv("c2",64,64,3,32),conv("c3",64,128,3,16),conv("c4",128,128,3,16),conv("c5",128,256,3,8),conv("c6",256,256,3,8,1,2),
  conv("c8",256,512,3,4),conv("c9",512,512,3,4,1,2),conv("c11",512,512,3,2,1,3),fc("fc",512,10)]
# ---- LLMs: one transformer block x layers; decode = 1 token (GEMV), prefill = N tokens ----
def llm(tag,h,ffn,nl,kv,tokens):
    rows=[("q",h,h,1,1,1,1,1,nl),("k",h,kv,1,1,1,1,1,nl),("v",h,kv,1,1,1,1,1,nl),("o",h,h,1,1,1,1,1,nl),("gate",h,ffn,1,1,1,1,1,nl),("up",h,ffn,1,1,1,1,1,nl),("down",ffn,h,1,1,1,1,1,nl)]
    return [(t,C,K,R,S,P,Q,s,n,tokens) for t,C,K,R,S,P,Q,s,n in rows]
M["llama2_7b_decode"]  = llm("l",4096,11008,32,4096,1)
M["llama2_7b_prefill512"] = llm("l",4096,11008,32,4096,512)
M["llama32_1b_decode"] = llm("l",2048,8192,16,512,1)
for name,layers in M.items():
    d=f"models/{name}"; os.makedirs(d+"/probs",exist_ok=True); os.makedirs(d+"/runs",exist_ok=True); spec={}
    for row in layers:
        tag,C,K,R,S,P,Q,s,n=row[:9]; N=row[9] if len(row)>9 else 1
        open(f"{d}/probs/{tag}.yaml","w").write(f"problem:\n  shape: cnn-layer\n  C: {C}\n  K: {K}\n  R: {R}\n  S: {S}\n  P: {P}\n  Q: {Q}\n  N: {N}\n  Wstride: {s}\n  Hstride: {s}\n  Wdilation: 1\n  Hdilation: 1\n")
        spec[tag]=[C,K,R,S,P,Q,s,n]
    json.dump(spec,open(f"{d}/probs/layers.json","w"))
    for f in ("arch_int8.yaml","constraints_rs.yaml","mapper.yaml","run_all.sh"):
        if not os.path.exists(f"{d}/{f}"): os.symlink(f"../../{f}",f"{d}/{f}")
    print(name,len(layers),"shapes, weights",sum(C*K*R*S*n for _,C,K,R,S,_,_,_,n in [r[:9] for r in layers])/1e6,"M")

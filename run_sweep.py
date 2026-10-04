import sys, json
sys.path.insert(0,'research/blackbox'); sys.path.insert(0,'research/attacks_core')
from sta_core import load_tokenizer
from fluency import Fluency
from beam import Semantic
from oracle import DetectorOracle
from span_attack import gen_span_z
from para_attack import attack

tok = load_tokenizer(); fl = Fluency(tok); sem = Semantic()
base = {r['prompt_id']: r for f in ['results/raw/baseline_safe.json','results/raw/baseline_extra.json'] for r in json.load(open(f))}

out_all = []
print('%4s %8s %8s %6s %8s %7s' % ('id','z0','z1','sents','queries','broke'), flush=True)
for pid in (25,20,0,22,24,2,23,1,27):
    r = base[pid]
    o = DetectorOracle(tok, mode='score')
    sz = lambda t: gen_span_z(tok, t, r['prompt_tokens'])
    z0 = sz(r['watermarked_text'])
    try:
        a, z1, t = attack(o, tok, fl.model, r['watermarked_text'],
                          len(r['prompt']), n_cands=60, max_sents=12,
                          span_z=sz, semantic=sem, sem_threshold=0.88,
                          fluency=fl, log=lambda *x: None)
    except Exception as e:
        print('%4d FAILED %s' % (pid, e), flush=True); continue
    out_all.append({'prompt_id':pid,'z0':z0,'z1':z1,'sentences':len(a),
                    'queries':o.queries,'broke':t is not None,
                    'rewrites':a,'text':t})
    print('%4d %8.3f %8.3f %6d %8d %7s' % (pid,z0,z1,len(a),o.queries,t is not None), flush=True)
    json.dump(out_all, open('results/raw/para_sweep.json','w'), indent=2)

print('', flush=True)
print('broke: %d of %d' % (sum(1 for x in out_all if x['broke']), len(out_all)), flush=True)

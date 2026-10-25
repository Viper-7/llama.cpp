#!/usr/bin/env python3
"""Isolated CPU synthetic media: exact tokens, image reuse and decode interleaving."""
import argparse, base64, concurrent.futures, hashlib, http.server, json, math, os, struct, subprocess, threading, time
from pathlib import Path
from test_glm53_live_concurrency import call, request
# Exceed a 512-token worker quantum so partial-media progress is observable.
MEDIA_TOKENS = 1024
prepared_media = threading.Event()
class Vision(http.server.BaseHTTPRequestHandler):
    def log_message(self,*a):pass
    def do_POST(self):
        body=self.rfile.read(int(self.headers['Content-Length']))
        variant=2 if b'changed-fixture' in body else 1
        data=struct.pack('<'+'f'*(MEDIA_TOKENS*128),*[.03*math.sin(i*.1+variant) for i in range(MEDIA_TOKENS*128)])
        self.send_response(200)
        for k,v in {'X-N-Visual-Tokens':str(MEDIA_TOKENS),'X-Hidden-Size':'128','X-N-Pos-Per-Embd':'1','X-N-Pos':str(MEDIA_TOKENS),
                    'X-Dtype':'fp32','X-Pos-Path':'/positions','X-Sha256':hashlib.sha256(data).hexdigest(),
                    'Content-Length':str(len(data))}.items():self.send_header(k,v)
        self.end_headers();self.wfile.write(data);prepared_media.set()
    def do_GET(self):
        data=struct.pack('<'+'i'*MEDIA_TOKENS,*range(MEDIA_TOKENS))
        self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
def media(slot,variant='same-fixture',cache='retained',prefix=160):
    return call('/completion',{'prompt':{'prompt_string':'<s>'*prefix+'<__media__>'+'</s>'*10,
        'multimodal_data':[base64.b64encode(variant.encode()).decode()]},
        'id_slot':slot,'n_predict':32,'temperature':0,'seed':42,'ignore_eos':True,'return_tokens':True,
        'cache_control':{'lifecycle':cache}})
def start(binary,fixture,logpath,owner_ubatch=512,serial=False):
    env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_SERVER_PAGE_CACHE','LLAMA_ARG','LD_LIBRARY_PATH'))}
    if serial:
        for key in list(env):
            if key.startswith('LLAMA_GLM5NEXT_'):
                del env[key]
    env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',LLAMA_MEDIA_MARKER='<__media__>')
    log=open(logpath,'w')
    p=subprocess.Popen([binary,'-m',fixture,'--host','127.0.0.1','--port','18087','-ngl','0','--override-kv','glm5next.context_length=int:4096',
        '-c','4096','-np','3','--kv-unified','--kv-unified-per-slot','2048','-b','1024','-ub',str(owner_ubatch) if env.get('LLAMA_GLM5NEXT_PREFILL_WORKERS') else '64',
        '-t','2','-tb','2','--cache-page-ram','64','--cache-page-tokens','128','--no-warmup',
        '--flash-attn','on','--visiontower-url','http://127.0.0.1:18088'],env=env,stdout=log,stderr=log)
    for _ in range(120):
        if p.poll() is not None:raise RuntimeError(Path(logpath).read_text()[-1500:])
        try:
            if call('/health').get('status')=='ok':return p,log
        except OSError:pass
        time.sleep(.25)
    raise RuntimeError('readiness timeout')
def stop(p,log):
    p.terminate()
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:p.kill();p.wait()
    log.close()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binary',required=True);ap.add_argument('--baseline',required=True)
    ap.add_argument('--fixture',required=True);ap.add_argument('--log',required=True);ap.add_argument('--owner-ubatch',type=int,choices=(256,512),default=512);a=ap.parse_args()
    assert all('build-cpu/bin/llama-server' in b for b in (a.binary,a.baseline))
    assert 'glm53-concurrency-fixture' in a.fixture
    vision=http.server.ThreadingHTTPServer(('127.0.0.1',18088),Vision)
    threading.Thread(target=vision.serve_forever,daemon=True).start()
    try:
        p,l=start(a.baseline,a.fixture,a.log+'.baseline',a.owner_ubatch,serial=True)
        try:
            reference=media(0,cache='one_shot');changed=media(0,'changed-fixture',cache='one_shot')
            pair_references=[media(0,variant,cache='one_shot',prefix=prefix)
                for variant,prefix in [('same-fixture',173),('changed-fixture',177)]]
        finally:stop(p,l)
        p,l=start(a.binary,a.fixture,a.log,a.owner_ubatch)
        try:
            seen=[]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                decode=ex.submit(request,0,7,100,1000)
                for _ in range(1000):
                    slots=call('/slots')
                    if slots[0].get('next_token',[{'n_decoded':0}])[0]['n_decoded']>2:break
                    time.sleep(.01)
                cold=ex.submit(media,1)
                while not cold.done():
                    slots=call('/slots')
                    offset=slots[1].get('media_tokens_processed',0)
                    if offset:seen.append((offset,slots[0].get('next_token',[{'n_decoded':0}])[0]['n_decoded']))
                    time.sleep(.01)
                out=cold.result();decode.result()
            assert reference['tokens']==out['tokens'],'sliced media changed tokens'
            assert len({d for _,d in seen})>1,seen
            repeated=media(2)
            assert repeated['tokens']==reference['tokens']
            assert repeated['timings']['cache_n']>=512,repeated['timings']
            different=media(2,'changed-fixture')
            assert different['tokens']==changed['tokens'],'changed media reused wrong state'
            assert different['timings']['cache_n']<repeated['timings']['cache_n']
            if os.environ.get('LLAMA_GLM5NEXT_EXECUTION_LANES'):
                timing=call('/v1/internal/cache-probe',{'prompt':[1]})['execution_timing']
                assert timing['media_preparations_started']==2,timing
                assert timing['media_preparations_completed']==2,timing
                assert timing['execution_waves_with_media']>0,timing
            if os.environ.get('LLAMA_GLM5NEXT_EXECUTION_LANES'):
                before=call('/v1/internal/cache-probe',{'prompt':[1]})['execution_timing']['execution_waves_with_media']
                gate=threading.Barrier(2)
                def paired(slot,variant,prefix):
                    gate.wait()
                    return media(slot,variant,cache='one_shot',prefix=prefix)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    pair=[ex.submit(paired,slot,variant,prefix) for slot,variant,prefix in
                        [(0,'same-fixture',173),(1,'changed-fixture',177)]]
                    actual_pair=[future.result() for future in pair]
                assert all(a['tokens']==b['tokens'] for a,b in zip(actual_pair,pair_references)), 'media/media wave changed output'
                after=call('/v1/internal/cache-probe',{'prompt':[1]})['execution_timing']['execution_waves_with_media']
                assert after>before, 'paired media never dispatched independently'
                print(json.dumps({'media_text_and_media_media_waves_exact':True,'additional_media_waves':after-before}),flush=True)
            # A fourth HTTP request prepares media while all three GPU slots
            # remain occupied. Its inference must wait for one of those slots.
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                decoders=[ex.submit(request,s,s*13+9,100,1000) for s in range(3)]
                deadline=time.monotonic()+30
                while sum(s['is_processing'] for s in call('/slots')) != 3:
                    assert time.monotonic()<deadline
                    time.sleep(.01)
                prepared_media.clear()
                queued=ex.submit(media,-1)
                assert prepared_media.wait(10), 'queued media did not prepare'
                assert not any(f.done() for f in decoders), 'preparation did not overlap occupied slots'
                assert sum(s['is_processing'] for s in call('/slots')) == 3
                queued_result=queued.result()
                for future in decoders: future.result()
                assert queued_result['tokens']==reference['tokens'], 'queued preparation changed output'
            print(json.dumps({'queued_media_prepared_with_three_busy_slots':True}),flush=True)
            print(json.dumps({'media_serial_equivalence':True,'decode_advanced_during_partial_media':True,
                'image_cache_hit':repeated['timings']['cache_n'],'changed_media_cache_tokens':different['timings']['cache_n'],
                'observed_decode_counts':sorted({d for _,d in seen})}),flush=True)
        finally:stop(p,l)
    finally:vision.shutdown();vision.server_close()
if __name__=='__main__':main()

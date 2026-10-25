#!/usr/bin/env python3
"""CPU-only regression: output reservations, exact SSD parking and decode progress."""
import argparse, concurrent.futures, json, os, subprocess, threading, time, urllib.request
from pathlib import Path

BASE='http://127.0.0.1:18087'
def call(path, body=None):
    req=urllib.request.Request(BASE+path, data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as r:return json.load(r)
def request(slot, offset, length=350, predict=800):
    return call('/completion', {'prompt':[1+(i+offset)%100 for i in range(length)],
        'id_slot':slot,'n_predict':predict,'temperature':0,'seed':42,'ignore_eos':True,
        'cache_prompt':True,'return_tokens':True})
def cancellation_isolation():
    refs=[request(0,s*13,length=100,predict=64) for s in (1,2)]
    gate=threading.Barrier(3)
    def cancelled_stream():
        body={'prompt':[1+i%100 for i in range(100)],'id_slot':0,'n_predict':1200,
            'temperature':0,'seed':42,'ignore_eos':True,'stream':True,'cache_prompt':True}
        req=urllib.request.Request(BASE+'/completion',data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json'})
        gate.wait()
        with urllib.request.urlopen(req,timeout=60) as response:
            count=0
            for line in response:
                if not line.startswith(b'data: '):continue
                event=json.loads(line[6:])
                if event.get('content'):count+=1
                if count>=3:return
        raise AssertionError('stream ended before cancellation')
    def survivor(slot):
        gate.wait()
        return request(slot,slot*13,length=100,predict=64)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        cancelled=pool.submit(cancelled_stream)
        remaining=[pool.submit(survivor,s) for s in (1,2)]
        cancelled.result()
        actual=[future.result() for future in remaining]
    for expected,result in zip(refs,actual):
        assert result['tokens']==expected['tokens'],'cancellation changed another request'
    deadline=time.monotonic()+10
    while any(s['is_processing'] for s in call('/slots')) and time.monotonic()<deadline:time.sleep(.02)
    assert not any(s['is_processing'] for s in call('/slots')),'cancelled slot did not release'


def chat_registry_isolation():
    assert call('/v1/internal/responses/capabilities')['chat_request_registry']
    rid='job_'+'7'*32+':0'
    started=threading.Event()
    cancelled_ack=threading.Event()
    body={'messages':[{'role':'user','content':'<s>'*10}], 'stream':True,
          'max_tokens':1000,'temperature':0,'ignore_eos':True,
          'serverless_request_id':rid}
    def stream_chat():
        req=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=60) as response:
            response.fp.raw._sock.settimeout(2)
            try:
                for line in response:
                    if cancelled_ack.is_set():break
                    if line.startswith(b'data: ') and b'[DONE]' not in line:
                        event=json.loads(line[6:])
                        if event.get('choices'):started.set()
            except TimeoutError:
                # The bridge closes its reader after the native release fence.
                assert cancelled_ack.is_set(), 'chat reader stalled before cancellation'
    expected=request(0,23,length=100,predict=128)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        chat=pool.submit(stream_chat)
        if not started.wait(20):
            if chat.done():
                try:chat.result()
                except urllib.error.HTTPError as exc:
                    raise AssertionError('fixture chat HTTP: '+exc.read().decode()) from exc
            raise AssertionError('chat never began')
        survivor=pool.submit(request,1,23,100,128)
        deadline=time.monotonic()+10
        while True:
            proof=call('/v1/internal/responses/cancel',{'request_id':rid})
            if proof['released']:
                cancelled_ack.set()
                break
            assert time.monotonic()<deadline,proof
            time.sleep(.01)
        chat.result()
        assert survivor.result()['tokens']==expected['tokens'],'chat cancellation altered peer'
    # A cancellation tombstone must reject a late chat before preparation.
    cancelled='job_'+'8'*32+':0'
    assert call('/v1/internal/responses/cancel',{'request_id':cancelled})['released']
    try:
        call('/v1/chat/completions',{**body,'stream':False,'serverless_request_id':cancelled})
        raise AssertionError('cancelled identity dispatched')
    except urllib.error.HTTPError:
        pass

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binary',required=True);ap.add_argument('--fixture',required=True)
    ap.add_argument('--log',required=True)
    ap.add_argument('--execution-lanes',action='store_true')
    ap.add_argument('--resident-prefix',action='store_true')
    ap.add_argument('--chat-only',action='store_true')
    args=ap.parse_args()
    assert 'build-cpu/bin/llama-server' in args.binary
    assert 'glm53-concurrency-fixture' in args.fixture
    env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_SERVER_PAGE_CACHE','LLAMA_ARG','LD_LIBRARY_PATH'))}
    env['CUDA_VISIBLE_DEVICES']='';env['HIP_VISIBLE_DEVICES']=''
    env.pop('LLAMA_GLM5NEXT_EXECUTION_LANES',None)
    if args.execution_lanes:env['LLAMA_GLM5NEXT_EXECUTION_LANES']='3'
    with open(args.log,'w') as log:
        proc=subprocess.Popen([args.binary,'-m',args.fixture,'--host','127.0.0.1','--port','18087',
            '-ngl','0','--override-kv','glm5next.context_length=int:4096','-c','2048','-np','3','--kv-unified','--kv-unified-per-slot','2048',
            '-b','256','-ub','64','-t','2','-tb','2','--cache-page-ram','64','--cache-page-tokens','128',
            '--no-warmup','--flash-attn','on','--chat-template',"{{ messages[0]['content'] }}"],env=env,stdout=log,stderr=log)
        try:
            for _ in range(120):
                if proc.poll() is not None:raise RuntimeError('CPU server failed: '+Path(args.log).read_text()[-1500:])
                try:
                    if call('/health').get('status')=='ok':break
                except OSError:pass
                time.sleep(.25)
            else:raise RuntimeError('CPU readiness timeout')
            if args.chat_only:
                chat_registry_isolation()
                print(json.dumps({'chat_registry_isolation':True}),flush=True)
                return
            refs=[request(0,s*13) for s in range(3)]
            gate=threading.Barrier(3);seen={'max_processing':0,'parked':False}
            def run(s):
                gate.wait()
                return request(s,s*13)
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                fs=[ex.submit(run,s) for s in range(3)]
                while not all(f.done() for f in fs):
                    slots=call('/slots')
                    seen['max_processing']=max(seen['max_processing'],sum(s['is_processing'] and not s.get('capacity_parked') for s in slots))
                    seen['parked'] |= any(s.get('capacity_parked') for s in slots)
                    time.sleep(.02)
                actual=[f.result() for f in fs]
            for ref,out in zip(refs,actual):
                assert out['tokens_predicted']==800,out['tokens_predicted']
                assert ref['tokens']==out['tokens'],'park/resume changed output'
            assert seen['max_processing']==3,seen
            assert seen['parked'],seen
            assert not any(x['is_processing'] for x in call('/slots'))
            from test_glm53_cpu_concurrency import main as original_regressions
            original_regressions(require_resident_prefix=args.resident_prefix)
            cancellation_isolation()
            chat_registry_isolation()
            seen['cancellation_isolation']=True
            if args.execution_lanes:
                memory=call('/v1/internal/cache-probe',{'prompt':[1]})['execution_memory']
                assert memory['logical_slots']==3 and len(memory['lanes'])==3, memory
                assert memory['sample']>1 and all(sum(b['compute_bytes'] for b in lane['buffers'])>0 for lane in memory['lanes']), memory
                timing=call('/v1/internal/cache-probe',{'prompt':[1]})['execution_timing']
                assert timing['independent_contexts']==3,timing
                assert sum(timing['execution_waves_by_lanes'][1:])>0,timing
                assert sum(timing['execution_prefill_waves_by_sequences'][1:])>0,timing
                assert timing['execution_mixed_prefill_decode_waves']>0,timing
                assert timing['execution_decode_resubmissions']>0,timing
                assert timing['execution_decode_completions_while_prefill_pending']>0,timing
                seen['execution_timing']=timing
            print(json.dumps({'exact_serial_equivalence':True,'output_tokens_each':800,**seen}),flush=True)
        finally:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
if __name__=='__main__':main()

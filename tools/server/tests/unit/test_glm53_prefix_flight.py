#!/usr/bin/env python3
"""CPU fixture: resident KV pressure, exact prefix dependencies and cancellation."""
import argparse
import concurrent.futures
import json
import os
import random
import subprocess
import threading
import time
import urllib.request
import test_glm53_live_concurrency as helpers

helpers.BASE = 'http://127.0.0.1:18088'
call = helpers.call


def payload(tokens, slot=-1, priority=2, rid=None):
    body = dict(prompt=tokens, n_predict=8, temperature=0, seed=42,
                ignore_eos=True, return_tokens=True, cache_prompt=True,
                serverless_execution_priority=priority)
    if slot >= 0: body['id_slot'] = slot
    if rid: body['serverless_request_id'] = rid
    return body


def stream(body, events, first=None, stopped=None):
    req = urllib.request.Request(helpers.BASE+'/completion', data=json.dumps(
        dict(body, stream=True, return_progress=True)).encode(), headers={'Content-Type': 'application/json'})
    final = None
    with urllib.request.urlopen(req, timeout=240) as response:
        if stopped: response.fp.raw._sock.settimeout(2)
        try:
            for line in response:
                if not line.startswith(b'data: ') or b'[DONE]' in line: continue
                if stopped and stopped.is_set(): return final
                event = json.loads(line[6:])
                assert 'error' not in event, event
                event['_observed_at'] = time.monotonic()
                events.append(event)
                if first and event.get('prompt_progress', {}).get('processed', 0) >= 128: first.set()
                if event.get('stop'): final = event
        except TimeoutError:
            if not stopped or not stopped.is_set(): raise
    if final is not None:
        final['_tokens'] = [token for event in events if 'prompt_progress' not in event
                            for token in event.get('tokens', []) if token >= 0]
    return final


def tokens(seed, length):
    rng = random.Random(seed)
    return [rng.randrange(1, 110) for _ in range(length)]


def capacity_case(restore_only=False):
    cold = tokens(1000, 1500)
    if not restore_only:
        for slot in range(1, 6): call('/completion', payload(tokens(slot, 700), slot))
    before = call('/v1/internal/cache-probe', {'prompt': cold})
    result = call('/completion', payload(cold, 0))
    assert result['timings']['prompt_n'] == 1500, result['timings']
    # Leave the desired checkpoint on RAM/SSD while foreign slots fill GPU KV.
    call('/completion', payload(tokens(1001, 32), 0))
    for slot in range(1, 6): call('/completion', payload(tokens(slot+20, 700), slot))
    probe = call('/v1/internal/cache-probe', {'prompt': cold})
    assert probe['page_cache']['matched_tokens'] >= 1408, probe['page_cache']
    restored = call('/completion', payload(cold, 0))
    assert restored['timings']['cache_n'] >= 1408, restored['timings']
    assert restored['tokens'] == result['tokens'], 'capacity reclamation changed output'
    after = call('/v1/internal/cache-probe', {'prompt': cold})
    assert after['page_cache']['failures'] == before['page_cache']['failures']
    return {'cold': result['timings'], 'restored': restored['timings'], 'cache_failures': after['page_cache']['failures']}


def active_restore_case():
    cached=tokens(1000,1500)
    holders=[]
    signals=[threading.Event(),threading.Event()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for index in range(2):
            body=payload(tokens(5000+index,1800),index,2)
            body['n_predict']=200
            holders.append(pool.submit(stream,body,[],signals[index]))
        assert all(signal.wait(60) for signal in signals)
        before=call('/v1/internal/cache-probe', {'prompt':cached})
        assert before['page_cache']['matched_tokens'] >= 1408
        result=call('/completion',payload(cached,2))
        assert result['timings']['cache_n'] >= 1408, result['timings']
        assert all(holder.result() is not None for holder in holders)
    after=call('/v1/internal/cache-probe', {'prompt':cached})
    assert after['page_cache']['failures'] == before['page_cache']['failures']
    return result['timings']


def prefix_case(cancel=False):
    head = tokens(2000+int(cancel), 1536)
    leader_events, follower_events, unrelated_events = [], [], []
    first = threading.Event()
    stopped = threading.Event()
    follower_body = payload(head+[111]*8, 1, 3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        leader = pool.submit(stream, payload(head+[112]*8, 0, 0), leader_events, first, stopped if cancel else None)
        deadline = time.monotonic()+60
        while not first.wait(.05):
            if leader.done():
                try: leader.result()
                except urllib.error.HTTPError as exc: raise AssertionError(exc.read().decode()) from exc
                raise AssertionError('leader finished without prefill progress')
            assert time.monotonic() < deadline, 'leader did not prefill'
        follower = pool.submit(stream, follower_body, follower_events)
        deadline = time.monotonic()+60
        while not any(e.get('prompt_progress', {}).get('waiting_for_prefix') for e in follower_events):
            assert not follower.done(), 'follower did not wait for shared prefix'
            assert time.monotonic() < deadline
            time.sleep(.005)
        independent = pool.submit(stream, payload(tokens(3000+int(cancel), 32), 2, 2), unrelated_events)
        if cancel:
            stopped.set()
            leader.result()  # Closing the producer reader cancels its exact native task.
        result = follower.result()
        other = independent.result()
        if not cancel: assert leader.result() is not None
        else:
            try: leader.result()
            except (TimeoutError, OSError): pass
    assert result and other
    if not cancel:
        assert other['_observed_at'] < result['_observed_at'], 'unrelated work was blocked by the prefix follower'
    assert result['timings']['cache_n'] >= (128 if cancel else 1536), result['timings']
    repeat = call('/completion', follower_body)
    assert repeat['tokens'] == result['_tokens'], ('prefix handoff changed output', repeat['tokens'], result['_tokens'])
    assert any(e.get('prompt_progress', {}).get('waiting_for_prefix') for e in follower_events)
    return {'cancelled_producer': cancel, 'follower': result['timings'], 'unrelated': other['timings']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--capacity-only', action='store_true')
    parser.add_argument('--restore-only', action='store_true')
    args=parser.parse_args()
    assert 'build-cpu/bin/llama-server' in args.binary
    assert 'glm53-concurrency-fixture' in args.fixture
    env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_SERVER_PAGE_CACHE','LLAMA_ARG','LD_LIBRARY_PATH'))}
    env.update(HIP_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='', LLAMA_GLM5NEXT_EXECUTION_LANES='6',
        LLAMA_GLM5NEXT_PREFILL_WORKERS='2', LLAMA_GLM5NEXT_WORKER_BATCHES='512,512,1,1,1,1')
    with open(args.log, 'w') as log:
        proc=subprocess.Popen([args.binary,'-m',args.fixture,'--host','127.0.0.1','--port','18088',
            '-ngl','0','--override-kv','glm5next.context_length=int:4096','-c','4096','-np','6','--kv-unified','--kv-unified-per-slot','2048',
            '-b','8192','-ub','16','-t','2','-tb','2','--cache-page-ram','64','--cache-page-tokens','128',
            '--no-cache-idle-slots','--cache-ram','0','--no-warmup','--flash-attn','on'],env=env,stdout=log,stderr=log)
        try:
            for _ in range(240):
                assert proc.poll() is None, 'fixture exited'
                try:
                    if call('/health').get('status')=='ok': break
                except OSError: pass
                time.sleep(.25)
            else: raise AssertionError('fixture readiness timeout')
            print(json.dumps({'capacity':capacity_case(args.restore_only)}), flush=True)
            if not args.capacity_only and not args.restore_only:
                print(json.dumps({'active_restore':active_restore_case()}), flush=True)
                print(json.dumps({'prefix':prefix_case()}), flush=True)
                print(json.dumps({'successor':prefix_case(cancel=True)}), flush=True)
        finally:
            proc.terminate()
            try: proc.wait(15)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()

if __name__ == '__main__': main()

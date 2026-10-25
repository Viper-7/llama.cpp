#!/usr/bin/env python3
"""CPU-only six/eight slot equivalence, sharing, parking and independent waves."""
import argparse,concurrent.futures,json,os,subprocess,threading,time
from pathlib import Path
from test_glm53_live_concurrency import call,request,cancellation_isolation,chat_registry_isolation

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binary',required=True);p.add_argument('--fixture',required=True)
    p.add_argument('--slots',type=int,choices=(3,6,8),required=True);p.add_argument('--log',required=True)
    p.add_argument('--heterogeneous', action='store_true')
    p.add_argument('--require-worker-profile', action='store_true')
    p.add_argument('--require-transport-profile', action='store_true')
    p.add_argument('--require-route-buffer', action='store_true')
    p.add_argument('--owner-ubatch', type=int, choices=(16,64,128,256,512), default=512)
    p.add_argument('--worker-batches')
    p.add_argument('--logical-batch',type=int,default=1024)
    a=p.parse_args()
    capacities = [int(x) for x in a.worker_batches.split(',')] if a.worker_batches else [256,256]+[1]*(a.slots-2)
    if a.worker_batches:
        assert a.heterogeneous and len(capacities)==a.slots
    assert 'build-cpu/bin/llama-server' in a.binary and 'glm53-concurrency-fixture' in a.fixture
    env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_SERVER_PAGE_CACHE','LLAMA_ARG','LD_LIBRARY_PATH'))}
    env.update(HIP_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',LLAMA_GLM5NEXT_EXECUTION_LANES=str(a.slots))
    if a.require_transport_profile:
        for key in ('GGML_SCHED_TRANSPORT_STATS','GGML_SCHED_TRANSPORT_TENSORS','GGML_SCHED_DEVICE_TIMING'):
            env.pop(key,None)
    if a.heterogeneous:
        env.update(LLAMA_GLM5NEXT_PREFILL_WORKERS='2', LLAMA_GLM5NEXT_PREFILL_BATCH='256')
    if a.worker_batches:
        env['LLAMA_GLM5NEXT_WORKER_BATCHES']=a.worker_batches
    if a.require_route_buffer:
        route_dir=Path(a.log+'.routes');route_dir.mkdir(exist_ok=False)
        env.update(LLAMA_GLM5NEXT_ROUTE_STATS='1', LLAMA_GLM5NEXT_ROUTE_STATS_DIR=str(route_dir))
    with open(a.log,'w') as log:
        proc=subprocess.Popen([a.binary,'-m',a.fixture,'--host','127.0.0.1','--port','18087',
            '-ngl','0','--override-kv','glm5next.context_length=int:4096','-c','4096','-np',str(a.slots),'--kv-unified','--kv-unified-per-slot','2048',
            '-b',str(a.logical_batch),'-ub',str(a.owner_ubatch) if a.heterogeneous else '64','-t','2','-tb','2','--cache-page-ram','64','--cache-page-tokens','128',
            '--no-warmup','--flash-attn','on','--chat-template',"{{ messages[0]['content'] }}"],
            env=env,stdout=log,stderr=log)
        try:
            for _ in range(120):
                if proc.poll() is not None:raise RuntimeError('Fixture exited')
                try:
                    if call('/health').get('status')=='ok':break
                except OSError:pass
                time.sleep(.25)
            else:raise RuntimeError('Fixture readiness timeout')
            if a.heterogeneous:
                def cold(slot, index):
                    return call('/completion', {
                        'prompt': [1+(i+index*17)%100 for i in range(480)],
                        'id_slot': slot, 'n_predict': 128, 'temperature': 0, 'seed': 42,
                        'ignore_eos': True, 'return_tokens': True, 'cache_prompt': False,
                        'cache_control': {'lifecycle': 'one_shot'},
                        'serverless_execution_priority': index % 4})
                cold_refs = [cold(0, i) for i in range(a.slots)]
                gate = threading.Barrier(a.slots)
                def launch(i):
                    gate.wait()
                    return cold(i, i)
                assignments = {i: set() for i in range(a.slots)}
                waiting = False
                with concurrent.futures.ThreadPoolExecutor(max_workers=a.slots) as pool:
                    futures = [pool.submit(launch, i) for i in range(a.slots)]
                    while not all(f.done() for f in futures):
                        for slot in call('/slots'):
                            if not slot['is_processing']:
                                continue
                            worker = slot.get('execution_worker', -1)
                            if worker >= 0:
                                assignments[slot['id']].add(worker)
                            waiting |= slot.get('waiting_for_prefill_workspace', False)
                            assert slot['execution_priority'] == slot['id'] % 4
                        time.sleep(.005)
                    cold_actual = [f.result() for f in futures]
                assert all(x['tokens'] == y['tokens'] for x,y in zip(cold_refs,cold_actual)), 'worker reassignment changed cold output'
                assert any(len(workers)>1 for workers in assignments.values()), assignments
                assert waiting, 'No waiting prefill workspace observed'
                print(json.dumps({'worker_reassignment_exact': True, 'waiting_observed': waiting,
                    'workers_by_slot': {i: sorted(v) for i,v in assignments.items()}}), flush=True)
            refs=[request(0,i*13,length=350,predict=800) for i in range(a.slots)]
            barrier=threading.Barrier(a.slots)
            def run(i):
                barrier.wait()
                return request(i,i*13,length=350,predict=800)
            observed=0;parked=False
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.slots) as pool:
                futures=[pool.submit(run,i) for i in range(a.slots)]
                while not all(f.done() for f in futures):
                    slots=call('/slots')
                    observed=max(observed,sum(s['is_processing'] and not s.get('capacity_parked') for s in slots))
                    parked |= any(s.get('capacity_parked') for s in slots)
                    time.sleep(.02)
                actual=[f.result() for f in futures]
            assert all(x['tokens']==y['tokens'] for x,y in zip(refs,actual)), 'expanded park/resume changed tokens'
            assert (parked or a.slots == 3) and observed==a.slots,(parked,observed)
            prefix=[1+i%100 for i in range(512)]
            def shared(slot,tail):
                return call('/completion',{'prompt':prefix+[110+tail]*8,'id_slot':slot,'n_predict':512,
                    'temperature':0,'seed':42,'ignore_eos':True,'return_tokens':True,'cache_prompt':True})
            refs=[shared(0,i) for i in range(a.slots)]
            time.sleep(2.05) # start a fresh, bounded occupancy sampling interval
            shared_gate=threading.Barrier(a.slots)
            def shared_run(i):
                shared_gate.wait()
                return shared(i,i)
            occupancy=None
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.slots) as pool:
                futures=[pool.submit(shared_run,i) for i in range(a.slots)]
                while not all(f.done() for f in futures):
                    sample=call('/v1/internal/cache-probe',{'prompt':[1]})['execution_memory']['kv_occupancy']
                    if (sample['active_sequences']>1 and sample['active_logical_cells']-sample['active_physical_cells']>=512):
                        occupancy=sample
                    time.sleep(.02)
                actual=[f.result() for f in futures]
            assert all(x['tokens']==y['tokens'] for x,y in zip(refs,actual)), 'shared head changed tokens'
            assert all(x['timings']['cache_n']>=512 for x in actual), 'shared head did not reuse'
            assert occupancy is not None,'No same-snapshot shared occupancy proof'
            assert occupancy['active_sequences']>1,occupancy
            assert occupancy['active_logical_cells']-occupancy['active_physical_cells']>=512,occupancy
            assert occupancy['physical_cells']<=4096,occupancy
            cancellation_isolation()
            chat_registry_isolation()
            probe=call('/v1/internal/cache-probe',{'prompt':[1]})
            timing=probe['execution_timing'];memory=probe['execution_memory']
            assert timing['independent_contexts']==a.slots
            if a.worker_batches:
                assert timing['execution_prompt_handoffs_while_prefill_pending'] > 0, timing
            assert len(timing['execution_waves_by_lanes'])==8
            assert timing['execution_waves_by_lanes'][a.slots-1]>0,timing
            if a.require_worker_profile:
                profiles=timing['worker_profiles']
                assert len(profiles)==a.slots
                for i,row in enumerate(profiles):
                    assert row['worker']==i and row['calls']>0,row
                    for key in ('planning_wait_us','planning_held_us','graph_submit_us',
                                'output_us','synchronize_us','input_us','graph_prepare_us'):
                        assert isinstance(row[key],int) and row[key]>=0,row
                    assert row['planning_held_us']>=row['input_us']+row['graph_prepare_us'],row
                later=call('/v1/internal/cache-probe',{'prompt':[1]})['execution_timing']['worker_profiles']
                assert profiles==later,'Idle profiling read mutated counters'
            if a.require_transport_profile:
                for row in timing['worker_profiles']:
                    for key in ('transport_dispatches','transport_copies','transport_bytes',
                                'transport_copy_submit_us','transport_dependency_submit_us',
                                'transport_blocking_wait_us','transport_sync_copy_us',
                                'transport_compute_submit_us'):
                        assert isinstance(row[key],int) and row[key]>=0,row
                    assert row['transport_dispatches']>0 and row['transport_compute_submit_us']>0,row
                    ready=[key for key in row if key.startswith('transport_ready_wait_') and key.endswith('_calls')]
                    assert ready, 'Per-backend readiness counters absent'
                    for key in ready:
                        assert isinstance(row[key],int) and row[key]>=0,row
                        assert isinstance(row[key[:-6]+'_us'],int) and row[key[:-6]+'_us']>=0,row
                assert 'ggml_sched_transport:' not in Path(a.log).read_text(), 'Numeric profiling emitted detailed logs'
            if a.require_route_buffer:
                for item in [memory,*memory['lanes']]:
                    assert item['route_capture_host']=={'buffer_type':'CPU','bytes':16*1024*1024},item['route_capture_host']
                snapshots=[json.loads(p.read_text()) for p in route_dir.glob('routes-*.json')]
                assert len(snapshots)==a.slots+1,len(snapshots)
                assert sum(r['routed_layer_rows'] for r in snapshots)>0
                assert all(r['dropped_layer_rows']==0 for r in snapshots)
            assert len(memory['lanes'])==a.slots
            if a.heterogeneous:
                assert [lane['batch_capacity'] for lane in memory['lanes']]==capacities
            assert not any(s['is_processing'] for s in call('/slots'))
            print(json.dumps({'slots':a.slots,'heterogeneous':a.heterogeneous,'owner_ubatch':a.owner_ubatch,'exact_serial_equivalence':True,'max_processing':observed,
                'capacity_parked':parked,'shared_head_reused':True,'shared_physical_cells':occupancy['active_physical_cells'],
                'shared_logical_cells':occupancy['active_logical_cells'],'shared_active_sequences':occupancy['active_sequences'],'cancellation_isolated':True,
                'execution_timing':timing}),flush=True)
        finally:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
if __name__=='__main__':main()

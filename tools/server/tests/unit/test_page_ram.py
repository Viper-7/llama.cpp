"""CPU-only RAM checkpoint contract tests; never load a model or allocate GPU state."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[4]


def test_ram_pages_crash_reopen_and_gc():
    if not Path('/dev/shm').is_dir() or not shutil.which('g++'):
        pytest.skip('Linux tmpfs and C++ compiler required')
    with tempfile.TemporaryDirectory(prefix='llama-ram-test-') as temporary:
        binary = Path(temporary) / 'test'
        subprocess.run(['g++', '-std=c++17', '-O1', '-pthread', '-I' + str(ROOT),
                        str(Path(__file__).with_name('test-page-ram.cpp')), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=30)


def test_ram_graph_parent_restore_and_branch_release():
    libraries = os.environ.get('LLAMA_PAGE_RAM_TEST_LIB_DIR')
    if not libraries:
        pytest.skip('Set LLAMA_PAGE_RAM_TEST_LIB_DIR to a CPU build bin directory')
    with tempfile.TemporaryDirectory(prefix='llama-ram-graph-test-') as temporary:
        source = (ROOT / 'tools/server/server-context.cpp').read_text()
        page_class = source[source.index('class server_prompt_page_cache {'):source.index('struct server_slot {')]
        # Access visibility only; run the exact native selection/history code.
        page_class = page_class.replace('private:', 'public:', 1)
        # Compile the real graph implementation with synthetic byte fixtures, without creating a llama context.
        fixture = Path(__file__).with_name('test-page-ram.cpp').read_text()
        test_source = Path(temporary) / 'graph.cpp'
        test_source.write_text('#include "tools/server/server-task.h"\n#include "tools/server/server-common.h"\n'
                               '#include "tools/server/server-page-ram.h"\n#include "tools/server/server-responses-leases.h"\n#include "src/llama-ext.h"\n#include <list>\n#include <unordered_set>\n'
                               + page_class + '\n#define TEST_PAGE_GRAPH\n' + fixture)
        binary = Path(temporary) / 'test'
        subprocess.run(['g++', '-std=c++17', '-O1', '-pthread', '-I' + str(ROOT), '-I' + str(ROOT / 'include'),
                        '-I' + str(ROOT / 'common'), '-I' + str(ROOT / 'ggml/include'),
                        '-I' + str(ROOT / 'tools/mtmd'), '-I' + str(ROOT / 'vendor'), str(test_source),
                        '-L' + libraries, '-Wl,-rpath,' + libraries, '-lllama-common', '-lllama', '-lggml-base',
                        '-lllama-server-impl', '-lmtmd', '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=30)


def test_age_frequency_replacement_policy():
    with tempfile.TemporaryDirectory(prefix='llama-cache-policy-test-') as temporary:
        binary = Path(temporary) / 'policy'
        subprocess.run(['g++', '-std=c++17', '-O1', '-pthread', '-I' + str(ROOT),
                        str(Path(__file__).with_name('test-cache-policy.cpp')), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=10)


def test_request_identity_precedes_live_or_physical_cache_adoption():
    source = (ROOT / 'tools/server/server-context.cpp').read_text()
    start = source.index('bool launch_slot_with_task(server_slot & slot, server_task && task)')
    assigned = source.index('task.params.cache_control.request_id = task.id;', start)
    adopted = source.index('restore_prompt_pages(slot, task);', start)
    assert assigned < adopted
    assert source.count('task.params.cache_control.request_id = task.id;') == 1


def test_responses_request_registry_covers_preparation_and_cancel_races():
    with tempfile.TemporaryDirectory(prefix='llama-responses-control-test-') as temporary:
        binary = Path(temporary) / 'control'
        subprocess.run(['g++', '-std=c++17', '-O1', '-pthread', '-pthread', '-I' + str(ROOT),
                        str(Path(__file__).with_name('test-responses-control.cpp')), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=10)


def test_responses_lease_revision_capacity_and_release_fences():
    with tempfile.TemporaryDirectory(prefix='llama-responses-lease-test-') as temporary:
        binary = Path(temporary) / 'leases'
        subprocess.run(['g++', '-std=c++17', '-O1', '-pthread', '-I' + str(ROOT),
                        str(Path(__file__).with_name('test-responses-leases.cpp')), '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=10)


def test_responses_control_queue_does_not_wake_sleeping_worker():
    libraries = os.environ.get('LLAMA_PAGE_RAM_TEST_LIB_DIR')
    if not libraries:
        pytest.skip('Set LLAMA_PAGE_RAM_TEST_LIB_DIR to a CPU build bin directory')
    with tempfile.TemporaryDirectory(prefix='llama-responses-queue-test-') as temporary:
        binary = Path(temporary) / 'queue'
        subprocess.run(['g++', '-std=c++17', '-O1', '-pthread', '-pthread', '-I' + str(ROOT),
                        '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'common'),
                        '-I' + str(ROOT / 'ggml/include'), '-I' + str(ROOT / 'tools/mtmd'),
                        str(Path(__file__).with_name('test-responses-queue.cpp')),
                        '-L' + libraries, '-Wl,-rpath,' + libraries, '-lllama-server-impl', '-lllama-common', '-lmtmd', '-lggml-base',
                        '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=10)


def test_responses_stream_partial_prefill_format():
    libraries = os.environ.get('LLAMA_PAGE_RAM_TEST_LIB_DIR')
    if not libraries:
        pytest.skip('Set LLAMA_PAGE_RAM_TEST_LIB_DIR to a CPU build bin directory')
    with tempfile.TemporaryDirectory(prefix='llama-responses-stream-test-') as temporary:
        binary = Path(temporary) / 'stream'
        subprocess.run(['g++', '-std=c++17', '-O1', '-pthread', '-I' + str(ROOT),
                        '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'common'),
                        '-I' + str(ROOT / 'ggml/include'), '-I' + str(ROOT / 'tools/mtmd'),
                        str(Path(__file__).with_name('test-responses-stream-result.cpp')),
                        '-L' + libraries, '-Wl,-rpath,' + libraries, '-lllama-server-impl',
                        '-lllama-common', '-lmtmd', '-lggml-base', '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=10)

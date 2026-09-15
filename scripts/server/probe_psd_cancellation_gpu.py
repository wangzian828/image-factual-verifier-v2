"""Check client/deadline cancellation against live vLLM metrics, with no replay."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import httpx

ROOT = Path('/volume/ybo/wza')
RUN = ROOT/'inference/psd-sft2056-safety-20260916'
CODE = ROOT/'training-artifacts/psd-serving-safety-20260916/gateway-v2'
OUT = RUN/'cancellation-probes-v3'
ALIAS = 'ifv-psd-sft2056-safety'


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


async def snapshot(client):
    result = {}
    for port in range(19002, 19006):
        response = await client.get(f'http://127.0.0.1:{port}/metrics')
        response.raise_for_status()
        values = {}
        for line in response.text.splitlines():
            if any(line.startswith(name+'{') for name in [
                    'vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:request_success_total']):
                key, value = line.rsplit(' ', 1)
                values[key] = float(value)
        assert any(k.startswith('vllm:num_requests_running{') for k in values)
        result[str(port)] = values
    return result


def active(snapshot):
    return sum(v for rows in snapshot.values() for k, v in rows.items()
               if k.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{')))


def finished(snapshot, reason=None):
    return sum(v for rows in snapshot.values() for k, v in rows.items()
               if k.startswith('vllm:request_success_total{') and
               (reason is None or f'finished_reason="{reason}"' in k))


async def empty(client, seconds=60):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        result = await snapshot(client)
        if active(result) == 0:
            return result
        await asyncio.sleep(0.5)
    raise RuntimeError('GPU requests did not drain')


async def stop(pid):
    assert pid > 1 and os.getpgid(pid) == pid
    os.killpg(pid, signal.SIGTERM)
    for _ in range(120):
        stat = Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split(') ', 1)[1][0] == 'Z':
            return
        await asyncio.sleep(0.5)
    raise RuntimeError('Owned process did not stop gracefully')


async def check(client, name, port, disconnect):
    before = await empty(client)
    health_before = (await client.get(f'http://127.0.0.1:{port}/health')).json()
    body = {'model': ALIAS, 'messages': [{'role': 'user', 'content': 'List integers in order and keep going.'}],
            'max_tokens': 32768, 'min_tokens': 32768, 'thinking_token_budget': 8192,
            'temperature': 0.7, 'seed': 0, 'request_id': 'ifv-psd-gpu-'+name}
    save(OUT/(name+'-request.json'), body)
    task = asyncio.create_task(client.post(f'http://127.0.0.1:{port}/v1/chat/completions', json=body))
    seen = False
    for _ in range(40):
        if active(await snapshot(client)) == 1:
            seen = True
            break
        if task.done():
            break
        await asyncio.sleep(0.1)
    try:
        assert seen, 'Probe never reached GPU scheduling'
        if disconnect:
            await asyncio.sleep(1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            status = 'client_cancelled'
        else:
            response = await task
            (OUT/(name+'-response.json')).write_bytes(response.content)
            assert response.status_code == 504, response.status_code
            status = response.status_code
        released = await empty(client, seconds=30)
        # Client aborts need not increment request_success_total. Use the actual
        # gateway dispatch counter plus GPU queue transitions to detect replay.
        samples = []
        for _ in range(10):
            after = await snapshot(client)
            samples.append({'time': time.time(), 'metrics': after})
            await asyncio.sleep(0.5)
        await asyncio.sleep(5)
        after = await snapshot(client)
        health = (await client.get(f'http://127.0.0.1:{port}/health')).json()
        save(OUT/(name+'-observed.json'), {'before': before, 'released': released,
                                         'after': after, 'samples': samples, 'gateway': health})
        assert all(x['inflight'] == 0 for x in health['replicas'])
        assert active(after) == 0
        assert health['post_dispatches']-health_before['post_dispatches'] == 1, 'Extra upstream POST'
        result = {'passed': True, 'http_status': status, 'gpu_scheduled': seen,
                  'before': before, 'released': released, 'after': after,
                  'abort_metric_delta': finished(after, 'abort')-finished(before, 'abort'),
                  'post_dispatch_delta': 1, 'gateway': health, 'gateway_before': health_before}
        save(OUT/(name+'-check.json'), result)
        print(name, 'GPU_RELEASED_SINGLE_POST', flush=True)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def main():
    assert (RUN/'generation-probes-v1/summary.json').exists(), 'Generation probes still running'
    OUT.mkdir()
    old = json.loads((RUN/'guard.json').read_text())
    pid = old['pid']
    command = [x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
    assert command == old['command'] and str(RUN/'idle-guard/state.json') in command
    env = dict(x.decode().split('=', 1) for x in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if x)
    save(OUT/'guard-before.json', old)
    # The dedicated deadline gateway changes only its own test timeout.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 19011))
    gateway = None
    paused = False
    try:
        await stop(pid)
        paused = True
        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            await empty(client)
            await check(client, 'client-disconnect', 19001, True)
            probe_env = {**os.environ, 'PYTHONPATH': str(CODE),
                         'QWEN_REPLICA_BACKENDS': 'http://127.0.0.1:19005',
                         'QWEN_REPLICA_MODEL_ID': ALIAS,
                         'PSD_GATEWAY_DEADLINE_SECONDS': '5',
                         'AGENT_LLM_REQUEST_TIMEOUT_SECONDS': '8',
                         'AGENT_STAGE_REQUEST_TIMEOUT_SECONDS': '10',
                         'AGENT_LLM_REQUEST_MAX_RETRIES': '0'}
            args = [str(ROOT/'envs/h20-qwen35-vllm-0181/bin/python'), '-m', 'uvicorn',
                    'scripts.server.psd_qwen_gateway:app', '--app-dir', str(CODE),
                    '--host', '127.0.0.1', '--port', '19011', '--log-level', 'warning']
            with (OUT/'deadline-gateway.log').open('xb') as log:
                gateway = subprocess.Popen(args, cwd=CODE, env=probe_env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(OUT/'deadline-gateway.json', {'pid': gateway.pid, 'command': args})
            for _ in range(60):
                try:
                    response = await client.get('http://127.0.0.1:19011/health')
                    response.raise_for_status()
                    break
                except httpx.HTTPError:
                    await asyncio.sleep(0.5)
            else:
                raise RuntimeError('Dedicated deadline gateway did not start')
            await check(client, 'gateway-deadline', 19011, False)
            save(OUT/'summary.json', {'cancellation_gpu_gate_passed': True,
                                     'psd_canary_passed': False, 'time': time.time()})
    except BaseException as error:
        save(OUT/'failure.json', {'type': type(error).__name__, 'message': str(error), 'time': time.time()})
        raise
    finally:
        try:
            if gateway is not None and gateway.poll() is None:
                await stop(gateway.pid)
        finally:
            if paused:
                with (RUN/'guard.log').open('ab') as log:
                    guard = subprocess.Popen(command, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                save(RUN/'guard.json', {'pid': guard.pid, 'command': command,
                                       'started_at': time.time(), 'replaces': pid})
                print('GUARD_RESTORED', guard.pid, flush=True)


if __name__ == '__main__':
    asyncio.run(main())

# gpu-07 / 8×V100 运维记录

This profile records the verified Jupyter access and hardware facts for the
`wza` account on `gpu-07`. It is an internal operations note and contains no
passwords, API keys, or private keys.

## Verified connection

The connection was verified through the existing SSH control endpoint and
remote Jupyter port forwarding:

```text
SSH control endpoint: 47.104.232.153:2429
SSH user: wza
Remote Jupyter: 127.0.0.1:8307
Local tunnel: 127.0.0.1:8307 -> 127.0.0.1:8307
```

The tunneled Jupyter API returned `TornadoServer/6.4.1`. An authenticated
kernel probe reported:

```text
hostname = gpu-07
user     = wza
```

The project client used for the probe was
`scripts/server/jupyter_remote.py`. The existing `8333` tunnel remains the
`gpu-13` connection; port `8307` is the separate `gpu-07` connection.

## Verified hardware

The live `nvidia-smi` probe reported:

```text
8 × Tesla V100-SXM2-32GB
CUDA memory per GPU: 32768 MiB
```

At the time of the probe, GPUs 0–3 reported zero allocated memory; GPUs 4–7
had existing allocations. This is a point-in-time observation and must not be
treated as an availability reservation.

The supplied V100 benchmark screenshot refers to a separate stack. Do not
transfer its reported CPU, operating-system, card-memory, model, or engine
versions to this host without a live probe.

## Safe-use notes

- Run read-only diagnostics first and do not stop processes owned by other
  users.
- Verify `hostname`, `id -un`, and the selected kernel before project work.
- Keep the port-8307 tunnel separate from the `gpu-13` port-8333 tunnel.
- Do not put credentials in this document or in tracked source files.

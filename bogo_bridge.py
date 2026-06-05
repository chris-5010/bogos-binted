#!/usr/bin/env python3
"""
bogo_bridge.py - connects a local GPU daemon to bogo.swapjs.dev

The GPU binary runs as a persistent daemon (OpenCL/Vulkan/HIP context stays
alive). Each dispatch is a line on stdin, result comes back as JSON on stdout.
Dispatch size auto-tunes to ~1s of GPU work.

Usage:
    python bogo_bridge.py --uuid UUID --code CODE --nick NICK [--binary ./bogo_vk]
"""

import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time

try:
    import websockets
except ImportError:
    print("pip install websockets", file=sys.stderr)
    sys.exit(1)


class StopToken:
    """Signal old lease threads to exit when a new job arrives."""
    def __init__(self): self._stop = False
    def stop(self):     self._stop = True
    def stopped(self):  return self._stop


class GpuDaemon:
    """Persistent GPU subprocess. Lives for the whole session."""

    def __init__(self, binary, work_items):
        self.binary     = binary
        self.work_items = work_items
        self.proc       = None
        self._lock      = threading.Lock()
        self._start()

    def _start(self):
        self.proc = subprocess.Popen(
            [self.binary, '--daemon', '--work-items', str(self.work_items)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, bufsize=1)
        print(f"[daemon] started pid={self.proc.pid}", flush=True)

    def dispatch(self, seed_lo, seed_hi, batch, index_offset):
        with self._lock:
            for attempt in range(2):
                try:
                    self.proc.stdin.write(f"{seed_lo} {seed_hi} {batch} {index_offset}\n")
                    self.proc.stdin.flush()
                    line = self.proc.stdout.readline()
                    if not line:
                        raise EOFError("daemon stdout closed")
                    return json.loads(line.strip())
                except Exception as e:
                    print(f"[daemon] error ({e}), restarting", file=sys.stderr, flush=True)
                    if attempt == 0:
                        try:
                            self.proc.terminate()
                            self.proc.wait(timeout=3)
                        except Exception:
                            pass
                        self._start()
            return None

    def close(self):
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            try: self.proc.terminate()
            except Exception: pass


# calibration results — filled in once at startup
_cal = {'done': False, 'gpu_rate': 250_000_000}

def calibrate(daemon):
    if _cal['done']:
        return
    print("[cal] measuring GPU throughput…", flush=True)
    daemon.dispatch(0x1234, 0x5678, 50_000_000, 0)  # warmup
    t0  = time.perf_counter()
    res = daemon.dispatch(0xCAFE, 0xBABE, 50_000_000, 0)
    dt  = time.perf_counter() - t0
    if res and dt > 0:
        _cal['gpu_rate'] = int(50_000_000 / dt)
        print(f"[cal] GPU: {_cal['gpu_rate']/1e6:.0f}M/s", flush=True)
    else:
        print("[cal] calibration failed, using estimate", flush=True)
    _cal['done'] = True


def _dispatch_size(rate, work_items):
    """Round rate up to a multiple of work_items, minimum 1M."""
    return max(1_000_000, ((int(rate) + work_items - 1) // work_items) * work_items)


async def run(args):
    print(f"[bridge] connecting to {args.url}", flush=True)
    loop   = asyncio.get_event_loop()
    daemon = GpuDaemon(args.binary, args.work_items)
    delay  = 2.0

    try:
        while True:
            try:
                async with websockets.connect(args.url, ping_interval=20, ping_timeout=10) as ws:
                    delay = 2.0

                    await ws.send(json.dumps({
                        'type': 'hello', 'v': 5,
                        'uuid': args.uuid, 'nickname': args.nick, 'code': args.code,
                    }))
                    print("[bridge] sent hello (v=5), waiting for welcome…", flush=True)

                    result_q     = asyncio.Queue()
                    current_seed = [None]
                    active_token = [None]

                    async def send_results():
                        while True:
                            r = await result_q.get()
                            if r is None or r.get('_seed') != current_seed[0]:
                                continue  # lease done or stale result
                            try:
                                await ws.send(json.dumps({
                                    'type':         'result',
                                    'seed':         current_seed[0],
                                    'total_done':   r['total_done'],
                                    'best_correct': r['best_correct'],
                                    'best_arr':     r['best_arr'],
                                    'best_index':   r['best_index'],
                                }))
                            except Exception as e:
                                print(f"[bridge] send error: {e}", file=sys.stderr)

                    sender = asyncio.ensure_future(send_results())

                    try:
                        async for raw in ws:
                            msg   = json.loads(raw)
                            mtype = msg.get('type')

                            if mtype == 'welcome':
                                lt  = msg.get('lifetime_shuffles', 0)
                                atb = msg.get('all_time_best', 0)
                                print(f"[bridge] welcome!  lifetime={lt:,}  all_time_best={atb}/25", flush=True)
                                await loop.run_in_executor(None, calibrate, daemon)

                            elif mtype == 'job':
                                seed_str = msg['seed']
                                count    = msg.get('count') or msg.get('batch_size', 10_000_000)
                                seed64   = int(seed_str) & 0xFFFFFFFFFFFFFFFF
                                seed_lo  = seed64 & 0xFFFFFFFF
                                seed_hi  = (seed64 >> 32) & 0xFFFFFFFF

                                # cancel old lease thread if still running
                                if active_token[0]:
                                    active_token[0].stop()
                                tok = StopToken()
                                active_token[0] = tok
                                current_seed[0] = seed_str
                                print(f"[bridge] job seed={seed_str}  count={count:,}", flush=True)

                                def _lease(seed_str=seed_str, seed_lo=seed_lo,
                                           seed_hi=seed_hi, count=count, tok=tok):
                                    batch  = _dispatch_size(_cal['gpu_rate'], args.work_items)
                                    done   = 0
                                    offset = 0
                                    best   = -1

                                    def put(r):
                                        if r is not None:
                                            r['_seed'] = seed_str
                                        loop.call_soon_threadsafe(result_q.put_nowait, r)

                                    while done < count and not tok.stopped():
                                        b  = min(batch, count - done)
                                        t0 = time.perf_counter()
                                        r  = daemon.dispatch(seed_lo, seed_hi, b, offset)
                                        dt = time.perf_counter() - t0

                                        if dt > 0:
                                            _cal['gpu_rate'] = int(b / dt)
                                            batch = _dispatch_size(b / dt, args.work_items)

                                        offset += b
                                        done   += b

                                        if r and r.get('best_correct', -1) >= 0:
                                            best = max(best, r['best_correct'])
                                            rate = b / dt if dt > 0 else 0
                                            print(f"[lease] gpu  best={best}/25  "
                                                  f"done={done:,}/{count:,}  rate={rate/1e6:.0f}M/s",
                                                  flush=True)
                                            put({'total_done': done,
                                                 'best_correct': r['best_correct'],
                                                 'best_arr':     r['best_arr'],
                                                 'best_index':   r.get('best_index', 0)})
                                        else:
                                            print(f"[lease] dispatch failed  done={done:,}/{count:,}",
                                                  file=sys.stderr, flush=True)

                                        if args.throttle > 0:
                                            time.sleep(args.throttle)

                                    put(None)

                                loop.run_in_executor(None, _lease)

                            elif mtype == 'credited':
                                credit   = msg.get('credit', 0)
                                lifetime = msg.get('lifetime_shuffles', 0)
                                sbest    = msg.get('my_session_best', '?')
                                print(f"[bridge] credited +{credit:,}  lifetime={lifetime:,}  session_best={sbest}/25", flush=True)
                                for b in msg.get('new_badges', []):
                                    print(f"[bridge] 🏅 badge: {b.get('name', b.get('id'))}", flush=True)

                            elif mtype == 'stats_tick':
                                total = msg.get('total_shuffles', 0)
                                rate  = msg.get('total_rate', 0)
                                best  = msg.get('all_time_best', 0)
                                conns = msg.get('connections', 0)
                                print(f"[stream] total={total/1e12:.3f}T  rate={rate/1e6:.0f}M/s  best={best}/25  connections={conns}", flush=True)

                            elif mtype == 'rejected':
                                print(f"[bridge] rejected: {msg}", file=sys.stderr)

                            elif mtype == 'banned':
                                print(f"[bridge] BANNED: {msg.get('reason','?')}", file=sys.stderr)
                                daemon.close()
                                sys.exit(1)

                            elif mtype == 'contributions_closed':
                                print("[bridge] contributions closed.", flush=True)
                                daemon.close()
                                sys.exit(0)

                            elif mtype == 'ping':
                                await ws.send(json.dumps({'type': 'pong', 't': msg.get('t')}))

                            elif mtype == 'client_outdated':
                                print(f"[bridge] WARNING: client outdated (server wants v{msg.get('min_version','?')})", file=sys.stderr)

                            else:
                                print(f"[bridge] unknown: {mtype} {msg}", flush=True)

                    finally:
                        sender.cancel()

            except websockets.exceptions.InvalidStatus as e:
                print(f"[bridge] HTTP {e.response.status_code}, retrying in {delay:.0f}s…", file=sys.stderr, flush=True)
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 60.0)

            except (websockets.exceptions.ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                print(f"[bridge] disconnected ({e}), retrying in {delay:.0f}s…", file=sys.stderr, flush=True)
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 60.0)

    finally:
        daemon.close()


def main():
    p = argparse.ArgumentParser(description="bogosort bridge for bogo.swapjs.dev")
    p.add_argument("--uuid",       required=True)
    p.add_argument("--code",       required=True)
    p.add_argument("--nick",       required=True)
    p.add_argument("--binary",     default="./bogo_gpu")
    p.add_argument("--url",        default="wss://bogo.swapjs.dev/ws")
    p.add_argument("--work-items", type=int, default=262144)
    p.add_argument("--throttle",   type=float, default=0.0,
                   help="sleep between dispatches (seconds)")
    args = p.parse_args()

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[bridge] stopped.", flush=True)


if __name__ == "__main__":
    main()

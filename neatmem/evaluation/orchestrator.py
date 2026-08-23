"""LOCOMO evaluate orchestrator (`neatmem evaluate`).

The command does three things: assemble env, manage process lifecycle, write
a manifest. All score-relevant variables are env vars.

Parameter philosophy: the orchestrator owns ONLY eval-side params (output dir,
stages, runs, workers, --top-k/--batch-size, dataset, qdrant binary). Any
`neatmem serve` flag is accepted verbatim, validated by the same parser that
serve uses (cli.add_serve_arguments), translated to env via
cli.serve_flags_to_env, and injected into ALL child processes — dedup and
extraction act at ingest time (write side), so forwarding flags to the server
alone would silently not affect the evaluation.

Pipeline phases: qdrant server -> ingest (in-process, whole dataset,
.ingest_done marker) -> neatmem server -> search+answer x runs -> judge x runs
-> score.

Env layering (later wins):
  --env-file (default ./.env) < process env < serve/eval flags
  < orchestrator-forced items (ports, db paths, MESSAGE_BATCHING_ENABLED=false)

Manifest records env in product namespaces only (secrets redacted); the full
env goes to child processes.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import neatmem
from neatmem.cli import add_serve_arguments, serve_flags_to_env

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVAL_DIR / "dataset/locomo10.json"
INGEST_SCRIPT = EVAL_DIR / "locomo/ingest_locomo.py"
SEARCH_SCRIPT = EVAL_DIR / "run_experiments.py"
JUDGE_SCRIPT = EVAL_DIR / "metrics/llm_judge.py"

# Manifest record = product config namespaces only. The child process gets the
# full env (pass-through, zero parameter knowledge), but the record must be
# safe to publish: platform noise varies per machine and a denylist can't
# enumerate it (2026-08-20: autodl panel token almost leaked into a manifest).
# New serve/eval env vars in existing families are recorded automatically.
RECORD_PREFIXES = (
    "NEATMEM_", "LLM_", "OPENAI_", "ANSWER_", "JUDGE_", "EMBEDDING_",
    "SILICONFLOW_", "ENABLE_", "DEDUP_", "EDIT_", "REWRITE_", "EXTRACTION_",
    "RERANK_", "CROSS_ENCODER_", "EXTRACT_", "GRAPH_", "MESSAGE_", "QDRANT_", "HISTORY_",
    "MEMORY_",
)
RECORD_EXACT = {"TOP_K", "BATCH_SIZE", "MAX_WORKERS", "INGEST_CUSTOM_INSTRUCTIONS"}

# Auto-assigned per run; recorded but excluded from the resume mismatch check.
VOLATILE_KEYS = {"NEATMEM_PORT", "NEATMEM_URL", "QDRANT_PORT", "QDRANT_HOST"}

# Rough call volumes for the preflight estimate (MiniMax-class stack, LOCOMO10).
EST_CALLS_INGEST = 12000
EST_CALLS_PER_RUN = 4000  # answer + judge


def die(msg):
    sys.exit(f"evaluate: error: {msg}")


def parse_env_file(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def find_free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def is_recorded(key):
    if not key:
        return False
    return key in RECORD_EXACT or any(key.startswith(p) for p in RECORD_PREFIXES)


def redact(d):
    return {k: ("<redacted>" if re.search(r"KEY|TOKEN|SECRET|PASSWORD", k, re.I)
                else v) for k, v in sorted(d.items())}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_env(args, flag_env, forced):
    """Layer: --env-file < process env < flags < forced.
    Returns (record_env, child_env): record_env is merged minus ambient noise,
    for the manifest; child_env is the full env for subprocesses."""
    merged = {}
    if args.env_file:
        env_file = Path(args.env_file)
        if not env_file.exists():
            die(f"--env-file not found: {args.env_file}")
        merged.update(parse_env_file(env_file))
    elif Path(".env").exists():
        merged.update(parse_env_file(".env"))
    merged.update(dict(os.environ))
    merged.update(flag_env)
    merged.update(forced)
    if not merged.get("OPENAI_API_KEY"):
        die("OPENAI_API_KEY not set: export it, or put it in ./.env, "
            "or pass --env-file <path>")
    record = {k: v for k, v in merged.items() if is_recorded(k)}
    return record, merged


def wait_health(url, patterns, proc, log_path, tries=40, interval=3, label="server"):
    import urllib.request
    for _ in range(tries):
        try:
            body = urllib.request.urlopen(url, timeout=5).read().decode()
            if any(p in body for p in patterns):
                return
        except Exception:
            pass
        if proc.poll() is not None:
            tail = Path(log_path).read_text(errors="replace")[-3000:]
            die(f"{label} died during startup:\n{tail}")
        time.sleep(interval)
    tail = Path(log_path).read_text(errors="replace")[-3000:]
    die(f"{label} health check timeout:\n{tail}")


def spawn(cmd, env, log_path, cwd):
    """Start a child in its own process group so stop() can kill the whole
    tree (2026-08-21: orphaned qdrant held the WAL lock and panicked the next
    arm's qdrant on the same storage)."""
    log = open(log_path, "ab")
    return subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)


def start_qdrant(args, sdir, qport, grpc_port, storage_path):
    cfg = sdir / "qdrant_config.yaml"
    cfg.write_text(
        f"service:\n  http_port: {qport}\n  grpc_port: {grpc_port}\n"
        f"storage:\n  storage_path: {storage_path}\ntelemetry_disabled: true\n"
    )
    proc = spawn([args.qdrant_bin, "--config-path", str(cfg)],
                 None, sdir / "logs/qdrant.log", sdir)
    wait_health(f"http://localhost:{qport}/healthz", ["ok", "passed", "healthy"],
                proc, sdir / "logs/qdrant.log", tries=20, interval=2, label="qdrant")
    return proc


def start_server(child_env, port, qport, db_dir, log_path, cwd):
    """Start via `neatmem serve`. All configuration flows through child_env
    (translated serve flags included); only orchestrator-forced flags are
    passed on the command line."""
    cmd = [sys.executable, "-m", "neatmem.cli", "serve",
           "--host", "127.0.0.1", "--port", str(port),
           "--vector-db-url", f"http://localhost:{qport}",
           "--history-db-path", str(db_dir / "history.db")]
    proc = spawn(cmd, child_env, log_path, cwd)
    wait_health(f"http://localhost:{port}/health", ["ok", "healthy", "true"],
                proc, log_path, label="neatmem server")
    return proc


def stop(proc):
    """SIGTERM the child's whole process group, then SIGKILL on timeout."""
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


def run_logged(cmd, env, log_path, cwd):
    proc = spawn(cmd, env, log_path, cwd)
    rc = proc.wait()
    if rc != 0:
        tail = Path(log_path).read_text(errors="replace")[-3000:]
        die(f"command failed rc={rc}: {shlex.join(cmd)}\n{tail}")


def slice_dataset(dataset, limit):
    data = json.loads(Path(dataset).read_text(encoding="utf-8"))
    if limit:
        data = data[:limit]
    return data


def ghost_check(sdir, expected_writes):
    """memory_history.db ADD count must ~= ingest-log writes (0816 ghost incident)."""
    db = sdir / "db/memory_history.db"
    if not db.exists():
        die(f"ghost check: {db} missing after ingest")
    conn = sqlite3.connect(db)
    add_count = conn.execute(
        "SELECT COUNT(*) FROM history WHERE event = 'ADD'").fetchone()[0]
    conn.close()
    if expected_writes == 0:
        die("ghost check: ingest logs show zero writes (log format changed?)")
    if abs(add_count - expected_writes) / expected_writes > 0.05:
        die(f"ghost check FAILED: history ADD={add_count} vs ingest writes="
            f"{expected_writes} (>5% diff, possible ghost writes)")


def results_conversation_count(results_file):
    data = json.loads(Path(results_file).read_text(encoding="utf-8"))
    return len(data)


def score_run(judged_file, out_txt):
    data = json.loads(Path(judged_file).read_text(encoding="utf-8"))
    from collections import defaultdict
    total = correct = 0
    cat = defaultdict(lambda: [0, 0])
    for _, qas in data.items():
        for x in qas:
            c = int(x.get("category", -1))
            if c == 5:
                continue
            total += 1
            ok = str(x.get("llm_label", "")).upper().strip() in ("1", "CORRECT", "TRUE", "YES")
            correct += ok
            cat[c][1] += 1
            cat[c][0] += ok
    lines = [f"Run score: {correct}/{total} = {correct/total:.4f}"]
    for k in sorted(cat):
        lines.append(f"  C{k}: {cat[k][0]}/{cat[k][1]} = {cat[k][0]/cat[k][1]:.4f}")
    Path(out_txt).write_text("\n".join(lines) + "\n")
    return correct / total


def search_rerank_arg(record_env):
    """The search script's --rerank is a per-request override: hardcoding
    "on" would defeat a RERANK_MODE=off arm (server default overridden per
    request). Follow the effective RERANK_MODE env instead."""
    v = (record_env.get("RERANK_MODE") or "llm").strip().lower()
    return "off" if v == "off" else "on"


def judge_env(child_env):
    """JUDGE_* namespace maps to what llm_judge.py reads (OPENAI_*, LLM_MODEL)."""
    env = dict(child_env)
    if "JUDGE_API_KEY" in child_env:
        env["OPENAI_API_KEY"] = child_env["JUDGE_API_KEY"]
    if "JUDGE_BASE_URL" in child_env:
        env["OPENAI_BASE_URL"] = child_env["JUDGE_BASE_URL"]
    if "JUDGE_MODEL" in child_env:
        env["LLM_MODEL"] = child_env["JUDGE_MODEL"]
    return env


def provenance():
    """Package version, plus git sha when running from a checkout."""
    prov = {"neatmem_version": neatmem.__version__, "git_sha": None}
    root = Path(neatmem.__file__).resolve().parent
    for parent in (root, *root.parents):
        if (parent / ".git").exists():
            r = subprocess.run(["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True)
            prov["git_sha"] = r.stdout.strip() or None
            break
    return prov


def load_or_validate_manifest(sdir, name, record_env, serve_args, dataset, args):
    mpath = sdir / "manifest.json"
    record = {
        "strategy": name,
        **provenance(),
        "dataset": {"path": str(dataset), "sha256": sha256_file(dataset)},
        "models": {
            "llm": record_env.get("LLM_MODEL"),
            "embedding": record_env.get("EMBEDDING_MODEL"),
            "answer": record_env.get("ANSWER_MODEL") or record_env.get("LLM_MODEL"),
            "judge": record_env.get("JUDGE_MODEL") or record_env.get("LLM_MODEL"),
        },
        "effective_env": redact(record_env),
        "serve_args": serve_args,
        "reuse_db": bool(args.reuse_db),
        "stages": {},
    }
    if mpath.exists() and not args.force:
        old = json.loads(mpath.read_text())
        if old.get("neatmem_version") != record["neatmem_version"]:
            die(f"resume check: neatmem version changed ({old.get('neatmem_version')} -> "
                f"{record['neatmem_version']}); use a new --output-dir")
        if old.get("git_sha") and record["git_sha"] and old.get("git_sha") != record["git_sha"]:
            die(f"resume check: git sha changed ({old.get('git_sha')} -> "
                f"{record['git_sha']}); use a new --output-dir")
        stable = lambda d: {k: v for k, v in d.get("effective_env", {}).items()
                            if k not in VOLATILE_KEYS}
        if stable(old) != stable(record) or old.get("serve_args") != serve_args:
            diff = {k for k in set(stable(old)) | set(stable(record))
                    if stable(old).get(k) != stable(record).get(k)}
            die(f"resume check: config changed (env keys {sorted(diff)}, "
                f"serve_args {old.get('serve_args')} -> {serve_args}); "
                f"use a new --output-dir")
        return old, mpath
    mpath.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record, mpath


def run_strategy(args, flag_env, dataset_for_stages):
    """One run = one strategy, fully described by env (--env-file/process
    env) + serve/eval flags. No bundled config layer (2026-08-23: --config
    and the packaged strategy .env files removed)."""
    name = "custom"
    sdir = Path(args.output_dir).resolve()
    (sdir / "logs").mkdir(parents=True, exist_ok=True)
    (sdir / "results/judge").mkdir(parents=True, exist_ok=True)

    if args.reuse_db:
        db_dir = Path(args.reuse_db).resolve()
        if not (db_dir / "storage").exists():
            die(f"--reuse-db {db_dir} has no storage/ subdir")
        stages = [s for s in args.stages if s != "ingest"]
        if stages != args.stages:
            print(f"[{name}] --reuse-db: ingest stage skipped")
    else:
        db_dir = sdir / "db"
        stages = args.stages
    db_dir.mkdir(parents=True, exist_ok=True)

    port, qport, grpc = find_free_port(), find_free_port(), find_free_port()
    forced = {
        "QDRANT_HOST": "localhost", "QDRANT_PORT": str(qport),
        "QDRANT_PATH": str(db_dir),
        "HISTORY_DB_PATH": str(db_dir / "history.db"),
        "MEMORY_HISTORY_DB_PATH": str(db_dir / "memory_history.db"),
        "NEATMEM_PORT": str(port),
        "NEATMEM_URL": f"http://localhost:{port}",
        "MESSAGE_BATCHING_ENABLED": "false",
    }
    record_env, child = build_env(args, flag_env, forced)
    manifest, mpath = load_or_validate_manifest(sdir, name, record_env,
                                                args.serve_args, args.dataset, args)
    top_k = record_env.get("TOP_K", "20")
    n_convs = len(dataset_for_stages)

    print(f"[{name}] stages={','.join(stages)} port={port} qport={qport} db={db_dir}")
    qdrant = server = None
    t0 = time.time()
    try:
        if "ingest" in stages:
            # Whole-dataset single ingest, .ingest_done marker. Per-conversation
            # slicing is NOT safe: the ingest script derives user_id from
            # enumerate() position (2026-08-20 contamination incident).
            marker = sdir / "db/.ingest_done"
            qdrant = start_qdrant(args, sdir, qport, grpc, db_dir / "storage")
            if marker.exists() and not args.force:
                print(f"[{name}] ingest skipped (marker found)")
            else:
                run_set = sdir / "dataset_slices/ingest_set.json"
                run_set.parent.mkdir(exist_ok=True)
                run_set.write_text(json.dumps(dataset_for_stages, ensure_ascii=False))
                env = dict(child, MAX_WORKERS=str(args.ingest_workers),
                           BATCH_SIZE=record_env.get("BATCH_SIZE", "10"),
                           DATASET=str(run_set))
                log = sdir / "logs/ingest.log"
                run_logged([sys.executable, str(INGEST_SCRIPT)], env, log, sdir)
                text = log.read_text(errors="replace")
                m = re.search(r"Successful: (\d+) / (\d+)", text)
                if not m or m.group(1) != m.group(2):
                    die(f"ingest: tasks not all successful, see {log}")
                writes = sum(int(n) for n in re.findall(r"实际写入 (\d+) 条", text))
                ghost_check(sdir, writes)
                marker.touch()
                print(f"[{name}] ingest done, writes={writes}")
            manifest["stages"]["ingest"] = {"secs": int(time.time() - t0)}
            mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

        if "search" in stages:
            scores = []
            if qdrant is None:
                qdrant = start_qdrant(args, sdir, qport, grpc, db_dir / "storage")
            server = start_server(child, port, qport, db_dir,
                                  sdir / "logs/server.log", sdir)
            t1 = time.time()
            full_set = sdir / "dataset_slices/search_set.json"
            full_set.parent.mkdir(exist_ok=True)
            full_set.write_text(json.dumps(dataset_for_stages, ensure_ascii=False))
            for run in range(1, args.runs + 1):
                out = sdir / f"results/run{run}"
                res_file = out / "neatmem_results.json"
                # Skip only a COMPLETE results file. A truncated file from a
                # killed search parses fine (per-conversation flush) but would
                # silently score a partial run (2026-08-21 orphan incident).
                if res_file.exists() and not args.force:
                    if results_conversation_count(res_file) == n_convs:
                        print(f"[{name}] search run {run} skipped (results exist)")
                        continue
                    print(f"[{name}] search run {run}: truncated results "
                          f"({results_conversation_count(res_file)}/{n_convs} conversations), rerunning")
                out.mkdir(parents=True, exist_ok=True)
                run_logged([sys.executable, str(SEARCH_SCRIPT), "--method", "search",
                            "--dataset", str(full_set), "--output-folder", str(out),
                            "--top-k", top_k, "--rerank", search_rerank_arg(record_env),
                            "--workers", str(args.search_workers)],
                           child, sdir / f"logs/search_run{run}.log", sdir)
                data = json.loads(res_file.read_text())
                nonempty = sum(
                    1 for xs in data.values() for x in xs
                    if x["num_speaker_1_memories"] + x["num_speaker_2_memories"] > 0)
                total = sum(len(xs) for xs in data.values())
                if nonempty / total < 0.9:
                    die(f"search run {run}: retrieval coverage "
                        f"{nonempty}/{total} <90% (user_id/collection mismatch?)")
                print(f"[{name}] search run {run} done "
                      f"(coverage {nonempty}/{total})")
            manifest["stages"]["search"] = {"secs": int(time.time() - t1)}
            mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

        if "judge" in stages:
            scores = []
            t2 = time.time()
            jenv = judge_env(child)
            for run in range(1, args.runs + 1):
                res = sdir / f"results/run{run}/neatmem_results.json"
                judged = sdir / f"results/judge/judged_run{run}.json"
                if not res.exists():
                    die(f"judge run {run}: {res} missing (run search stage first)")
                if judged.exists() and not args.force:
                    if results_conversation_count(judged) == results_conversation_count(res):
                        print(f"[{name}] judge run {run} skipped (judged exists)")
                        continue
                    print(f"[{name}] judge run {run}: truncated judged file, rerunning")
                run_logged([sys.executable, str(JUDGE_SCRIPT),
                            "--input_file", str(res), "--output_file", str(judged),
                            "--workers", str(args.judge_workers)],
                           jenv, sdir / f"logs/judge_run{run}.log", sdir)
                print(f"[{name}] judge run {run} done")
            for run in range(1, args.runs + 1):
                judged = sdir / f"results/judge/judged_run{run}.json"
                if judged.exists():
                    scores.append(score_run(judged, sdir / f"results/score_run{run}.txt"))
            if scores:
                manifest["stages"]["judge"] = {"secs": int(time.time() - t2)}
                manifest["score_mean"] = sum(scores) / len(scores)
                manifest["scores"] = [round(s, 4) for s in scores]
                mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
                print(f"[{name}] scores: {manifest['scores']} "
                      f"mean={manifest['score_mean']:.4f}")
    finally:
        stop(server)
        stop(qdrant)
    print(f"[{name}] complete, wall={int(time.time()-t0)}s")


def preflight(args, flag_env, record_env, dataset_for_stages):
    est = EST_CALLS_INGEST + args.runs * EST_CALLS_PER_RUN
    print("Strategy    : custom (env + flags; no bundled configs since 2026-08-23)")
    print(f"Dataset     : {args.dataset} (sha256:{sha256_file(args.dataset)[:8]}…) "
          f"{len(dataset_for_stages)} conversations"
          + (f" [--limit {args.limit}]" if args.limit else ""))
    print(f"Stages      : {','.join(args.stages)}  runs={args.runs}  "
          f"workers={args.ingest_workers}/{args.search_workers}/{args.judge_workers}")
    if args.serve_args:
        print(f"Serve args  : {shlex.join(args.serve_args)} (translated to env for all stages)")
    print(f"Estimated   : ~{est//1000}k LLM calls (rough)")
    print(f"Output      : {args.output_dir}")
    print(f"  llm={record_env.get('LLM_MODEL')} "
          f"embed={record_env.get('EMBEDDING_MODEL')} "
          f"answer={record_env.get('ANSWER_MODEL') or record_env.get('LLM_MODEL')} "
          f"judge={record_env.get('JUDGE_MODEL') or record_env.get('LLM_MODEL')}")
    print(f"  dedup: enabled={record_env.get('DEDUP_ENABLED')} "
          f"detector={record_env.get('DEDUP_DETECTOR')} "
          f"resolver={record_env.get('DEDUP_RESOLVER')} "
          f"thr={record_env.get('DEDUP_RECALL_THRESHOLD')}")
    rerank_mode = record_env.get('RERANK_MODE', 'llm')
    cands_default = '100' if rerank_mode == 'cross_encoder' else '20'
    cands = (record_env.get('CROSS_ENCODER_CANDS') if rerank_mode == 'cross_encoder'
             else record_env.get('LLM_RERANK_CANDS')) or cands_default
    print(f"  top_k={record_env.get('TOP_K', '20')} "
          f"rerank={rerank_mode} "
          f"cands={cands} "
          f"bm25={record_env.get('ENABLE_BM25')} "
          f"entity={record_env.get('ENABLE_ENTITY')}")
    if args.dry_run:
        for k, v in redact(record_env).items():
            print(f"    {k}={v}")


def build_eval_parser():
    p = argparse.ArgumentParser(
        prog="neatmem evaluate",
        description=__doc__.splitlines()[0],
        epilog="unrecognized args must be valid `neatmem serve` flags; they are "
               "translated to env and applied to ALL stages (ingest included)")
    p.add_argument("--stages", default="ingest,search,judge",
                   help="comma subset of ingest,search,judge")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--limit", type=int, help="first N conversations (smoke)")
    p.add_argument("--force", action="store_true", help="ignore idempotency markers")
    p.add_argument("--output-dir", default="runs/default")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--reuse-db", help="existing arm db dir for scouting "
                                      "(skips ingest; manifest flagged)")
    p.add_argument("--env-file", help="bottom-layer env file (default: ./.env)")
    p.add_argument("--top-k", type=int, help="retrieval top_k (env TOP_K, default 20)")
    p.add_argument("--batch-size", type=int, help="ingest batch size (env BATCH_SIZE, default 10)")
    p.add_argument("--qdrant-bin", help="qdrant server binary path "
                                        "(env QDRANT_BIN, default: PATH lookup)")
    p.add_argument("--ingest-workers", type=int, default=20)
    p.add_argument("--search-workers", type=int, default=16)
    p.add_argument("--judge-workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true",
                   help="print preflight + merged env, do not execute")
    return p


def parse_serve_args(serve_args):
    """Validate passthrough args against the serve flag definitions and
    translate them to env (injected into every pipeline child)."""
    sp = argparse.ArgumentParser(prog="neatmem evaluate (serve flags)")
    add_serve_arguments(sp)
    ns = sp.parse_args(serve_args)
    return serve_flags_to_env(ns)


def run_evaluate(argv):
    if "--config" in argv:
        die("--config was removed (2026-08-23): one run = one strategy, "
            "described by env + serve flags. See the strategy variants table "
            "in https://neatmem.readthedocs.io/en/latest/evaluation/")
    p = build_eval_parser()
    args, serve_args = p.parse_known_args(argv)
    args.serve_args = serve_args
    args = args_normalize(args)
    flag_env = parse_serve_args(serve_args)
    if args.top_k is not None:
        flag_env["TOP_K"] = str(args.top_k)
    if args.batch_size is not None:
        flag_env["BATCH_SIZE"] = str(args.batch_size)

    dataset_for_stages = slice_dataset(args.dataset, args.limit)

    # Preflight builds the merged env once (also fires the OPENAI_API_KEY
    # precheck inside build_env); run_strategy rebuilds with forced ports.
    record_env, _ = build_env(args, flag_env, {})
    preflight(args, flag_env, record_env, dataset_for_stages)
    if args.dry_run:
        return
    run_strategy(args, flag_env, dataset_for_stages)


def args_normalize(args):
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = set(stages) - {"ingest", "search", "judge"}
    if bad:
        die(f"unknown stages: {sorted(bad)}")
    args.stages = stages
    if not Path(args.dataset).exists():
        die(f"dataset not found: {args.dataset}")
    args.qdrant_bin = (args.qdrant_bin or os.environ.get("QDRANT_BIN")
                       or shutil.which("qdrant"))
    if not args.qdrant_bin or not Path(args.qdrant_bin).exists():
        die("qdrant binary not found: pass --qdrant-bin, set QDRANT_BIN, "
            "or put qdrant on PATH")
    for f in (INGEST_SCRIPT, SEARCH_SCRIPT, JUDGE_SCRIPT):
        if not Path(f).exists():
            die(f"required file missing: {f}")
    return args


if __name__ == "__main__":
    run_evaluate(sys.argv[1:])

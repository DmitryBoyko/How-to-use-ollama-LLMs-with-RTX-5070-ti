#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


RECOMMENDED_MODELS: List[Tuple[str, str]] = [
    ("qwen2.5:7b-instruct-q4_K_M", "general"),
    ("qwen2.5:14b-instruct-q4_K_M", "general"),
    ("mistral:7b-instruct-q4_K_M", "general"),
    ("mistral:7b-instruct-v0.3-q4_K_M", "general"),
    ("llama3.1:8b-instruct-q4_K_M", "general"),
    ("gemma2:9b-instruct-q4_K_M", "general"),
    ("gemma3:12b-it-q4_K_M", "general"),
    ("phi4:14b-q4_K_M", "general"),
    ("qwen2.5-coder:7b-instruct-q4_K_M", "coder"),
    ("qwen2.5-coder:14b-instruct-q4_K_M", "coder"),
]

SYSTEM_RU_ONLY = (
    "Ты — полезный ассистент.\n"
    "Отвечай строго на русском языке. Запрещено использовать любые другие языки, иероглифы и латиницу.\n"
    "Если вопрос неясен — задай уточняющий вопрос по-русски."
)


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size_bytes: Optional[int] = None
    quant: Optional[str] = None
    params: Optional[str] = None
    family: Optional[str] = None
    modified_at: Optional[str] = None


def eprint(*args: object, end: str = "\n") -> None:
    print(*args, file=sys.stderr, end=end, flush=True)


def http_json(method: str, url: str, payload: Optional[dict], timeout_s: int) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def http_ndjson(method: str, url: str, payload: dict, timeout_s: int) -> Iterable[dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout_s)
    try:
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line.decode("utf-8"))
            except Exception:
                continue
    finally:
        resp.close()


def api_base(host_url: str) -> str:
    return host_url.rstrip("/")


def api_tags(host_url: str) -> Dict[str, ModelInfo]:
    base = api_base(host_url)
    resp = http_json("GET", f"{base}/api/tags", None, timeout_s=30)
    out: Dict[str, ModelInfo] = {}
    for m in resp.get("models", []) or []:
        name = m.get("name") or m.get("model")
        if not name:
            continue
        details = m.get("details") or {}
        out[name] = ModelInfo(
            name=name,
            size_bytes=m.get("size"),
            quant=details.get("quantization_level"),
            params=details.get("parameter_size"),
            family=details.get("family"),
            modified_at=m.get("modified_at"),
        )
    return out


def api_pull(host_url: str, model: str) -> None:
    base = api_base(host_url)
    eprint(f"Pulling model: {model}")
    last_status = ""
    for ev in http_ndjson("POST", f"{base}/api/pull", {"model": model, "stream": True}, timeout_s=3600):
        status = str(ev.get("status") or "")
        if status:
            last_status = status
        total = ev.get("total")
        completed = ev.get("completed")
        pct = None
        if isinstance(total, int) and isinstance(completed, int) and total > 0:
            pct = int((completed / total) * 100)
        if pct is not None:
            eprint(f"\r{status:30s} {pct:3d}% ", end="")  # stderr only
        else:
            eprint(f"\r{status:30s}     ", end="")
        if status == "success":
            break
    eprint("")
    if last_status and last_status != "success":
        raise RuntimeError(f"pull failed (last status: {last_status})")


def api_generate_stream(
    host_url: str,
    model: str,
    prompt: str,
    num_predict: Optional[int] = None,
    system: str = SYSTEM_RU_ONLY,
) -> Iterable[dict]:
    base = api_base(host_url)
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "options": {"temperature": 0.2},
    }
    if num_predict is not None:
        payload["options"]["num_predict"] = int(num_predict)
    return http_ndjson("POST", f"{base}/api/generate", payload, timeout_s=3600)


def api_generate_once(host_url: str, model: str, prompt: str, num_predict: int = 1, system: str = SYSTEM_RU_ONLY) -> None:
    base = api_base(host_url)
    payload = {"model": model, "prompt": prompt, "system": system, "stream": False, "options": {"num_predict": int(num_predict)}}
    try:
        http_json("POST", f"{base}/api/generate", payload, timeout_s=180)
    except Exception:
        pass


def api_delete(host_url: str, model: str) -> None:
    base = api_base(host_url)
    try:
        http_json("POST", f"{base}/api/delete", {"model": model}, timeout_s=60)
        return
    except Exception:
        # Fallback to DELETE-with-body (older servers) – urllib doesn't support body on DELETE cleanly,
        # so we fallback to container CLI below.
        pass
    # Fallback: use ollama rm inside container (requires docker compose)
    run_cmd(["docker", "compose", "exec", "-T", "ollama", "ollama", "rm", model], check=True)


def run_cmd(cmd: List[str], check: bool) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=check)


def gpu_util_percent(gpu_index: int) -> Optional[int]:
    # Prefer host nvidia-smi if present
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = run_cmd(
                [smi, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                check=True,
            ).stdout.strip()
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            if gpu_index >= len(lines):
                return None
            v = lines[gpu_index]
            return int(v) if v.isdigit() else None
        except Exception:
            pass

    # Fallback: ask inside gpu-metrics container
    try:
        out = run_cmd(
            ["docker", "exec", "gpu-metrics", "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            check=True,
        ).stdout.strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if gpu_index >= len(lines):
            return None
        v = lines[gpu_index]
        return int(v) if v.isdigit() else None
    except Exception:
        return None


class GpuTicker:
    def __init__(self, gpu_index: int, interval_s: float = 0.5) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.is_set():
                util = gpu_util_percent(self.gpu_index)
                if util is not None:
                    eprint(f"\rGPU {self.gpu_index}: {util:3d}%  ", end="")
                time.sleep(self.interval_s)
            eprint("\r" + (" " * 20) + "\r", end="")

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t:
            t.join(timeout=2)


def ensure_model_ready(host_url: str, model: str, require_gpu_only: bool) -> None:
    installed = api_tags(host_url)
    if model not in installed:
        api_pull(host_url, model)

    # warmup
    eprint(f"Warming up: {model}")
    api_generate_once(host_url, model, " ", num_predict=1)

    if require_gpu_only:
        # Parse `ollama ps` from container and require 100% GPU
        try:
            out = run_cmd(["docker", "compose", "exec", "-T", "ollama", "ollama", "ps"], check=True).stdout
            ok = False
            for line in out.splitlines():
                if model in line and "100% GPU" in line:
                    ok = True
                    break
            if not ok:
                raise RuntimeError("not 100% GPU")
        except Exception:
            raise RuntimeError("GPU-only check failed (model not shown as '100% GPU' in `ollama ps`). Choose a smaller model.")


def choose_model_interactive(host_url: str, current: str) -> str:
    installed = api_tags(host_url)
    print("")
    print(f"Выбери модель (номер). Текущая: {current}")
    for i, (name, kind) in enumerate(RECOMMENDED_MODELS, start=1):
        flag = "installed" if name in installed else "not installed"
        print(f"{i:2d}) {name} [{kind}, {flag}]")
    print(" 0) оставить текущую")
    while True:
        raw = input("Model # ").strip()
        if raw == "" or raw == "0":
            return current
        try:
            n = int(raw)
        except ValueError:
            continue
        if 1 <= n <= len(RECOMMENDED_MODELS):
            return RECOMMENDED_MODELS[n - 1][0]


def delete_model_interactive(host_url: str) -> None:
    installed = api_tags(host_url)
    names = sorted(installed.keys())
    if not names:
        print("Нет установленных моделей.")
        return
    print("")
    print("Удалить модель (номер):")
    for i, name in enumerate(names, start=1):
        print(f"{i:2d}) {name}")
    print(" 0) отмена")
    while True:
        raw = input("Delete # ").strip()
        if raw == "" or raw == "0":
            return
        try:
            n = int(raw)
        except ValueError:
            continue
        if 1 <= n <= len(names):
            name = names[n - 1]
            print(f"Удаляю {name} ...")
            api_delete(host_url, name)
            print("Готово.")
            return


def repl(args: argparse.Namespace) -> int:
    host_url: str = args.host
    gpu_index: int = args.gpu_index
    require_gpu_only: bool = args.require_gpu_only
    current_model: str = args.model

    if args.show_gpu:
        util0 = gpu_util_percent(gpu_index)
        util_s = "n/a" if util0 is None else f"{util0}%"
    else:
        util_s = "monitoring disabled"
    print("Ollama Python CLI REPL")
    print(f"Host:  {host_url}")
    print(f"GPU:   index {gpu_index} (util now {util_s})")
    print("Команды: /exit, /delete, /model")

    # Choose once at startup
    current_model = choose_model_interactive(host_url, current_model)
    ensure_model_ready(host_url, current_model, require_gpu_only=require_gpu_only)

    while True:

        print("")
        prompt = input("> ").strip()
        if not prompt:
            continue
        if prompt == "/exit":
            return 0
        if prompt == "/delete":
            delete_model_interactive(host_url)
            continue
        if prompt == "/model":
            current_model = choose_model_interactive(host_url, current_model)
            ensure_model_ready(host_url, current_model, require_gpu_only=require_gpu_only)
            continue

        ticker = None
        if args.show_gpu:
            ticker = GpuTicker(gpu_index=gpu_index, interval_s=0.5)
            ticker.start()
        try:
            for ev in api_generate_stream(host_url, current_model, prompt, num_predict=args.num_predict):
                chunk = ev.get("response")
                if chunk:
                    sys.stdout.write(str(chunk))
                    sys.stdout.flush()
                if ev.get("done") is True:
                    break
        finally:
            if ticker:
                ticker.stop()
            sys.stdout.write("\n")
            sys.stdout.flush()


def smoke(args: argparse.Namespace) -> int:
    host_url: str = args.host
    model: str = args.model

    if args.choose_model:
        model = choose_model_interactive(host_url, model)

    if args.delete_model:
        print(f"Удаляю модель: {model} ...")
        api_delete(host_url, model)
        print("Готово.")
        return 0

    ensure_model_ready(host_url, model, require_gpu_only=args.require_gpu_only)

    q = args.question
    if not q:
        q = "Дай краткое описание квантовой механики."

    print("Ollama smoke")
    print(f"Host:  {host_url}")
    print(f"Model: {model}")
    if args.show_gpu:
        util0 = gpu_util_percent(args.gpu_index)
        util_s = "n/a" if util0 is None else f"{util0}%"
        print(f"GPU:   index {args.gpu_index} (util now {util_s})")
    else:
        print(f"GPU:   index {args.gpu_index} (monitoring disabled)")
    print("")
    print("Q:")
    # Avoid Windows PowerShell encoding quirks by forcing UTF-8 output
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(q)
    print("")
    print("A:")

    ticker = None
    if args.show_gpu:
        ticker = GpuTicker(gpu_index=args.gpu_index, interval_s=0.5)
        ticker.start()
    try:
        parts = 0
        full = ""
        prompt = q
        while True:
            parts += 1
            done_reason = None
            for ev in api_generate_stream(host_url, model, prompt, num_predict=args.num_predict):
                chunk = ev.get("response")
                if chunk:
                    full += str(chunk)
                    sys.stdout.write(str(chunk))
                    sys.stdout.flush()
                    if args.out_file:
                        with open(args.out_file, "a", encoding="utf-8") as f:
                            f.write(str(chunk))
                if ev.get("done") is True:
                    done_reason = ev.get("done_reason")
                    break
            sys.stdout.write("\n")
            sys.stdout.flush()
            if args.target_words and len(full.split()) >= args.target_words:
                break
            if done_reason != "length":
                break
            if parts >= args.max_parts:
                break
            prompt = "Продолжай ровно с места остановки. Не повторяй уже сказанное. Продолжай текст далее."
    finally:
        if ticker:
            ticker.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ollama_cli.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", default=os.environ.get("OLLAMA_HOST_URL", "http://localhost:11435"))
    common.add_argument("--gpu-index", type=int, default=0)
    common.add_argument("--require-gpu-only", action="store_true", default=True)
    common.add_argument("--show-gpu", action="store_true", default=False, help="Show GPU utilization percentage")

    p_repl = sub.add_parser("repl", parents=[common], help="Interactive REPL with model selection")
    p_repl.add_argument("--model", default="qwen2.5:14b-instruct-q4_K_M")
    p_repl.add_argument("--num-predict", type=int, default=None)
    p_repl.set_defaults(func=repl)

    p_smoke = sub.add_parser("smoke", parents=[common], help="One question then exit")
    p_smoke.add_argument("--model", default="qwen2.5:14b-instruct-q4_K_M")
    p_smoke.add_argument("--choose-model", action="store_true", default=False)
    p_smoke.add_argument("--delete-model", action="store_true", default=False)
    p_smoke.add_argument("--question", default="")
    p_smoke.add_argument("--num-predict", type=int, default=2048)
    p_smoke.add_argument("--target-words", type=int, default=0)
    p_smoke.add_argument("--max-parts", type=int, default=10)
    p_smoke.add_argument("--out-file", default="")
    p_smoke.set_defaults(func=smoke)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        eprint(f"HTTP error: {e.code} {e.reason}")
        if body:
            eprint(body)
        return 2
    except Exception as e:
        eprint(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


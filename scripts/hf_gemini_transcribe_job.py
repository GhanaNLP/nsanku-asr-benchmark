"""HF Job: benchmark Google Gemini 3.5 Transcribe / Transcribe Live.

Run from HF Jobs (CPU is enough — these are hosted API models, no GPU needed):

  python3 scripts/hf_gemini_transcribe_job.py [sync|live] [--langs ewe dag ...]

The script clones the nsanku-asr-benchmark repo, installs runtime deps into a
venv, runs the Gemini Transcribe benchmark (default: the synchronous model on
all eval languages), folds the LLM results into benchmarks/, and pushes the
updated YAMLs back to GitHub main. Requires GITHUB_TOKEN (repo push) and
GEMINI_API_KEY secrets/environment variables.

Results are committed and pushed **per language** so a job timeout never
loses work already scored.
"""
import json
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/GhanaNLP/nsanku-asr-benchmark.git"
BRANCH = "main"
WORK = "/work"

DEPS = [
    "datasets>=3.0.0",
    "soundfile>=0.12.0",
    "numpy",
    "jiwer>=3.0.0",
    "google-genai>=1.0.0",
    "pyyaml>=6.0",
    "huggingface_hub>=0.26.0",
]


def _run(cmd, **kw):
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def _git_commit_push(repo_dir, model_name, scope, token):
    """Stage and push changed results, skipping when there's nothing to do."""
    _run(["git", "add", "-A", "benchmarks/", "transcriptions/"], cwd=repo_dir)
    changed = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_dir,
        capture_output=True, text=True).stdout.strip()
    if not changed:
        print(f"  no changes for {scope} - skipping push", flush=True)
        return
    _run(["git", "-c", "user.email=jobs@nsanku-asr-benchmark",
          "-c", "user.name=nsanku-benchmark-jobs",
          "commit", "-m",
          f"Evaluate {model_name} ({scope}) via HF Jobs (ISO-639-3 hints)"],
         cwd=repo_dir)
    auth_repo = REPO.replace("https://", f"https://{token}@")
    _run(["git", "push", auth_repo, f"HEAD:{BRANCH}"], cwd=repo_dir)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("sync", "live") else "sync"
    model_name = ("gemini-3.5-transcribe-live" if model == "live"
                  else "gemini-3.5-transcribe")
    langs = None
    if "--langs" in sys.argv:
        i = sys.argv.index("--langs")
        langs = sys.argv[i + 1:]

    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set")
    if not os.environ.get("GITHUB_TOKEN"):
        sys.exit("GITHUB_TOKEN not set (needed to push results back)")

    repo_dir = os.path.join(WORK, "nsanku-asr-benchmark")
    if os.path.isdir(repo_dir):
        shutil.rmtree(repo_dir)
    os.makedirs(WORK, exist_ok=True)

    print("== clone repo ==", flush=True)
    _run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, repo_dir])

    print("== setup venv + deps ==", flush=True)
    venv = os.path.join(WORK, ".venv")
    _run(["python3", "-m", "venv", venv])
    pip = os.path.join(venv, "bin", "pip")
    _run([pip, "install", "--quiet", "--upgrade", "pip"])
    _run([pip, "install", "--quiet"] + DEPS)

    py = os.path.join(venv, "bin", "python")
    env = dict(os.environ)
    env["GEMINI_API_KEY"] = os.environ["GEMINI_API_KEY"]
    env.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

    with open(os.path.join(repo_dir, "data", "eval_configs.json")) as f:
        all_langs = list(json.load(f).keys())
    work_langs = langs or all_langs
    print(f"== run {model_name} on {len(work_langs)} language(s) ==", flush=True)
    for i, iso in enumerate(work_langs, 1):
        print(f"\n===== [{i}/{len(work_langs)}] {iso} =====", flush=True)
        _run([py, "run_gemini_transcribe.py", "--model", model_name,
              "--langs", iso], cwd=repo_dir, env=env)
        print(f"== merge LLM results into benchmarks/ ({iso}) ==", flush=True)
        _run([py, "merge_llm.py"], cwd=repo_dir, env=env)
        _git_commit_push(repo_dir, model_name, iso,
                         os.environ["GITHUB_TOKEN"])

    token = os.environ["GITHUB_TOKEN"]
    _git_commit_push(repo_dir, model_name, "final", token)
    print("== DONE ==", flush=True)


if __name__ == "__main__":
    main()
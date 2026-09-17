"""HF Job: benchmark Google Gemini 3.5 Transcribe / Transcribe Live.

Run from HF Jobs (CPU is enough — these are hosted API models, no GPU needed):

  python3 scripts/hf_gemini_transcribe_job.py [sync|live] [--langs ewe dag ...]

The script clones the nsanku-asr-benchmark repo, installs runtime deps into a
venv, runs the Gemini Transcribe benchmark (default: the synchronous model on
all eval languages), folds the LLM results into benchmarks/, and pushes the
updated YAMLs back to GitHub main. Requires GITHUB_TOKEN (repo push) and
GEMINI_API_KEY secrets/environment variables.
"""
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

    print(f"== run {model_name} ==", flush=True)
    cmd = [py, "run_gemini_transcribe.py", "--model", model_name]
    if langs:
        cmd += ["--langs"] + langs
    _run(cmd, cwd=repo_dir, env=env)

    print("== merge LLM results into benchmarks/ ==", flush=True)
    _run([py, "merge_llm.py"], cwd=repo_dir, env=env)

    print("== commit + push ==", flush=True)
    _run(["git", "add", "benchmarks/", "transcriptions/", "benchmarks_llm/"],
         cwd=repo_dir)
    _run(["git", "-c", "user.email=jobs@nsanku-asr-benchmark",
          "-c", "user.name=nsanku-benchmark-jobs",
          "commit", "-m",
          f"Evaluate Gemini {model_name} via HF Jobs (ISO-639-3 hints)"],
         cwd=repo_dir)
    # Push with the token embedded in the URL.
    token = os.environ["GITHUB_TOKEN"]
    auth_repo = REPO.replace("https://", f"https://{token}@")
    _run(["git", "push", auth_repo, f"HEAD:{BRANCH}"], cwd=repo_dir)
    print("== DONE ==", flush=True)


if __name__ == "__main__":
    main()
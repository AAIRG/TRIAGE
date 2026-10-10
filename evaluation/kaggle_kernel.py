#!/usr/bin/env python3
"""Kaggle entry point. Clones AAIRG/TRIAGE and evaluates SMET on the TRIAGE test CVEs.

Steps
 1. Score the SMET test predictions committed in the repo (stdlib only).
 2. Set up a Python 3.9 env (uv) with SMET/requirements.txt, run the released SMET model
    on the test CVEs, and score its fresh predictions.
 3. Compare both against evaluation/paper_reference_metrics.json.
Outputs go to /kaggle/working. Each step logs errors and the script continues.
"""
import json, os, subprocess, sys, urllib.request, shutil

WORK = "/kaggle/working"
TMP = "/kaggle/tmp"
REPO = f"{TMP}/TRIAGE"
VENV = f"{TMP}/smet_py39"
PY39 = f"{VENV}/bin/python"
results = {}


def sh(cmd, cwd=None, check=False):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    if r.stdout:
        print(r.stdout[-3000:], flush=True)
    if r.stderr:
        print(r.stderr[-3000:], flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r.returncode


shutil.rmtree(REPO, ignore_errors=True)
os.makedirs(TMP, exist_ok=True)
sh(f"git clone --depth 1 https://github.com/AAIRG/TRIAGE {REPO}", check=True)
sh(f"git -C {REPO} log --oneline -1 && ls {REPO}/models/uncategorized_mapping/SMET_output", check=False)
urllib.request.urlretrieve("https://raw.githubusercontent.com/basel-a/SMET/main/funs.py",
                           f"{REPO}/SMET/funs.py")

# 1) Score committed SMET predictions
rc = sh(f"python3 evaluation/score_smet.py --predictions models/uncategorized_mapping/SMET_output/test_mappings.json "
        f"--label committed_repo_predictions --out {WORK}/scores_committed.json", cwd=REPO)
results["committed"] = rc == 0

# 2) Fresh SMET run in a Python 3.9 env
env_ok = sh(f"pip install -q uv && uv python install 3.9 && uv venv -q --seed -p 3.9 {VENV} && "
            f"grep -v -i '^pattern' SMET/requirements.txt > {TMP}/smet_req.txt && "
            f"uv pip install -q --python {PY39} -r {TMP}/smet_req.txt", cwd=REPO) == 0
if env_ok:
    # Pattern==3.6 does not resolve under uv; install it with pip (needed by nlp_general.py)
    env_ok = sh(f"{PY39} -m pip install -q Pattern==3.6", cwd=REPO) == 0
results["env_py39"] = env_ok
if env_ok:
    sh(f"{PY39} -m nltk.downloader -q wordnet stopwords punkt", cwd=REPO)
    rc = sh(f"{PY39} evaluation/run_smet_fresh.py --out {WORK}/smet_fresh_test_predictions.json", cwd=REPO)
    results["fresh_run"] = rc == 0
    if rc == 0:
        rc = sh(f"python3 evaluation/score_smet.py --predictions {WORK}/smet_fresh_test_predictions.json "
                f"--label fresh_released_smet --out {WORK}/scores_fresh.json", cwd=REPO)
        results["fresh_score"] = rc == 0
else:
    print("Python 3.9 environment failed; fresh SMET run skipped.", flush=True)

with open(f"{WORK}/run_status.json", "w") as f:
    json.dump(results, f, indent=2)
print("STATUS", results, flush=True)
for name in ["scores_committed.json", "scores_fresh.json"]:
    p = f"{WORK}/{name}"
    if os.path.exists(p):
        print(f"== {name}", flush=True)
        print(open(p).read(), flush=True)

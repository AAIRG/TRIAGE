#!/usr/bin/env python3
"""Run the released SMET model (SMET/SMET.py) over the TRIAGE test CVEs.

Mirrors SMET/run_smet.py with SPLIT="test". Must run in the Python 3.9 SMET environment
(see SMET/requirements.txt). Writes a predictions JSON in the same format as
models/uncategorized_mapping/SMET_output/test_mappings.json.
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMET_DIR = os.path.join(ROOT, "SMET")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import pandas as pd
    # Registers AllenNLP's 'srl' dataset reader; parse_class.py needs it but SMET does not import it.
    import importlib
    for _mod in ["allennlp_models.structured_prediction", "allennlp_models.syntax.srl"]:
        try:
            importlib.import_module(_mod)
            print("registered SRL reader via", _mod, flush=True)
            break
        except ImportError as e:
            print("import failed:", _mod, e, flush=True)
    sys.path.insert(0, SMET_DIR)
    os.chdir(SMET_DIR)
    from SMET import map_text, get_clf_model, get_emb_model, get_attack_ids

    cve_base = pd.read_csv(os.path.join(ROOT, "data/pre_processed/cve_base.csv"))
    test = pd.read_csv(os.path.join(ROOT, "data/pre_processed/cves_test.csv"))
    df = pd.merge(cve_base, test, on="CVE ID").loc[:, ["CVE ID", "description"]]
    if args.limit:
        df = df.head(args.limit)
    print(f"test CVEs: {len(df)}", flush=True)

    id2label = get_attack_ids(os.path.join(SMET_DIR, "id2ATT&CK_V2.json"))
    clf = get_clf_model(os.path.join(SMET_DIR, "LR_ATT&CK_model_V2.pkl"))
    emb = get_emb_model()

    out, t0 = [], time.time()
    for i, (cve, desc) in enumerate(zip(df["CVE ID"], df["description"])):
        mapping = map_text(desc, clf, emb, id2label, CVE=True)
        mapping = list(zip(*mapping))
        out.append({"target_cve": cve, "predictions": list(mapping[0]), "confidence": list(mapping[1])})
        if i % 10 == 0:
            print(f"  mapped {i + 1}/{len(df)} ({time.time() - t0:.0f}s)", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {len(out)} entries to {args.out}", flush=True)


if __name__ == "__main__":
    main()

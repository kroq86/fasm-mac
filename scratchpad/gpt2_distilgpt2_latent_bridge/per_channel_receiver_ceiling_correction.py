#!/usr/bin/env python3
"""Replace the confounded sender ceiling with an already measured receiver ceiling."""
import argparse, json, math
from pathlib import Path


def summary(rows):
    return {"nll": sum(r["nll"] for r in rows)/len(rows),
            "top1": sum(r["top1"] for r in rows)/len(rows)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--ceiling",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args(); d=json.loads(args.input.read_text()); c=json.loads(args.ceiling.read_text())
    full=c["per_document"]["receiver_full_context"]
    neutral=d["per_document_dev"]["neutral"]
    correct=d["per_document_dev"]["correct"]
    assert [r["sha256"] for r in neutral] == [r["sha256"] for r in full]
    headroom=sum(a["nll"]-b["nll"] for a,b in zip(neutral,full))/len(full)
    gain=sum(a["nll"]-b["nll"] for a,b in zip(neutral,correct))/len(full)
    fraction=gain/headroom
    d["dev"].pop("full_context_sender",None)
    d["per_document_dev"].pop("full_context_sender",None)
    d["dev"]["full_context_receiver"]=summary(full)
    d["per_document_dev"]["full_context_receiver"]=full
    d["dev"]["headroom_nll"]=headroom
    d["dev"]["fraction_of_headroom_closed"]=fraction
    d["ceiling_correction"]="receiver-matched DistilGPT2 full context; no retraining"
    content_ci=d["dev"]["comparisons"]["shuffled_minus_correct"]["bootstrap_95pct_ci"]
    zero_ci=d["dev"]["comparisons"]["zero_sender_minus_correct"]["bootstrap_95pct_ci"]
    neutral_ci=d["dev"]["comparisons"]["neutral_minus_correct"]["bootstrap_95pct_ci"]
    sufficient=all(math.isfinite(x) for x in (headroom,gain,fraction)) and fraction>=.20 and min(content_ci)>0 and min(zero_ci)>0 and min(neutral_ci)>0
    d["verdict"]="PER_CHANNEL_SUFFICIENT" if sufficient else "PER_CHANNEL_INSUFFICIENT"
    args.output.write_text(json.dumps(d,sort_keys=True,indent=2)+"\n")
    print(f"headroom={headroom:.6f} fraction={fraction:.6f} verdict={d['verdict']}")


if __name__=="__main__": main()

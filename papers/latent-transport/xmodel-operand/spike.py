"""Cross-model operand spike (see PREREG.md). Inference only.

usage: python spike.py <receiver: gpt2|distilgpt2|EleutherAI/pythia-160m>
"""
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
torch.set_grad_enabled(False)

CATS = ["animal", "plant", "tool", "vehicle", "furniture"]
# Fixed before any run. Screening decides which survive per receiver.
CANDIDATES = {
    "animal": ["dog", "cat", "horse", "cow", "pig", "sheep", "lion", "tiger", "wolf", "bear", "rabbit", "fox"],
    "plant": ["rose", "oak", "tree", "grass", "fern", "moss", "pine", "tulip", "lily", "weed", "shrub", "vine"],
    "tool": ["hammer", "wrench", "saw", "drill", "screwdriver", "chisel", "shovel", "axe", "knife", "pliers", "rake", "file"],
    "vehicle": ["car", "truck", "bus", "bike", "bicycle", "van", "train", "boat", "tractor", "motorcycle", "taxi", "tank"],
    "furniture": ["chair", "table", "sofa", "desk", "bed", "couch", "stool", "bench", "cabinet", "dresser", "shelf", "wardrobe"],
}
PROMPT = ["A", " thing", " is", " a", " type", " of"]
PATCH_POS = 1


def single_token(tok, word):
    ids = tok.encode(" " + word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


class Receiver:
    def __init__(self, name):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
        self.is_gpt2 = hasattr(self.model, "transformer")
        if self.is_gpt2:
            self.wte = self.model.transformer.wte.weight
            self.wpe = self.model.transformer.wpe.weight
        else:
            self.wte = self.model.get_input_embeddings().weight
            self.wpe = None
        ids = [self.tok.encode(p, add_special_tokens=False) for p in PROMPT]
        assert all(len(i) == 1 for i in ids), f"prompt not single-token per piece: {ids}"
        self.prompt_ids = torch.tensor([i[0] for i in ids])
        self.cat_ids = [single_token(self.tok, c) for c in CATS]
        assert all(c is not None for c in self.cat_ids), "category word not single token"

    def layer0_input(self, token_id):
        """Full layer-0 input (token + absolute position if any) at PATCH_POS."""
        v = self.wte[token_id].clone()
        if self.wpe is not None:
            v = v + self.wpe[PATCH_POS]
        return v

    def run(self, patch_vecs):
        """patch_vecs: [n, d] full layer-0 inputs to place at PATCH_POS. Returns [n, 5] category logits."""
        n = patch_vecs.shape[0]
        emb = self.wte[self.prompt_ids].unsqueeze(0).repeat(n, 1, 1).clone()
        pv = patch_vecs.clone()
        if self.wpe is not None:
            pv = pv - self.wpe[PATCH_POS]  # model adds wpe itself
        emb[:, PATCH_POS, :] = pv
        out = self.model(inputs_embeds=emb).logits[:, -1, :]
        return out[:, self.cat_ids]

    def native_logits(self, word_ids):
        n = len(word_ids)
        ids = self.prompt_ids.unsqueeze(0).repeat(n, 1).clone()
        ids[:, PATCH_POS] = torch.tensor(word_ids)
        return self.model(input_ids=ids).logits[:, -1, :][:, self.cat_ids]


def sender_states(sender, word_ids):
    """GPT-2 final-block output before ln_f, word alone at position 0."""
    captured = {}

    def hook(_m, _i, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    h = sender.transformer.h[-1].register_forward_hook(hook)
    sender(input_ids=torch.tensor(word_ids).unsqueeze(1))
    h.remove()
    return captured["h"][:, 0, :]


def score(logits, cat_idx):
    logits = np.asarray(logits, dtype=np.float64)
    n = len(cat_idx)
    S = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i and cat_idx[j] != cat_idx[i]]
        ctrl = logits[others].mean(axis=0)
        delta = logits[i] - ctrl
        c = cat_idx[i]
        S[i] = delta[c] - np.mean([delta[k] for k in range(len(CATS)) if k != c])
    return S


def summarize(S):
    mean, std = S.mean(), S.std()
    t = mean / (std / np.sqrt(len(S)) + 1e-12)
    return {"pos": int((S > 0).sum()), "n": len(S), "mean": round(float(mean), 4), "std": round(float(std), 4), "t": round(float(t), 2)}


def score_against_real(cand_logits, real_logits, cat_idx):
    """Condition logits for member i vs leave-one-out control built from REAL raw logits."""
    real = np.asarray(real_logits, dtype=np.float64)
    cand = np.asarray(cand_logits, dtype=np.float64)
    n = len(cat_idx)
    S = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i and cat_idx[j] != cat_idx[i]]
        delta = cand[i] - real[others].mean(axis=0)
        c = cat_idx[i]
        S[i] = delta[c] - np.mean([delta[k] for k in range(len(CATS)) if k != c])
    return S


def main(receiver_name):
    sender_tok = AutoTokenizer.from_pretrained("gpt2")
    sender = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32).eval()
    R = Receiver(receiver_name)

    # screening on the bare receiver with the native word
    words, s_ids, r_ids, cat_idx = [], [], [], []
    rejected = []
    for c, cat in enumerate(CATS):
        for w in CANDIDATES[cat]:
            si, ri = single_token(sender_tok, w), single_token(R.tok, w)
            if si is None or ri is None:
                rejected.append((w, "multi-token"))
                continue
            lg = R.native_logits([ri])[0].numpy()
            margin = lg[c] - max(lg[k] for k in range(len(CATS)) if k != c)
            if margin >= 1.0:
                words.append(w); s_ids.append(si); r_ids.append(ri); cat_idx.append(c)
            else:
                rejected.append((w, f"margin {margin:+.2f}"))
    per_cat = {cat: sum(1 for c in cat_idx if c == k) for k, cat in enumerate(CATS)}
    print(f"receiver={receiver_name}  screened {len(words)}/{sum(len(v) for v in CANDIDATES.values())}  per-category={per_cat}")

    raw_vec = sender_states(sender, s_ids)
    native_vec = torch.stack([R.layer0_input(i) for i in r_ids])
    gen = torch.Generator().manual_seed(424242)
    rnd = torch.randn(raw_vec.shape, generator=gen)
    rnd = rnd / rnd.norm(dim=1, keepdim=True) * raw_vec.norm(dim=1, keepdim=True)

    # aligned baseline: ridge sender-final -> receiver layer-0 input, fit on disjoint vocab
    cand_words = {w for v in CANDIDATES.values() for w in v} | set(CATS) | {"thing", "type"}
    train_s, train_r = [], []
    for tid in range(300, 50257):
        s = sender_tok.decode([tid])
        if not (s.startswith(" ") and s[1:].isalpha() and s[1:].islower() and len(s) > 3):
            continue
        w = s[1:]
        if w in cand_words:
            continue
        ri = single_token(R.tok, w)
        if ri is None:
            continue
        train_s.append(tid); train_r.append(ri)
        if len(train_s) == 2000:
            break
    Xs = torch.cat([sender_states(sender, train_s[k:k + 500]) for k in range(0, len(train_s), 500)]).double()
    Yr = torch.stack([R.layer0_input(i) for i in train_r]).double()
    Xa = torch.cat([Xs, torch.ones(len(Xs), 1, dtype=torch.float64)], 1)
    lam = 1.0
    W = torch.linalg.solve(Xa.T @ Xa + lam * torch.eye(Xa.shape[1], dtype=torch.float64), Xa.T @ Yr)
    aligned_vec = (torch.cat([raw_vec.double(), torch.ones(len(raw_vec), 1, dtype=torch.float64)], 1) @ W).float()

    res = {"receiver": receiver_name, "words": words, "per_category": per_cat, "rejected": rejected,
           "align_train_n": len(train_s)}
    logits = {}
    for name, vec in [("native", native_vec), ("raw", raw_vec), ("aligned", aligned_vec)]:
        logits[name] = R.run(vec).numpy()
        res[name] = summarize(score(logits[name], cat_idx))
    res["random_vs_real"] = summarize(score_against_real(R.run(rnd).numpy(), logits["raw"], cat_idx))
    S_raw = score(logits["raw"], cat_idx)
    res["corr_S_norm"] = round(float(np.corrcoef(S_raw, raw_vec.norm(dim=1).numpy())[0, 1]), 3)
    rng = np.random.default_rng(7)
    real_mean = S_raw.mean()
    perm_means = [score(logits["raw"], list(rng.permutation(cat_idx))).mean() for _ in range(200)]
    res["perm"] = {"mean": round(float(np.mean(perm_means)), 4), "ge_real": int(sum(m >= real_mean for m in perm_means))}
    res["raw_per_word"] = {w: round(float(s), 3) for w, s in zip(words, S_raw)}
    res["norms"] = {"raw_mean": round(float(raw_vec.norm(dim=1).mean()), 1),
                    "native_mean": round(float(native_vec.norm(dim=1).mean()), 1)}
    print(json.dumps({k: v for k, v in res.items() if k not in ("raw_per_word", "rejected", "words")}, indent=1))
    out = f"result_{receiver_name.replace('/', '_')}.json"
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1])

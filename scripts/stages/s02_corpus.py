"""Stage 2 - build the calibration corpus.

Long pole, and deliberately dependency-free so it runs in parallel with source staging.

Robustness is the priority: a corpus build that dies on one bad dataset after six hours is
worse than one that logs the failure and moves to the next source. Every source is attempted
independently; a bucket succeeds if it reaches its quota from *any* combination of its sources.

Resumable at two levels: raw text per bucket is appended to JSONL and re-read on restart, and
tokenised shards are written incrementally.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, MODEL_ID, log, metric, kv_set, kv_get, publish, hf_token  # noqa: E402
import corpus_spec as SPEC  # noqa: E402
import importlib

STAGE = "s02_corpus"
CORPUS = ROOT / "corpus"
RAW = CORPUS / "raw"
SHARDS = CORPUS / "shards"
for d in (RAW, SHARDS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 1337
MIN_CHARS = 200


def _band(n_chars: int) -> str:
    approx_tok = n_chars / 3.6
    if approx_tok < SPEC.BAND_EASY[1]:
        return "easy"
    if approx_tok < SPEC.BAND_MEDIUM[1]:
        return "medium"
    return "hard"


def _quota_by_band(total: int) -> dict[str, int]:
    return {b: int(total * f) for b, f in SPEC.DIFFICULTY_TARGET.items()}


def _stream(hf_id, config, split):
    """Self-healing loader.

    Config/split names drift and are easy to get wrong from a model card. Rather than failing a
    source outright, recover the two mistakes that account for nearly all of them: a missing
    config name, and a split that does not exist under the chosen config.
    """
    from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
    tok = hf_token()
    try:
        return load_dataset(hf_id, config, split=split, streaming=True, token=tok)
    except Exception as e:
        msg = str(e)
        if "Config name is missing" in msg or "BuilderConfig" in msg:
            cfgs = get_dataset_config_names(hf_id, token=tok)
            if cfgs:
                config = cfgs[0]
                log(f"{hf_id}: config auto-corrected to '{config}'", STAGE, "WARN")
        elif "Bad split" not in msg and "Unknown split" not in msg:
            raise
        try:
            return load_dataset(hf_id, config, split=split, streaming=True, token=tok)
        except Exception:
            splits = get_dataset_split_names(hf_id, config, token=tok)
            if not splits:
                raise
            log(f"{hf_id}({config}): split auto-corrected '{split}' -> '{splits[0]}'",
                STAGE, "WARN")
            return load_dataset(hf_id, config, split=splits[0], streaming=True, token=tok)


def collect_bucket(bucket: str, quota: int) -> int:
    """Fill one text bucket, respecting the difficulty bands. Returns samples written."""
    out = RAW / f"{bucket}.jsonl"
    have = 0
    seen_bands: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    if out.exists():
        with out.open() as fh:
            for line in fh:
                try:
                    seen_bands[json.loads(line)["band"]] += 1
                    have += 1
                except Exception:
                    pass
    if have >= quota:
        log(f"{bucket}: already have {have}/{quota}; skipping", STAGE)
        return have

    band_quota = _quota_by_band(quota)
    log(f"{bucket}: need {quota} (have {have}); bands {band_quota}", STAGE)
    rng = random.Random(SEED + hash(bucket) % 10_000)

    with out.open("a") as fh:
        for hf_id, config, split, weight, text_fn in SPEC.SOURCES[bucket]:
            if have >= quota:
                break
            want = int(quota * weight)
            got = 0
            t0 = time.time()
            try:
                ds = _stream(hf_id, config, split)
                for row in ds:
                    if have >= quota or got >= want:
                        break
                    if time.time() - t0 > 1800:      # never let one source stall the build
                        log(f"{bucket}/{hf_id}: 30 min cap hit at {got}", STAGE, "WARN")
                        break
                    try:
                        text = text_fn(row)
                    except Exception:
                        continue
                    if not text or len(text) < MIN_CHARS:
                        continue
                    band = _band(len(text))
                    if seen_bands[band] >= band_quota.get(band, 0) and have < quota * 0.95:
                        continue          # band already satisfied; keep looking
                    fh.write(json.dumps({"text": text[:80_000], "band": band,
                                         "src": hf_id, "bucket": bucket}) + "\n")
                    seen_bands[band] += 1
                    have += 1
                    got += 1
                    if got % 250 == 0:
                        fh.flush()
                log(f"{bucket}/{hf_id}: +{got} in {(time.time()-t0)/60:.1f} min "
                    f"(bucket {have}/{quota})", STAGE)
            except Exception as e:
                log(f"{bucket}/{hf_id}: SKIPPED ({type(e).__name__}: {str(e)[:180]})",
                    STAGE, "WARN")
                continue
    metric(STAGE, f"raw_samples_{bucket}", have)
    if have < quota * 0.5:
        log(f"{bucket}: only reached {have}/{quota} - under half quota", STAGE, "WARN")
    return have


def collect_multimodal(quota: int) -> int:
    """Real image-text pairs only. A text description of an image routes like text and
    protects nothing (risk R3)."""
    import torch
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token())
    outdir = SHARDS / "multimodal"
    outdir.mkdir(parents=True, exist_ok=True)
    done = kv_get("mm_done", 0)
    if done >= quota:
        log(f"multimodal: already have {done}/{quota}", STAGE)
        return done

    buf, idx, have = [], len(list(outdir.glob("mm_*.pt"))), done
    for hf_id, config, split, weight in SPEC.MM_SOURCES:
        if have >= quota:
            break
        want = int(quota * weight)
        got, t0 = 0, time.time()
        try:
            ds = _stream(hf_id, config, split)
            for row in ds:
                if have >= quota or got >= want or time.time() - t0 > 2400:
                    break
                img = row.get("image") or row.get("images")
                if isinstance(img, list):
                    img = img[0] if img else None
                if img is None:
                    continue
                try:
                    img = img.convert("RGB")
                    q = (row.get("question") or row.get("query") or
                         row.get("texts") or "Describe this image in detail.")
                    if isinstance(q, list):
                        q = json.dumps(q)[:2000]
                    msgs = [{"role": "user", "content": [{"type": "image"},
                                                         {"type": "text", "text": str(q)[:2000]}]}]
                    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                    b = proc(text=[text], images=[img], return_tensors="pt")
                    n_img = int((b["input_ids"][0] == 154854).sum())
                    if n_img == 0:
                        continue
                    buf.append({k: v[0] if hasattr(v, "__getitem__") else v
                                for k, v in b.items()})
                    have += 1
                    got += 1
                except Exception:
                    continue
                if len(buf) >= 64:
                    torch.save(buf, outdir / f"mm_{idx:05d}.pt")
                    idx += 1
                    buf = []
                    kv_set("mm_done", have)
            log(f"multimodal/{hf_id}({config}): +{got} ({have}/{quota})", STAGE)
        except Exception as e:
            log(f"multimodal/{hf_id}({config}): SKIPPED ({type(e).__name__}: {str(e)[:180]})",
                STAGE, "WARN")
            continue
    if buf:
        torch.save(buf, outdir / f"mm_{idx:05d}.pt")
    kv_set("mm_done", have)
    metric(STAGE, "raw_samples_multimodal", have)
    return have


def tokenize_text() -> dict:
    """Tokenise all raw text buckets into fixed shards of input_ids."""
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token())
    outdir = SHARDS / "text"
    outdir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}
    total_tokens = 0
    for raw in sorted(RAW.glob("*.jsonl")):
        bucket = raw.stem
        shard_path = outdir / f"{bucket}.pt"
        if shard_path.exists():
            d = torch.load(shard_path, weights_only=False)
            stats[bucket] = len(d)
            total_tokens += sum(len(x["input_ids"]) for x in d)
            continue
        items = []
        with raw.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                ids = tok(rec["text"], truncation=True, max_length=SPEC.MAX_TOKENS,
                          return_tensors="pt")["input_ids"][0]
                if len(ids) < 32:
                    continue
                items.append({"input_ids": ids.to(torch.int32), "bucket": bucket,
                              "band": rec["band"], "src": rec["src"]})
        torch.save(items, shard_path)
        stats[bucket] = len(items)
        total_tokens += sum(len(x["input_ids"]) for x in items)
        log(f"tokenised {bucket}: {len(items)} samples", STAGE)
    metric(STAGE, "corpus_text_tokens", total_tokens)
    return {"per_bucket": stats, "total_tokens": total_tokens}


def run() -> dict:
    importlib.reload(SPEC)   # pick up spec fixes on retry without a service restart
    quotas = {b: round(SPEC.TOTAL_SAMPLES * f) for b, f in SPEC.MIXTURE.items()}
    res: dict = {"quotas": quotas}

    for bucket in [b for b in quotas if b != "multimodal"]:
        res.setdefault("raw", {})[bucket] = collect_bucket(bucket, quotas[bucket])

    res.setdefault("raw", {})["multimodal"] = collect_multimodal(quotas["multimodal"])
    res["text"] = tokenize_text()

    # The single highest-value assertion in the pipeline (risk R3): if calibration carries no
    # image tokens, REAP deletes vision-serving experts with certainty at any prune ratio.
    n_mm = res["raw"]["multimodal"]
    if n_mm == 0:
        raise AssertionError(
            "calibration corpus contains ZERO image-text samples. Proceeding would delete "
            "vision-serving experts with certainty (R3). Blocking.")
    if n_mm < quotas["multimodal"] * 0.5:
        log(f"multimodal only reached {n_mm}/{quotas['multimodal']} - vision protection is "
            f"weaker than planned", STAGE, "WARN")

    res["total_raw"] = sum(res["raw"].values())
    manifest = CORPUS / "manifest.json"
    manifest.write_text(json.dumps(res, indent=2, default=str))
    publish(manifest, "artifacts", "stage02/manifest.json", stage=STAGE)
    log(f"corpus built: {res['total_raw']} samples, "
        f"{res['text']['total_tokens']:,} text tokens, {n_mm} image-text", STAGE)
    return res

"""catalog.py - models that exist outside, as facts to choose from (D-13 in claude/data/decisions.md): what a hub lists, each file's size and checksum, how it would fit and run on each machine, and a general capability scale fitted here from public benchmark results.
Nothing here contacts anything: callers fetch through a machine's web_fetch (the node does the outside contact, D-02) and hand the text in. Every fetched text is kept as a dated snapshot, so a choice can be traced to what was read and when, and read again offline.
Sources are data: a hub with the Hugging Face API shape is another base address; the benchmark scale is any table of (model, benchmark, score) rows.
The scale is fitted the way Epoch AI's Capabilities Index is in spirit (a capability per model and a difficulty and slope per benchmark, scores on a logistic curve) but by a small local fit, so its numbers are its own: shown as standard deviations from the mean model, never as Epoch's published ECI values."""
import re, csv, io, json, math, time
from pathlib import Path
from urllib.parse import quote

DIR = Path("./data/_tools/aviary/catalog")   # set by Aviary to its own data folder
HF = "https://huggingface.co"
_QUANT = re.compile(r"(?i)(?:^|[-_.])((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?|NVFP4))(?=[-_.]|$)")
_SHARD = re.compile(r"-\d{5}-of-(\d{5})(?=\.gguf$)")

# --- Addresses ---

def search_url(q: str, limit: int = 12, base: str = HF) -> str: return f"{base}/api/models?search={quote(q)}&filter=gguf&sort=downloads&direction=-1&limit={int(limit)}"
def detail_url(mid: str, base: str = HF) -> str: return f"{base}/api/models/{mid}?blobs=true"
def file_url(mid: str, name: str, base: str = HF) -> str: return f"{base}/{mid}/resolve/main/{quote(name)}"
def page_url(mid: str, base: str = HF) -> str: return f"{base}/{mid}"

# --- Snapshots ---

def keep(source: str, name: str, text: str) -> Path:
    """A fetched text kept under <DIR>/<source>/<time>-<name>; nothing is replaced."""
    p = DIR / source / f"""{time.strftime("%Y%m%d-%H%M%S")}-{re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120]}"""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p

def latest(source: str, name: str = "") -> tuple:
    """(text, its file) of the newest snapshot of a source (and name), or ("", None)."""
    hits = sorted((DIR / source).glob(f"*-{name}*" if name else "*")) if (DIR / source).exists() else []
    return (hits[-1].read_text(), hits[-1]) if hits else ("", None)

# --- Hub facts ---

def _tagged(tags: list, prefix: str) -> str: return next((t.split(":")[-1] for t in tags if t.startswith(prefix)), "")

def parse_search(text: str) -> list:
    """[{id, downloads, likes, license, base, gated}] from a hub's model search answer."""
    return [{"id": m["id"], "downloads": m.get("downloads", 0), "likes": m.get("likes", 0), "license": _tagged(m.get("tags", []), "license:"), "base": _tagged(m.get("tags", []), "base_model:"), "gated": bool(m.get("gated"))} for m in json.loads(text)]

def quant_of(name: str) -> str: return (m.group(1).upper() if (m := _QUANT.search(Path(name).stem)) else "?")

def parse_detail(text: str) -> dict:
    """One model's facts: parameters, architecture, context, license, base model, and its GGUF files - split files joined into one entry (first part's name, summed size, every part's path), vision projectors marked."""
    d = json.loads(text)
    g, tags, files = d.get("gguf") or {}, d.get("tags", []), {}
    for s in d.get("siblings", []):
        n = s["rfilename"]
        if not n.endswith(".gguf"): continue
        key = _SHARD.sub("", n)
        f = files.setdefault(key, {"name": n, "quant": quant_of(key), "bytes": 0, "sha256": [], "parts": 0, "paths": [], "projector": "mmproj" in n.lower()})
        f["name"], f["bytes"], f["parts"] = min(f["name"], n), f["bytes"] + (s.get("size") or 0), f["parts"] + 1
        f["sha256"].append((s.get("lfs") or {}).get("sha256", ""))
        f["paths"] = sorted(f["paths"] + [n])   # every part's path in the repository: a split model is downloaded part by part, side by side
    return {"id": d["id"], "params": g.get("total") or 0, "arch": g.get("architecture", ""), "ctx": g.get("context_length") or 0, "license": (d.get("cardData") or {}).get("license") or _tagged(tags, "license:"), "base": _tagged(tags, "base_model:"), "gated": bool(d.get("gated")), "downloads": d.get("downloads", 0), "files": sorted(files.values(), key=lambda f: f["bytes"])}

def bits_per_weight(f: dict, params: int) -> float: return round(f["bytes"] * 8 / params, 2) if params else 0.0

# --- Fit on a machine (estimates; a machine's own /plan is exact once the file is there) ---

def fit(file_bytes: int, ram_gb: float, rate_bytes_s: float, reserve_gb: float = 4.0, kv_gb: float = 1.5) -> dict:
    """Whether a file would fit in a machine's memory beside its system (reserve_gb) with a working context (kv_gb), and its decode speed from the machine's effective rate. A mixture-of-experts file reads less per token than its size: this counts the whole file, so it understates their speed."""
    need = file_bytes / 1e9 + kv_gb
    return {"need_gb": round(need, 1), "fits": bool(ram_gb) and need <= ram_gb - reserve_gb, "tps": round(rate_bytes_s / file_bytes, 1) if rate_bytes_s and file_bytes else None}

# --- General capability scale, fitted here from public benchmark results ---

def bench_rows(text: str) -> list:
    """[(model, benchmark, score 0..1)] from a table with columns model (or Model), benchmark and performance."""
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        try: y = float(r.get("performance") or "")
        except ValueError: continue
        if 0.0 <= y <= 1.0 and (m := r.get("model") or r.get("Model")) and r.get("benchmark"): rows.append((m.strip(), r["benchmark"].strip(), y))
    return rows

def fit_scale(rows: list, rounds: int = 200, lr: float = 0.05) -> dict:
    """{model: capability} by a logistic fit: score = 1 / (1 + exp(-slope_b * (capability_m - difficulty_b))), fitted by gradient steps; capabilities are then shown as standard deviations from the mean model. Models with fewer than 2 results are left out (one score cannot place a model)."""
    count = {}
    for m, _, _ in rows: count[m] = count.get(m, 0) + 1
    rows = [r for r in rows if count[r[0]] >= 2]
    c, d, a = {r[0]: 0.0 for r in rows}, {r[1]: 0.0 for r in rows}, {r[1]: 1.0 for r in rows}
    for _ in range(rounds):
        for m, b, y in rows:
            z = a[b] * (c[m] - d[b])
            g = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))) - y
            c[m], d[b], a[b] = c[m] - lr * g * a[b], d[b] + lr * g * a[b], min(5.0, max(0.2, a[b] - lr * g * (c[m] - d[b])))
        mean = sum(c.values()) / len(c)
        c = {m: v - mean for m, v in c.items()}
    sd = math.sqrt(sum(v * v for v in c.values()) / len(c)) or 1.0
    return {m: round(v / sd, 2) for m, v in c.items()}

def _norm(s: str) -> str: return re.sub(r"[^a-z0-9.]", "", s.lower().split("/")[-1])

def scale_of(names: list, scale: dict) -> tuple:
    """(score, the scale's own name for it) for the first of names (a base model, the repository id) that the scale holds under the same name, letters and digits compared; (None, "") when none does. No fuzzy matching: a near name is a different model."""
    by = {_norm(k): k for k in scale}
    return next(((scale[by[_norm(n)]], by[_norm(n)]) for n in names if n and _norm(n) in by), (None, ""))

# --- Finder: the old AI Calc's search, filters and ranking, its tables as data (catalog_config.json beside this file; an instance replaces any key in <DIR>/config.json) ---

_SHIPPED = Path(__file__).parent / "catalog_config.json"
_BENCH_ALIASES = {"mmlu": "mmlu", "massive multitask": "mmlu", "mmlu_pro": "mmlu_pro", "arc": "arc", "arc_challenge": "arc", "ai2_arc": "arc", "hellaswag": "hellaswag", "truthfulqa": "truthfulqa", "truthful_qa": "truthfulqa", "winogrande": "winogrande", "gsm8k": "gsm8k", "ifeval": "ifeval", "humaneval": "humaneval", "pass@1": "humaneval", "bbh": "bbh", "big bench hard": "bbh", "gpqa": "gpqa", "gpqa_diamond": "gpqa", "hle": "hle"}
_AVG_W = {"mmlu": 3, "mmlu_pro": 3.5, "arc": 2, "hellaswag": 1.5, "truthfulqa": 1.5, "winogrande": 1, "gsm8k": 1.5, "ifeval": 2, "bbh": 2, "gpqa": 2.5, "hle": 4}
_INTEL_W = {"hle": 4.0, "gpqa": 3.0, "mmlu_pro": 2.5, "ifeval": 2.5, "bbh": 2.0, "gsm8k": 1.5, "mmlu": 1.0, "arc": 1.0, "hellaswag": 0.5}
CURVES = {"cap": "credit rises to 2x the target, then flat", "open": "keeps rising past the target (0.5 at the target)", "peak": "best at the target: prefers the biggest model that just meets it"}

def config() -> dict:
    """The finder's tables: the shipped file, each key replaced whole by the instance's <DIR>/config.json when that has it."""
    own = DIR / "config.json"
    return {**json.loads(_SHIPPED.read_text()), **(json.loads(own.read_text()) if own.exists() else {})}

_QTOKEN = re.compile(r"^(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|FP16|F16|F32|MXFP4(?:_MOE)?|NVFP4)$")
_NOISE = {"gguf", "ud", "latest", "hf.co", "huggingface.co"}

def words_of(text: str) -> tuple:
    """(name words, quant words) from what was typed: split on spaces, commas, - / and :, so a name typed whole (Qwen3.5-4B-Q8_0) and its parts in any order (4b qwen3.5 q8) read the same. Quant words (q8, Q4_K_M, bf16) choose quants, not models."""
    parts = [x for x in re.split(r"[\s,/:\-]+", text.strip()) if x and x.lower() not in _NOISE]
    return [x.lower() for x in parts if not _QTOKEN.match(x.upper())], [x.upper().replace("FP16", "F16") for x in parts if _QTOKEN.match(x.upper())]

def _alnum(s: str) -> str: return re.sub(r"[^a-z0-9]", "", s.lower())

def quant_wanted(q: str, wanted: list) -> bool: return not wanted or any(q.replace("FP16", "F16") == w or q.startswith(w + "_") or (w[-1].isdigit() and q.startswith(w) and not q[len(w)].isdigit()) for w in wanted if q)

def finder_urls(preset: dict, words: str = "", extra: str = "", base: str = HF) -> list:
    """The Hub lists to read for one search, each by likes and by downloads, with each model's file names (full=true) so quants are known before any model page is read: the words typed searched whole, then shorter (qwen3.5-4b-instruct, qwen3.5-4b, qwen3.5 - the Hub matches a name's continuous part), the extra query, the task's own queries (3 at most in all); the most liked and downloaded GGUF models when a task is chosen or nothing was typed; the task's first 3 tags. Only filter=gguf: many chat models are listed under other tasks (image-text-to-text for Qwen3.5)."""
    gg, names = "filter=gguf&full=true&direction=-1", words_of(words)[0]
    qs = list(dict.fromkeys(["-".join(names[:n]) for n in range(len(names), 0, -1)][:3] + ([extra] if extra else []) + list(preset.get("queries", []))))[:3 if not names else 4]
    out = [f"{base}/api/models?search={quote(q)}&{gg}&sort={s}&limit=100" for q in qs for s in ("likes", "downloads")]
    if preset.get("queries") or not (names or extra): out += [f"{base}/api/models?{gg}&sort={s}&limit=200" for s in ("likes", "downloads")]
    return out + [f"{base}/api/models?{gg}&filter={quote(t)}&sort=likes&limit=60" for t in preset.get("any_tags", [])[:3]]

def params_from_name(name: str):
    nl = name.lower().replace("_", "-")
    for pat in (r"(\d+\.?\d*)-?b(?:illion)?(?:\b|[_\-])", r"(?:^|[_\-])(\d+\.?\d*)b(?:[_\-]|$)", r"\b(\d+\.?\d*)b\b"):
        if (m := re.search(pat, nl)) and 0.1 <= float(m.group(1)) <= 500: return float(m.group(1))
    return None

def params_b(m: dict):
    """Billions of parameters: from the file header (gguf.total) or safetensors total when the Hub gives them, else from the name ("4B")."""
    t = (m.get("gguf") or {}).get("total") or (m.get("safetensors") or {}).get("total")
    return round(t / 1e9, 2) if t and t > 1e6 else params_from_name(m.get("id", ""))

def traits(m: dict, cfg: dict) -> dict:
    """kv_frac (share of layers with a context cache) and mtp (has MTP / nextn heads), from architecture, id and tags."""
    blob = f"""{m.get("id", "")} {" ".join(m.get("tags") or [])} {(m.get("gguf") or {}).get("architecture", "") or m.get("arch", "")}""".lower()
    return {"kv_frac": next((f for k, f in cfg["KV_LAYER_FRACTION"].items() if not k.startswith("_") and k in blob), 1.0), "mtp": any(p in blob for p in cfg["MTP_PATTERNS"])}

def kv_gb(pb: float, ctx: int, kv_frac: float, cfg: dict, kv_type: str = "q8_0") -> float:
    """Context cache for ctx tokens, from the size class's layer layout (ARCH_TABLE): 2 x layers x KV heads x head size x bytes per element, times the share of layers that keep one."""
    lo_hi = next((r for r in cfg["ARCH_TABLE"] if r[0] <= pb < r[1]), cfg["ARCH_TABLE"][-1])
    return 2 * lo_hi[2] * lo_hi[5] * (lo_hi[3] // lo_hi[4]) * cfg["KV_TYPES"][kv_type] * kv_frac * ctx / 1e9

def prefilter(raw: list, f: dict, cfg: dict) -> tuple:
    """(kept models, counts of why the others went): the name words typed (all in the repository's name, letters and digits compared, in any order - or, when no model has them all, the models with the most of them), exclude, must contain all, any of (over id and tags), parameter range, quants at or above the lowest quality allowed and of the quant words typed. Each kept model gains params_b and quants (cheapest first). counts["named"]: (words matched, words typed)."""
    words = lambda k: [w.strip().lower() for w in str(f.get(k) or "").split(",") if w.strip()]
    ex, must, any_ = words("exclude"), words("must"), words("any")
    names, qwant = words_of(str(f.get("words") or ""))
    floor = cfg["QUANTS"].get(f.get("min_quant") or "Q2_K", {"quality": 0})["quality"]
    seen, kept, counts = set(), [], {"name": 0, "exclude": 0, "words": 0, "params": 0, "quant": 0}
    hits = lambda m: sum(_alnum(w) in _alnum(m.get("id", "")) for w in names)
    best = max([hits(m) for m in raw] or [0])
    counts["named"] = (best, len(names))
    for m in raw:
        if m.get("id") in seen: continue
        seen.add(m.get("id"))
        if names and (not best or hits(m) < best): counts["name"] += 1; continue
        blob = f"""{m.get("id", "")} {" ".join(m.get("tags") or [])}""".lower()
        if ex and any(w in blob for w in ex): counts["exclude"] += 1; continue
        if (must and not all(w in blob for w in must)) or (any_ and not any(w in blob for w in any_)): counts["words"] += 1; continue
        pb = params_b(m)
        if pb is None or not (float(f.get("min_params") or 0) <= pb <= float(f.get("max_params") or 1e9)): counts["params"] += 1; continue
        qs = sorted({q for s in m.get("siblings") or [] if s.get("rfilename", "").endswith(".gguf") and "mmproj" not in s["rfilename"].lower() for q in [quant_of(s["rfilename"]).replace("UD-", "")] if q in cfg["QUANTS"] and cfg["QUANTS"][q]["quality"] >= floor and quant_wanted(q, qwant)}, key=lambda q: cfg["QUANTS"][q]["bpp"])
        if not qs: counts["quant"] += 1; continue
        kept.append({**m, "params_b": pb, "quants": qs})
    return kept, counts

def _norm_bench(raw: str):
    r = raw.lower().strip().replace("-", "_").replace(" ", "_")
    return _BENCH_ALIASES.get(r) or next((v for k, v in _BENCH_ALIASES.items() if k.replace(" ", "_") in r), None)

def card_scores(detail: dict) -> dict:
    """Benchmark scores a model's card states (its model-index), 0-100, by benchmark."""
    out = {}
    for e in (detail.get("cardData") or {}).get("model-index") or []:
        for r in e.get("results") or []:
            for mt in r.get("metrics") or []:
                try: v = float(str(mt.get("value")).replace("%", "").strip())
                except ValueError: continue
                v = v * 100 if 0 < v <= 1 else v
                if (b := _norm_bench(str(mt.get("name") or mt.get("type") or ""))) and 0 < v <= 100: out[b] = max(out.get(b, 0), v)
    return out

def known_scores(mid: str, base: str, cfg: dict) -> dict:
    """Curated scores (KNOWN_BENCHMARKS) for the repository or its base model, by id or short name."""
    kb = cfg["KNOWN_BENCHMARKS"]
    short = {k.split("/")[-1].lower(): k for k in kb}
    return dict(next((kb[k] for k in (mid, base) if k in kb), None) or next((kb[short[n]] for n in (mid.split("/")[-1].lower(), (base or "").split("/")[-1].lower()) if n in short), {}))

def _wavg(scores: dict, w: dict) -> float: return round(sum(float(scores[b]) * x for b, x in w.items() if scores.get(b)) / max(sum(x for b, x in w.items() if scores.get(b)), 0.001), 2) if any(scores.get(b) for b in w) else 0.0

def speed_score(tps: float, target: float, curve: str = "cap") -> float:
    if not tps or tps <= 0: return 0.0
    return tps / (tps + target) if curve == "open" else min(tps / target, target / tps) if curve == "peak" else min(tps / max(target * 2.0, 1.0), 1.0)

def rank(models: list, f: dict, machine: dict, cfg: dict, general: dict = None) -> list:
    """One row per model and quant that fits the machine with the context asked for, scored 0-1 on each of SCORE_DIMS and weighted (the old finder's method). A file's size is the real one once the model's page was read (model["files"]), else parameters x bytes per parameter. machine: {"ram_gb", "rate"} (rate: effective bytes/s while decoding, 0 when unknown: then speed is not scored and min_tps not applied)."""
    ctx, target, wts, rows = int(f.get("ctx") or 8192), float(f.get("target_tps") or 5), f.get("weights") or cfg["DEFAULT_WEIGHTS"], []
    for m in models:
        t, sizes = traits(m, cfg), {(fl["quant"] or "").replace("UD-", ""): fl["bytes"] for fl in m.get("files") or [] if not fl.get("projector")}
        scores = {**known_scores(m["id"], m.get("base", ""), cfg), **(m.get("card") or {})}
        g = scale_of([m.get("base"), m["id"]], general or {})[0]
        for q in m["quants"]:
            size = sizes.get(q) or m["params_b"] * 1e9 * cfg["QUANTS"][q]["bpp"]
            x = fit(int(size), machine.get("ram_gb") or 0, machine.get("rate") or 0, kv_gb=kv_gb(m["params_b"], ctx, t["kv_frac"], cfg))
            if not x["fits"] or (machine.get("rate") and x["tps"] is not None and x["tps"] < float(f.get("min_tps") or 0)): continue
            sub = {"lb_avg": min(_wavg(scores, _AVG_W) / 100, 1.0), "quant_qual": cfg["QUANTS"][q]["quality"], "speed": speed_score(x["tps"] or 0, target, f.get("curve") or "cap"), "mem_head": max(0.0, (machine["ram_gb"] - x["need_gb"]) / machine["ram_gb"]), "ctx_fit": 1.0,
                   "popularity": min(max(m.get("likes", 0) / 50.0, (m.get("downloads") or 0) / 5000.0) / 16.0, 1.0), "intel": _wavg(scores, _INTEL_W) / 100, "general": 1 / (1 + math.exp(-g)) if g is not None else 0.0}
            total = sum(sub[d] * float(wts.get(d, cfg["DEFAULT_WEIGHTS"].get(d, 1))) for d in cfg["SCORE_DIMS"]) / max(sum(float(wts.get(d, cfg["DEFAULT_WEIGHTS"].get(d, 1))) for d in cfg["SCORE_DIMS"]), 0.001)
            rows.append({"id": m["id"], "quant": q, "gb": round(size / 1e9, 1), "exact": q in sizes, "tps": x["tps"], "need_gb": x["need_gb"], "sub": {k: round(v, 3) for k, v in sub.items()}, "total": round(total, 4), "general": g, "benches": scores, "params_b": m["params_b"], "mtp": t["mtp"]})
    return sorted(rows, key=lambda r: -r["total"])

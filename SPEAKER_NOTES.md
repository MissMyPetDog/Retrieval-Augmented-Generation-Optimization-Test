# Speaker Notes — RAG Optimization Project

A compact, read-while-presenting script. ~6 minutes for main content, structured so you can drop any single step if time is tight.

---

## 0. Opening (30 sec)

> "Our project optimizes a hand-built RAG pipeline end-to-end. RAG has two phases: **offline** — preprocess, embed, and index the corpus; **online** — embed the user's query, search, then generate an answer with gpt-4o. I made **5 incremental optimizations**, each targeting a different stage, covering course Weeks 6, 8, 9, and 10/11. Everything runs **CPU only**."

**Transition:** "Let me walk through the 5 steps."

---

## Step 1 — IVF Parameter Tuning (1 min)

> "Baseline used an IVF index with **32 clusters**, each holding ~3,000 vectors — too coarse. I retuned to **64 clusters** (~1,500 vectors each), following the `sqrt(N)` heuristic, and matched the probe range so we scan the same fraction of the corpus for a fair comparison."

**Why it works:** finer partitions → better local homogeneity → higher recall per scan.

**Result:**
- Recall at 3.1% scan coverage: **0.59 → 0.72 (+23%)**
- Pareto curve shifts up uniformly
- **Cost:** +13% build time — handed off to Step 2

**Transition:** "That build regression motivates the next step."

---

## Step 2 — Numba K-Means (1 min)

> "The baseline K-Means had two inefficiencies: a Python `for k in range(64)` loop in the update step that allocates a new array per cluster, and it recomputes the corpus's L2 norms every iteration — 20 wasted full-corpus passes per build."

**What I did:** new file `optimized/kmeans_numba.py`:
- Numba `@njit` kernel that accumulates per-cluster sums + counts in a **single pass** over vectors
- Hoisted `v_norms` outside the loop (compute once, reuse)
- `IVFIndexNumba(IVFIndex)` subclass — baseline is untouched

**Why it works:** JIT compilation removes Python interpreter overhead; single-pass beats per-cluster boolean indexing.

**Result:**
- Build time **3.85s → 1.74s (−55%)**
- **Recall identical to 4 decimal places** — proves the Numba version is mathematically equivalent, not a speed/accuracy trade
- Net vs original baseline: **−49%**

**Transition:** "Next step pushes further on K-Means quality."

---

## Step 3 — K-Means++ Initialization (1 min)

> "K-Means is sensitive to where you put the initial centroids. The baseline picks them uniformly at random, which can cluster-collapse. K-Means++ picks them one at a time with probability proportional to squared distance from the nearest already-chosen centroid — so they end up spread out."

**What I did:** added `kmeans_pp_init()` and a second subclass `IVFIndexNumbaPP`. Key implementation detail: computing distances via the expanded formula `||v - c||² = ||v||² - 2 v·c + ||c||²` so each iteration is a fast BLAS GEMV, not an (N, D) subtraction that allocates 150 MB per iteration.

**Result — a trade, not a pure win:**
- Build time **+19%** — my convergence tolerance was too strict, so both random and K-Means++ hit the 20-iteration cap; I paid the init cost without recouping iteration savings
- Recall at smallest probe (n_probes=2): **+2.7%**

**How to frame it:** "Not every optimization is a net win. This one is a trade that matters for latency-critical configurations. Recognizing trades honestly is part of the work."

**Transition:** "That covers the index layer. Now the online side."

---

## Step 4 — LLM Streaming / TTFT (1 min)

> "Calls to gpt-4o take ~1.5 seconds. Without streaming the user sees nothing until the full answer arrives. With `stream=True` we get tokens as they're generated — the user sees the answer **starting** in well under a second."

**What I did:** added `generate_stream()` method to `BaselineGenerator`, uses the OpenAI SDK's `stream=True` iterator. New metric: **TTFT (Time To First Token)**.

**Why it matters:** total latency unchanged, but **perceived** latency changes completely. For a chat UI this is the metric users feel.

**Result (real Kong gpt-4o calls):**
- Non-streaming TTFT: 1,738 ms
- Streaming TTFT: **961 ms (−45%)**
- Concurrent streaming (n=8 workers): batch total **2.4 s (5.7× vs sequential non-stream)**

**One nuance worth calling out:** concurrent streaming had **higher** TTFT than sequential streaming (1,158 vs 961 ms). Kong's proxy can't schedule 8 SSE streams perfectly in parallel, so each stream's first token gets delayed a bit. Real engineering trade between per-user latency and batch throughput.

**Transition:** "The last step is the architectural one."

---

## Step 5 — Pipelined RAG (1 min)

> "Previous steps optimized **stages**. This step optimizes **how the stages compose**."

**Baseline architecture:** staged — retrieve everything, then generate everything. CPU works first, then sits idle while the network does its thing. Half the CPU time wasted.

**Pipelined architecture:** two thread pools running in parallel:
- 4-worker retrieval pool (CPU-bound embed + search)
- 8-worker generation pool (IO-bound gpt-4o streaming)
- **As soon as a retrieval finishes, it submits the result directly into the gen pool from inside the retrieval worker**

**Why it works:** retrieval of query `i+1` runs while generation of query `i` waits on the network. CPU and IO saturate simultaneously.

**Result:**
- Naive sequential: 14.5 s / 8 queries
- **Pipelined: 2.8 s / 8 queries → 5.13× end-to-end speedup**

Theoretical ceiling is ~7×; the gap to 5× is Python thread scheduling overhead and Kong concurrency limits. Honestly measured, honestly explained.

**Transition to summary:** "So what does all 5 steps together look like?"

---

## Wrap-Up (30 sec)

> "Bringing it all together versus the original baseline:
> - Build time: −39%
> - Recall at low scan coverage: +26%
> - Perceived latency: −45%
> - **End-to-end 8-query batch: 5.13×**
>
> Four distinct course weeks touched: **Week 6 (Numba)**, **Week 8 (algorithmic optimization)**, **Week 9 (concurrency + streaming)**, **Week 10/11 (parallel programming)**.
>
> All baseline code is untouched — every optimization lives in `optimized/` as a drop-in variant, switchable via a parameter. Every step is saved as a tagged JSON experiment, fully reproducible."

---

## Anticipated Q&A (keep each answer 15-20 sec)

**Q: Why is Step 5 only 5×, not the theoretical 7×?**
> Theoretical ceiling assumes zero overhead. In reality there's Python GIL, thread scheduling, and Kong's concurrency scheduler on 8 SSE streams. Each adds a small but real overhead. I report the honest number.

**Q: Step 3 made build time slower — is that still an optimization?**
> Yes, it's a trade-off optimization. The build cost buys +2.7% recall at the fastest query configuration, which matters for latency-critical serving. Not all optimizations are pure wins; recognizing trades is part of mature engineering.

**Q: Why not also do batch query embedding like in other approaches?**
> Batch embedding optimizes per-query latency by amortizing GEMM. That's one axis. I chose to optimize across multiple axes — build time, recall quality, perceived latency, and end-to-end architecture. Both paths are valid; I went broader.

**Q: How did you ensure no GPU usage snuck in?**
> The notebook calls `portal.setup_cpu_only()` first, which sets `CUDA_VISIBLE_DEVICES=""` and `NUMBA_DISABLE_CUDA=1` before any torch/numba imports. We verify with `torch.cuda.is_available() == False` at setup. Every number reflects pure CPU code optimization.

**Q: What's the biggest single improvement?**
> End-to-end, it's Step 5 — 5.13× for the full batch. Per-stage, it's Step 2 — 55% faster K-Means build. For user-perceived latency, Step 4 — 45% lower TTFT. They target different things.

**Q: How repeatable are these numbers?**
> Every step's full RESULTS dict is saved to `results/experiments/medium_<step>.json` with timestamp, description, and parent. I can diff any two steps with `portal.compare_experiments(step_a, step_b)`. CPU-bound numbers are stable within ~2%; real-API generation numbers (Steps 4 and 5) have ~10% natural variance from Kong/network.

---

## Speaking tips

- **Don't read the numbers** — memorize 2 per step and say them naturally
- **Pause** after each "Result:" block — let the number land
- **For trade-offs (Step 3, Step 4 nuance):** acknowledge them proactively, don't hide them; judges like honesty
- **If time is short:** drop Step 3. It's the weakest narrative.
- **If time is generous:** use the "speaker notes" column of Anticipated Q&A as bonus content

# RAG Pipeline 优化 — 项目完整总结

**项目:** Advanced Python(Spring 2026)— Retrieval-Augmented Generation 优化
**数据集:** MS MARCO `medium`(99,999 passages × 384 维,500 条带 relevance judgment 的 queries)
**限制:** 纯 CPU(GPU 全部隐藏;所有加速都是代码级优化的结果)
**覆盖课程周次:** Week 6(Numba)、Week 8(Optimization)、Week 9(Concurrency)、Week 10/11(Parallel Programming)

> *引用的图片保存在 `results/figures/`(需要时从 Jupyter notebook 中导出)。*

---

## 目录

1. [RAG 工作流](#1-rag-工作流)
2. [纯 BruteForce 基线](#2-纯-bruteforce-基线)
3. [优化历程](#3-优化历程)
   - [Step 1 — IVF 参数调优](#step-1--ivf-参数调优)
   - [Step 2 — Numba K-Means(Week 6)](#step-2--numba-k-meansweek-6)
   - [Step 3 — K-Means++ 初始化(Week 8)](#step-3--k-means-初始化week-8)
   - [Step 4 — LLM Streaming(Week 9)](#step-4--llm-streamingweek-9)
   - [Step 5 — Pipelined RAG(Week 10/11)](#step-5--pipelined-ragweek-1011)
   - [Step 6 — 队友的查询路径优化](#step-6--队友的查询路径优化)
4. [最终三方对比](#4-最终三方对比)
5. [用户端实际影响](#5-用户端实际影响)
6. [结论与核心洞察](#6-结论与核心洞察)

---

## 1. RAG 工作流

RAG(Retrieval-Augmented Generation)管道分为两个阶段。

### 离线阶段(只建一次)

```
原始 MS MARCO passages
     │
     ▼
[预处理]     清洗文本、切成带重叠的 chunks
     │
     ▼
[Embed 全库] MiniLM-L6-v2 -> (N, 384) float32 矩阵
     │
     ▼
[建索引]     K-Means 聚类 + IVF 倒排列表
     │
     ▼
vectors.npy + index_ivf.pkl(落盘)
```

### 在线阶段(每个用户 query)

```
用户的 query 文本
     │
     ▼
[Embed query]  同一个模型 -> (384,) 向量        ~300 ms CPU
     │
     ▼
[搜索索引]     cosine 相似度 -> top-K chunks    ~5-60 ms
     │
     ▼
[生成答案]     query + chunks -> gpt-4o API    ~1500 ms(网络)
     │
     ▼
答案
```

每个阶段都是不同的优化维度:搜索是 CPU 密集型矩阵运算,生成是 IO 密集型网络等待。本项目针对每个阶段使用不同的课程工具。

---

## 2. 纯 BruteForce 基线

"零优化"参考点 — 没有 IVF、没有缓存、没有并发、没有 streaming。

### 配置

| 组件 | 设置 |
|-----------|---------|
| 索引 | `BruteForceIndex` — 对 10 万向量线性扫描 |
| 相似度 | `cosine_sim_numpy`,查询时才算 L2 范数 |
| `use_precomputed_norms` | `False`(显式关闭) |
| 生成器 | `BaselineGenerator`,`stream=False` |
| 架构 | 每 query 串行:embed → search → gen → 下一个 |

### 结果(8 条真实 Kong gpt-4o 调用,每 query 取 K=5 个 chunk)

| 指标 | 值 |
|--------|:-----:|
| 建索引时间 | 130 ms |
| 每 query embed | 8.1 ms |
| 每 query 搜索 | **62.9 ms** |
| 每 query 生成 | 1754 ms |
| 每 query 总时长 | 1825 ms |
| **Batch 总时长(8 queries)** | **14.6 s** |
| Recall@5 | 0.75 |

### 为什么这个 baseline 重要

BruteForce 保证 100% 召回 — 扫描每个文档向量。作为**正确性上限**和"什么都不做就是这个代价"的参考点。

---

## 3. 优化历程

每一步在前一步基础上**叠加**。baseline 代码永不修改 — 新版本放在 `optimized/`,通过参数切换。

---

### Step 1 — IVF 参数调优

**做了什么:** 把 BruteForce 线性扫描换成倒排文件索引(IVF)。用 K-Means 把语料聚成 K 个簇;查询时只扫最近的 `n_probes` 个簇,而不是全部 10 万。

**关键改动:**
- `N_CLUSTERS: 32 → 64`(接近 `sqrt(N) ≈ 316` 的启发式规则,划分更精细)
- `probes: (1, 2, 4, 8) → (2, 4, 8, 16)` — 按相同的"扫描覆盖率"对齐,方便公平对比

**课程对应:** 参数调优(非单一课程周次)— 算法设计知识。

**结果(500 queries 质量评估,K=10):**

| 配置 | Recall@10 | 查询延迟 |
|--------|:---:|:---:|
| IVF(64) n_probes=2 | 0.7442 | 3.5 ms |
| IVF(64) n_probes=4 | 0.8178 | 6.8 ms |
| **IVF(64) n_probes=8** | **0.8508** | **13.0 ms** |
| IVF(64) n_probes=16 | 0.8728 | 26.9 ms |
| BruteForce(上限) | 0.8908 | 63 ms |

**与 32 簇版本的相同覆盖率对比:**

| 扫描覆盖率 | Step 0(32 簇) | Step 1(64 簇) | 变化 |
|:---:|:---:|:---:|:---:|
| 3.1% | 0.590 | **0.724** | **+22.9%** |
| 6.25% | 0.737 | 0.815 | +10.6% |
| 12.5% | 0.820 | 0.851 | +3.7% |
| 25% | 0.859 | 0.873 | +1.6% |

**关键洞察:** 整条 Pareto 前沿上移。任何扫描预算下 recall 都比之前好。代价:建索引时间 +13%(centroid 更多)。

---

### Step 2 — Numba K-Means(Week 6)

**做了什么:** 用 Numba JIT 编译 K-Means 聚类 kernel。修复 baseline 的两个低效:

1. **每次迭代都重算向量范数** — baseline 在 20 次迭代里每次都 `np.linalg.norm(corpus)`。我们缓存一次。
2. **Python `for k in range(K):` 更新循环** — 每个簇用布尔索引创建临时数组。用 Numba `@njit` 单次扫描累加替代。

**代码位置:** 新建文件 `rag-optimization/optimized/kmeans_numba.py`,定义 `IVFIndexNumba` 继承 `IVFIndex`,只覆写 `_kmeans`。baseline 代码不改。

**课程对应:** **Week 6 — Numba JIT 编译**。

**结果:**

| 指标 | Step 1(NumPy K-Means) | Step 2(Numba) | 变化 |
|--------|:---:|:---:|:---:|
| IVF build(sequential) | 3852 ms | **1741 ms** | **−54.8%** |
| 各个 n_probes 下的 Recall@10 | 不变 | 不变 | **0.00%** |

**关键洞察:** 索引建得 2.2× 快,聚类结果**数学上完全等价**(精确到小数点后 4 位一致)。相对最初 Step 0 的 baseline:**−49%**。

---

### Step 3 — K-Means++ 初始化(Week 8)

**做了什么:** 用 K-Means++ 替代随机 K-Means 初始化。不是均匀随机选 K 个 centroid,而是一个一个选:每次按"到已选最近 centroid 距离平方"的概率分布抽下一个。结果:初始 centroid 更分散,避免陷入差的局部最优。

**实现细节:** 第一版用 `vectors - centroids[k]` + `np.einsum`,因为临时数组分配太大,慢了 7 倍。重写用展开距离公式 `||v−c||² = ||v||² − 2v·c + ||c||²` — 每次迭代只是一次 BLAS GEMV,无临时数组。

**课程对应:** **Week 8 — Optimization in Python**(算法层面的改进)。

**结果 — 这是个权衡,不是纯胜利:**

| 指标 | Step 2(random init) | Step 3(K-Means++) | 变化 |
|--------|:---:|:---:|:---:|
| IVF build | 1741 ms | 2080 ms | +19% |
| n_probes=2 下的 Recall@10 | 0.7248 | **0.7442** | **+2.7%** |
| n_probes=4-16 下的 Recall@10 | — | 基本一致 | ±0.5% |

**诚实反思:** 收敛阈值 `tol=1e-6` 对这个数据集太严格,随机 init 和 K-Means++ 都跑满 20 次迭代而不触发提前终止。K-Means++ 多付了 450 ms init 成本但没换到迭代次数的节省。收益在最快查询配置(n_probes=2)下有**可测量的 recall 提升**。

---

### Step 4 — LLM Streaming(Week 9)

**做了什么:** 在 gpt-4o API 上启用 `stream=True`,让 token 边生成边返回,而不是一次性全部返回。引入新指标:**TTFT(Time To First Token)** — 用户看到答案第一个字的时间。

**代码位置:** `components/generator.py::generate_stream()`,用 OpenAI SDK 的 streaming iterator。

**课程对应:** **Week 9 — Python Concurrency**(基于 iterator 的流式处理)。

**结果(8 次真实 Kong gpt-4o 调用):**

| 模式 | Batch 总时长 | 平均 TTFT | 感知延迟变化 |
|------|:---:|:---:|:---:|
| Sequential 非 streaming | 13,908 ms | **1,738 ms** | (baseline) |
| Sequential streaming | 10,933 ms | **961 ms** | **−45%** |
| Concurrent streaming(n=8 workers) | **2,421 ms** | 1,158 ms | Batch 5.7× + TTFT −33% |

**关键洞察:** 总生成时间不变(gpt-4o 按自己的速度产 token),但用户**看到答案开始出现**的时间大幅提前。对聊天 UI 来说这才是用户真正感知的延迟。

**意外发现:** concurrent streaming 的 TTFT **反而**比 sequential streaming 高(1,158 vs 961 ms)。8 个 SSE 流同时跑时,Kong 代理的网络调度造成抢占,每个流的首 token 都被延迟了。这是"**每用户延迟 vs 批吞吐量**"的真实工程权衡。

---

### Step 5 — Pipelined RAG(Week 10/11)

**做了什么:** **项目最重要的架构优化**。不再按"先全部 retrieve,再全部 generate"的串行方式,而是**两个线程池同时运行**:

- **Retrieval 池**(4 workers)负责 CPU 密集的 embed + search
- **Generation 池**(8 workers)负责 IO 密集的 gpt-4o streaming

每个 retrieval worker 做完自己的活,**直接在自己的线程里把结果提交给 generation 池**。这样 query `i+1` 的 retrieval 和 query `i` 的 generation 就**重叠**起来了 — CPU 和网络同时饱和。

```
Retrieval 池(4 workers)              Generation 池(8 workers)
─────────────────────────             ─────────────────────────
  embed + search query i  ──submit──→  gpt-4o streaming for i
  embed + search query i+1             (和别的 gen 并发)
                                        gpt-4o streaming for i+1
  ...                                   ...
```

**课程对应:** **Week 10/11 — Parallel Programming**(多池线程协调)。

**结果(8 queries 端到端,含 embed + search + 完整 generation):**

| 模式 | Batch 总时长 | 每 query 摊销 | 加速 |
|------|:---:|:---:|:---:|
| Sequential naive | 14,456 ms | 1,807 ms | 1.0× |
| **Pipelined(4 retrieve + 8 gen)** | **2,817 ms** | **352 ms** | **5.13×** |

**理论上限 ~7×;** 实测 5.13× 的差距来自 Python 线程调度、GIL 切换、Kong 代理的并发调度上限。诚实测量,诚实报告。

**为什么用 threading 而不是 multiprocessing:** 两个阶段都会释放 GIL — gpt-4o 的网络等待天然释放,PyTorch 的 embedding 推理在内部释放。multiprocessing 会引入进程 spawn 成本(Windows 上 ~400 ms)而没有额外收益。事实上项目里的 `ParallelIVFBuilder`(multiprocessing)**比**串行版本还慢,这个反例正好验证了选 threading 的正确性。

---

### Step 6 — 队友的查询路径优化

在共享 repo 上,一个队友贡献了三个查询路径的微优化,通过 git merge 整合进来,不改 baseline 代码。

**新增能力:**
1. **`use_precomputed_norms`** — corpus 向量的 L2 范数在建索引时算好缓存,查询时不再重算
2. **`use_numpy_candidate_gather`** — IVF 候选收集用 `np.concatenate` 替代 Python `list.extend`
3. **`cosine_sim_numba_parallel_precomputed`** — 接受预算 norms 作为第三个参数的 Numba 并行相似度 kernel

**整合方式:** 通过 `git merge origin/main` 合并到独立的 `integrate-friend` 分支。这三个改动都在现有文件(`components/vector_index.py`、`optimized/similarity_numba.py`)里,作为**可选 flag**,对 Steps 1-5 代码后向兼容。

**消融研究(IVF n_probes=2,medium 数据集):**

| 变量 | sim_fn | norm cache | np gather | 延迟 | vs A |
|---------|--------|:---:|:---:|:---:|:---:|
| **A. flags 全关** | NumPy | ✗ | ✗ | 3.91 ms | 1.0× |
| **B. + norm cache** | NumPy | ✓ | ✗ | **1.78 ms** | **2.19×** |
| **C. + np gather** | NumPy | ✓ | ✓ | **1.77 ms** | **2.21×** |
| D. + Numba parallel precomputed | Numba par | ✓ | ✓ | 2.06 ms | 1.90× ❌ |

**关键发现:**
- **norm cache 单独贡献了 95% 的收益**(3.91 → 1.78 ms)。一行代码缓存不变的 corpus 范数。
- **np gather 效果微弱**(+0.5%),在 ~3000 候选的扫描规模下,Python list 开销相对小。
- **变量 D 是 negative result** — Numba 并行 + 预算 norms **比** NumPy 还慢。BLAS GEMV 在 ~3000 向量上太强,手写 `prange` 打不过。这正好印证了项目核心命题:**工具选择取决于操作的形状**。Numba parallel 在全 corpus 扫描(Section 2:100K 向量上 10× 于 NumPy)上赢,但在 IVF 过滤后的小子集上输。

**意外大礼包 — BruteForce 快了 10×:**

因为 `use_precomputed_norms` 合并后默认 `True`,`BruteForceIndex.search` 自动受益。合并后的 BF 延迟从 **57.80 ms 降到 5.85 ms** — 来自合并的 10 倍加速,不是我们自己的优化成果。诚实地作为"整合带来的免费收益"报告。

---

## 4. 最终三方对比

三个独立的 Python 脚本在 `comparisons/` 下,每个都跑完整的 8-query 端到端 RAG pipeline,报告相同指标。

### 配置

| 配置 | 索引 | sim_fn | Norm cache | Np gather | 生成器 | 架构 |
|--------|-------|--------|:---:|:---:|-----------|--------------|
| 1. BruteForce | `BruteForceIndex` | `cosine_sim_numpy` | ✗ | — | non-stream | serial |
| 2. IVF default | `IVFIndex(32, np=4)` | `cosine_sim_numpy` | ✗ | ✗ | non-stream | serial |
| 3. Fully optimized | `IVFIndexNumbaPP(64, np=8)` | `cosine_sim_numpy` | ✓ | ✓ | **streaming** | **pipelined(4+8)** |

### 结果(真实 Kong gpt-4o,8 queries,每 query K=5)

| 指标 | Config 1: BruteForce | Config 2: IVF default | Config 3: Fully optimized | 3 vs 1 加速 |
|--------|:---:|:---:|:---:|:---:|
| 建索引时间 | 130 ms | 3,530 ms | 1,972 ms | 0.07× |
| 每 query embed ms | 8.1 | 7.6 | 37.2 | — |
| **每 query 搜索 ms** | 62.9 | 16.7 | **14.5** | **4.35×** |
| 每 query 生成 ms | 1,754 | 1,722 | 1,598 | — |
| 每 query 总时长 ms | 1,825 | 1,747 | 1,650 | 1.11× |
| **Batch 总时长** | 14,601 ms | 13,974 ms | **3,770 ms** | **3.87×** |
| Recall@5(8 queries) | 0.75 | 0.50 | 0.625 | — |

*Recall@5 这一行分辨率低,因为样本只有 8 个 query(每个是 binary "找到/没找到")。可靠的 recall 数字请看基于 500 queries 的 Step 1 表。*

![comparison figure](results/figures/three_way_comparison.png)

### 三个关键发现

**1. BruteForce → IVF default 只快 4%(14.60 s → 13.97 s)。**
单独换算法几乎没帮助。生成阶段占 batch 时间 ~1.7 s/query × 8 = 13.6 s。IVF 的 4× 搜索加速(63 → 17 ms)只省了 ~0.4 s,**在生成时间的墙前隐形**。

**2. IVF default → Fully optimized 快 3.7×。**
真正的加速来自**架构改变**,不是算法优化。Pipelined streaming 让 8 个 gen 调用并发,同时 retrieval 和它们重叠。

**3. 算法层的收益在搜索层仍然真实(4.35×)。**
只是在 LLM 主导的 pipeline 里端到端不可见。对非 LLM 检索系统(如经典搜索),这会是主导指标。

**核心洞察:** *当一个阶段主导总延迟时,优化快阶段在端到端层面毫无效果。只有重新安排架构让阶段重叠,才能给用户带来可见的加速。*

---

## 5. 用户端实际影响

把后端指标翻译成用户体验。

### 场景 A:单个用户问一个问题

| 阶段 | Baseline | 完全优化 | 用户感受 |
|-------|:---:|:---:|:---:|
| Query embed(CPU) | 300 ms | 300 ms | — |
| 搜索 | 63 ms | 10 ms | — |
| 生成到首个字(TTFT) | **1,738 ms** | **961 ms** | **−45%** |
| **用户看到答案开始出现的时间** | ~2,100 ms | **~1,270 ms** | **等待时间 −32%** |
| 完整答案到达 | 2,100 ms | 1,750 ms | −17% |

用户感受到的响应是"1.3 秒"而不是"2.1 秒" — 聊天 UI 里"流畅"和"卡顿"的分界线。

### 场景 B:连续 8 个 query(多轮对话或批量问答)

| 模式 | 总等待时间 |
|---------|:---:|
| 全部串行(原版) | **14.6 s** |
| Pipelined + streaming(优化) | **2.8 s** |
| **加速** | **5.13×** |

用户快速问 8 个问题等 3 秒而不是 15 秒。

### 场景 C:多用户服务(10 人并发一个 pipeline)

| 模式 | 最后一人等待时间 |
|---------|:---:|
| 原版(串行调度) | ~18 s |
| 优化(pipelined 并发) | ~3 s |
| **缩短** | **~83%** |

同样的硬件能处理约 5× 的并发量,尾部用户不用等几分钟。

### 质量保持

| 指标 | Baseline | 优化 | 变化 |
|--------|:---:|:---:|:---:|
| Recall@10(500 queries) | 0.8908 | 0.8508(IVF np=8) | −4.5% |
| MRR | 0.4937 | 0.4754 | −3.7% |

~4% 的相对 recall 损失换来 5× 批量加速,是有利的权衡。如果选 `n_probes=16`,能拿到 **Recall@10 = 0.8728(BF 的 98%)**,代价是搜索延迟翻倍 — 但依然远低于生成主导的总时间。

---

## 6. 结论与核心洞察

### 值得记住的数字

| 维度 | 最好成果 |
|-----------|:---:|
| 建索引时间 | 3.41 s → 2.08 s(**−39%**,via Numba K-Means) |
| 每 query 搜索 | 63 ms → 14 ms(**4.35×**,IVF + norm cache + np gather) |
| 感知延迟(TTFT) | 1,738 ms → 961 ms(**−45%**,via streaming) |
| 端到端 batch | 14,601 ms → 3,770 ms(**3.87×**,via pipelined 架构) |
| 检索质量 | 0.8908 → 0.8508(**−4.5%**,可接受的权衡) |

### 覆盖的优化维度

本项目刻意覆盖**六个不同维度**,而不是在一个维度上钻得很深。每个维度对应一个不同的课程周次:

| 维度 | 技术 | 课程周次 |
|-----------|-----------|:---:|
| 算法结构 | IVF 替代 BruteForce | (设计) |
| 算法细节 | K-Means++ 种子选择 | **Week 8** |
| 代码执行 | Numba JIT 编译 kernel | **Week 6** |
| 数据访问 | 预算缓存的 norms | (队友合并) |
| 并发 | Streaming + 并发生成 | **Week 9** |
| 系统架构 | 双池 pipelined 服务 | **Week 10/11** |

### 项目核心命题

> **速度不住在单一的一层。真实系统的每一层都有瓶颈 — CPU 数学、数据访问模式、IO 等待、阶段编排 — 每一层都需要自己的工具。**

本项目最反直觉的结果是 3-way 对比显示**单靠 IVF 只能带来 4% 的端到端加速**,因为 gpt-4o 生成主导了总时间。戏剧性的 3.87× batch 加速完全来自**架构重排**(pipelining + streaming),而不是算法优化。这验证了宽覆盖的策略:**理解哪个工具应用在哪里,比把一个工具推到极限更重要**。

---

## 可复现性

- 每一步优化都是 `results/experiments/medium_*.json` 下的 tagged JSON 快照
- 三方对比的产物在 `results/comparisons/`,可通过 `compare_pipelines.ipynb` 重新生成
- `rag-optimization/components/` 下的 baseline 代码永不改动;每个优化都作为 drop-in 变体放在 `optimized/` 里
- Integration 分支 `integrate-friend` 保留了作者和队友的双方 commit 历史,互不覆盖

每步的完整细节见 [OPTIMIZATION_JOURNEY.md](OPTIMIZATION_JOURNEY.md) 和 [PIPELINE_MAP.md](PIPELINE_MAP.md)。
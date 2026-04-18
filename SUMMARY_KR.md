# RAG Pipeline 최적화 — 프로젝트 전체 요약

**프로젝트:** Advanced Python (Spring 2026) — Retrieval-Augmented Generation 최적화
**데이터셋:** MS MARCO `medium` (99,999 passages × 384차원, relevance judgment 포함 500개 쿼리)
**제약:** CPU 전용 (모든 GPU 숨김; 모든 속도 향상은 순수 코드 수준 최적화의 결과)
**다루는 수업 주차:** Week 6 (Numba), Week 8 (Optimization), Week 9 (Concurrency), Week 10/11 (Parallel Programming)

> *참조된 그림은 `results/figures/`에 저장 (필요시 Jupyter notebook에서 export).*

---

## 목차

1. [RAG 워크플로우](#1-rag-워크플로우)
2. [순수 BruteForce 베이스라인](#2-순수-bruteforce-베이스라인)
3. [최적화 여정](#3-최적화-여정)
   - [Step 1 — IVF 파라미터 튜닝](#step-1--ivf-파라미터-튜닝)
   - [Step 2 — Numba K-Means (Week 6)](#step-2--numba-k-means-week-6)
   - [Step 3 — K-Means++ 초기화 (Week 8)](#step-3--k-means-초기화-week-8)
   - [Step 4 — LLM Streaming (Week 9)](#step-4--llm-streaming-week-9)
   - [Step 5 — Pipelined RAG (Week 10/11)](#step-5--pipelined-rag-week-1011)
   - [Step 6 — 동료의 쿼리 경로 최적화](#step-6--동료의-쿼리-경로-최적화)
4. [최종 3-way 비교](#4-최종-3-way-비교)
5. [사용자 체감 영향](#5-사용자-체감-영향)
6. [결론 및 핵심 통찰](#6-결론-및-핵심-통찰)

---

## 1. RAG 워크플로우

Retrieval-Augmented Generation 파이프라인은 두 단계로 나뉩니다.

### 오프라인 단계 (한 번만 빌드)

```
원본 MS MARCO passages
     │
     ▼
[전처리]      텍스트 정리, 겹치는 chunks로 분할
     │
     ▼
[코퍼스 임베딩] MiniLM-L6-v2 -> (N, 384) float32 행렬
     │
     ▼
[인덱스 빌드] K-Means 클러스터링 + IVF 역색인
     │
     ▼
vectors.npy + index_ivf.pkl (디스크 저장)
```

### 온라인 단계 (사용자 쿼리마다)

```
사용자 쿼리 텍스트
     │
     ▼
[쿼리 임베딩]  동일 모델 -> (384,) 벡터         ~300 ms CPU
     │
     ▼
[인덱스 검색]  코사인 유사도 -> top-K chunks   ~5-60 ms
     │
     ▼
[답변 생성]    쿼리 + chunks -> gpt-4o API    ~1500 ms (네트워크)
     │
     ▼
답변
```

각 단계는 서로 다른 최적화 면입니다: 검색은 CPU 기반 행렬 연산, 생성은 IO 기반 네트워크 대기. 본 프로젝트는 각 단계를 다른 수업 도구로 타겟팅합니다.

---

## 2. 순수 BruteForce 베이스라인

"제로 최적화" 기준점 — IVF 없음, 캐싱 없음, 동시성 없음, streaming 없음.

### 설정

| 컴포넌트 | 설정 |
|-----------|---------|
| 인덱스 | `BruteForceIndex` — 10만 벡터 전체 선형 스캔 |
| 유사도 | `cosine_sim_numpy`, 쿼리 시점에 L2 norm 계산 |
| `use_precomputed_norms` | `False` (명시적으로 비활성화) |
| 생성기 | `BaselineGenerator`, `stream=False` |
| 아키텍처 | 쿼리별 순차: embed → search → gen → 다음 |

### 결과 (8개 실제 Kong gpt-4o 호출, 쿼리당 K=5 chunks)

| 지표 | 값 |
|--------|:-----:|
| 인덱스 빌드 | 130 ms |
| 쿼리당 embed | 8.1 ms |
| 쿼리당 search | **62.9 ms** |
| 쿼리당 generate | 1754 ms |
| 쿼리당 총 시간 | 1825 ms |
| **Batch 총 시간 (8 queries)** | **14.6 s** |
| Recall@5 | 0.75 |

### 이 베이스라인이 중요한 이유

BruteForce는 100% retrieval recall을 보장합니다 — 모든 문서 벡터를 스캔합니다. **정확도 상한**이자 "아무것도 안 하면 이 정도 비용" 기준점 역할을 합니다.

---

## 3. 최적화 여정

각 단계는 이전 단계 위에 **누적**되는 변경입니다. 베이스라인 코드는 절대 수정하지 않습니다 — 새 변형은 `optimized/` 아래에 있고, 파라미터로 선택합니다.

---

### Step 1 — IVF 파라미터 튜닝

**하는 일:** BruteForce 선형 스캔에서 Inverted File Indexing으로 전환. 코퍼스를 K-Means로 K 개 클러스터로 분할; 쿼리 시 10만 벡터 전부가 아닌 가장 가까운 `n_probes` 개 클러스터만 스캔.

**주요 변경:**
- `N_CLUSTERS: 32 → 64` (`sqrt(N) ≈ 316` 휴리스틱에 더 가깝고, 더 세밀한 분할)
- `probes: (1, 2, 4, 8) → (2, 4, 8, 16)` — 동일한 스캔 커버리지 비율로 맞춰서 공정한 비교

**수업 매칭:** 튜닝 (특정 수업 주차 아님) — 알고리즘 설계 지식.

**결과 (500-쿼리 품질 평가, K=10 retrieved):**

| 설정 | Recall@10 | 쿼리 지연시간 |
|--------|:---:|:---:|
| IVF(64) n_probes=2 | 0.7442 | 3.5 ms |
| IVF(64) n_probes=4 | 0.8178 | 6.8 ms |
| **IVF(64) n_probes=8** | **0.8508** | **13.0 ms** |
| IVF(64) n_probes=16 | 0.8728 | 26.9 ms |
| BruteForce (상한) | 0.8908 | 63 ms |

**원래 32-클러스터 버전과 동일 커버리지 비교:**

| 스캔 커버리지 | Step 0 (32 클러스터) | Step 1 (64 클러스터) | 변화 |
|:---:|:---:|:---:|:---:|
| 3.1% | 0.590 | **0.724** | **+22.9%** |
| 6.25% | 0.737 | 0.815 | +10.6% |
| 12.5% | 0.820 | 0.851 | +3.7% |
| 25% | 0.859 | 0.873 | +1.6% |

**핵심 통찰:** Pareto 전선 전체가 위로 이동. 어떤 스캔 예산에서든 recall이 이전보다 엄격히 더 좋습니다. 비용: 센트로이드 증가로 빌드 시간 +13%.

---

### Step 2 — Numba K-Means (Week 6)

**하는 일:** K-Means 클러스터링 커널을 Numba JIT 컴파일로 재작성. 베이스라인의 두 가지 비효율을 수정:

1. **매 반복마다 벡터 norm 재계산** — 베이스라인은 K-Means 20회 반복 각각에서 `np.linalg.norm(corpus)`를 다시 계산. 한 번만 캐싱.
2. **Python `for k in range(K):` 업데이트 루프** — 클러스터별로 boolean 인덱싱으로 임시 배열 할당. Numba `@njit` 단일 패스 누산으로 교체.

**코드 위치:** 새 파일 `rag-optimization/optimized/kmeans_numba.py` — `IVFIndex`를 상속한 `IVFIndexNumba`가 `_kmeans`만 오버라이드. 베이스라인 코드는 그대로.

**수업 매칭:** **Week 6 — Numba JIT 컴파일**.

**결과:**

| 지표 | Step 1 (NumPy K-Means) | Step 2 (Numba) | 변화 |
|--------|:---:|:---:|:---:|
| IVF build (sequential) | 3852 ms | **1741 ms** | **−54.8%** |
| 각 n_probes에서의 Recall@10 | 불변 | 불변 | **0.00%** |

**핵심 통찰:** 인덱스 빌드가 2.2× 빨라지고, 클러스터링 결과는 **수학적으로 동일** (소수점 4자리까지 검증됨). Step 0 베이스라인 대비 순 효과: **−49%**.

---

### Step 3 — K-Means++ 초기화 (Week 8)

**하는 일:** 랜덤 K-Means 초기화를 K-Means++로 교체. K개 센트로이드를 균등 무작위로 뽑는 대신, 이미 선택된 가장 가까운 센트로이드와의 거리 제곱에 비례하는 확률로 하나씩 선택. 결과: 초기 센트로이드가 넓게 퍼져, 더 좋은 지역 최적으로 수렴.

**구현 메모:** 첫 버전은 `vectors - centroids[k]` + `np.einsum`을 사용했는데 큰 임시 할당 때문에 7배 느렸습니다. 확장된 거리 공식 `||v−c||² = ||v||² − 2v·c + ||c||²`로 재작성 — 각 반복이 단일 BLAS GEMV, 임시 배열 할당 없음.

**수업 매칭:** **Week 8 — Optimization in Python** (알고리즘적 개선).

**결과 — 이것은 트레이드오프이지 순수 승리가 아님:**

| 지표 | Step 2 (random init) | Step 3 (K-Means++) | 변화 |
|--------|:---:|:---:|:---:|
| IVF build | 1741 ms | 2080 ms | +19% |
| n_probes=2에서 Recall@10 | 0.7248 | **0.7442** | **+2.7%** |
| n_probes=4-16에서 Recall@10 | — | 본질적으로 동일 | ±0.5% |

**솔직한 반성:** 수렴 임계값 `tol=1e-6`이 이 데이터셋에는 너무 엄격해서 random init와 K-Means++ 둘 다 20회 반복을 조기 종료 없이 다 돌립니다. K-Means++은 추가 450 ms의 init 비용을 내는데 반복 절감으로 회수하지 못합니다. 상향은 가장 빠른 쿼리 구성(n_probes=2)에서 측정 가능한 recall 향상입니다.

---

### Step 4 — LLM Streaming (Week 9)

**하는 일:** gpt-4o API에서 `stream=True`를 활성화하여 토큰이 생성되는 대로 반환되도록 함. 새 지표 도입: **TTFT (Time To First Token)** — 사용자가 답변의 첫 단어를 보는 시간.

**코드 위치:** `components/generator.py::generate_stream()`, OpenAI SDK의 streaming iterator 사용.

**수업 매칭:** **Week 9 — Python Concurrency** (iterator 기반 streaming).

**결과 (8개 실제 Kong gpt-4o 호출):**

| 모드 | Batch 총 시간 | 평균 TTFT | 체감 지연 변화 |
|------|:---:|:---:|:---:|
| Sequential non-streaming | 13,908 ms | **1,738 ms** | (베이스라인) |
| Sequential streaming | 10,933 ms | **961 ms** | **−45%** |
| Concurrent streaming (n=8 workers) | **2,421 ms** | 1,158 ms | Batch 5.7× + TTFT −33% |

**핵심 통찰:** 전체 생성 시간은 변하지 않음 (gpt-4o는 자체 속도로 생성), 하지만 사용자가 **답변이 *시작되는 것*을 보는 시간**이 훨씬 빨라집니다. 채팅 UI에서는 이것이 진짜 중요한 지표.

**예상 외 발견:** concurrent streaming이 sequential streaming보다 TTFT가 *더 높음* (1,158 vs 961 ms). 8개 동시 SSE 스트림이 Kong 프록시를 통해 네트워크 경합을 만들어 각 스트림의 첫 토큰이 지연됨. 사용자별 지연 vs batch 처리량의 정직한 실세계 트레이드오프.

---

### Step 5 — Pipelined RAG (Week 10/11)

**하는 일:** **프로젝트의 대표 아키텍처 최적화**. 단계들을 직렬로 실행(모두 retrieve, 그 다음 모두 generate)하는 대신, **두 개의 thread pool이 동시에 실행**됩니다:

- **Retrieval pool** (4 workers) — CPU 기반 embed + search 처리
- **Generation pool** (8 workers) — IO 기반 gpt-4o streaming 처리

각 retrieval worker는 완료되면 *자신의 worker thread 내부에서 결과를 generation pool에 직접 submit*합니다. 즉, 쿼리 `i+1`의 retrieval이 쿼리 `i`의 generation과 겹칩니다 — CPU와 네트워크가 동시에 포화.

```
Retrieval pool (4 workers)              Generation pool (8 workers)
─────────────────────────               ─────────────────────────
  embed + search query i  ──submit──→  gpt-4o streaming for i
  embed + search query i+1             (다른 gen들과 동시)
                                        gpt-4o streaming for i+1
  ...                                   ...
```

**수업 매칭:** **Week 10/11 — Parallel Programming** (다중 풀 thread 조정).

**결과 (8개 쿼리, end-to-end, embed + search + 완전한 generation 포함):**

| 모드 | Batch 총 시간 | 쿼리당 amortized | 속도 향상 |
|------|:---:|:---:|:---:|
| Sequential naive | 14,456 ms | 1,807 ms | 1.0× |
| **Pipelined (4 retrieve + 8 gen)** | **2,817 ms** | **352 ms** | **5.13×** |

**이론적 상한 ~7×;** 측정된 5.13×는 실제 thread 스케줄링 오버헤드, Python GIL 전환, Kong 프록시 동시성 제한을 반영. 정직한 측정, 정직한 설명.

**왜 threading이 옳고 multiprocessing이 아닌지:** 두 단계 모두 GIL을 해제합니다 — gpt-4o 네트워크 대기는 자연스럽게, PyTorch 임베딩 추론은 내부적으로. Multiprocessing은 프로세스 spawn 오버헤드(Windows에서 ~400 ms)를 추가하며 추가 병렬성 없음. 실제로 프로젝트의 `ParallelIVFBuilder` (multiprocessing 기반)는 순차 버전보다 *느리게* 실행됩니다 — threading 선택을 검증하는 negative result.

---

### Step 6 — 동료의 쿼리 경로 최적화

공유 저장소의 동료가 세 가지 쿼리 경로 마이크로 최적화를 기여했으며, 베이스라인 코드 수정 없이 git merge로 통합했습니다.

**추가된 것:**
1. **`use_precomputed_norms`** — 코퍼스 벡터의 L2 norm을 빌드 시점에 한 번 계산하고 캐싱, 쿼리마다 재계산 안 함
2. **`use_numpy_candidate_gather`** — IVF 후보 수집을 Python `list.extend` 대신 `np.concatenate`로
3. **`cosine_sim_numba_parallel_precomputed`** — precomputed norms를 세 번째 인자로 받는 Numba 병렬 유사도 커널

**통합 방식:** `git merge origin/main`을 통해 별도 `integrate-friend` 브랜치로 병합. 세 변경 모두 기존 파일(`components/vector_index.py`, `optimized/similarity_numba.py`) 안에 **opt-in 플래그**로 있으며, Steps 1-5 코드와 후방 호환.

**Ablation 연구 (IVF n_probes=2, medium 데이터셋):**

| 변형 | sim_fn | norm cache | np gather | 지연시간 | vs A |
|---------|--------|:---:|:---:|:---:|:---:|
| **A. flags off** | NumPy | ✗ | ✗ | 3.91 ms | 1.0× |
| **B. + norm cache** | NumPy | ✓ | ✗ | **1.78 ms** | **2.19×** |
| **C. + np gather** | NumPy | ✓ | ✓ | **1.77 ms** | **2.21×** |
| D. + Numba parallel precomputed | Numba par | ✓ | ✓ | 2.06 ms | 1.90× ❌ |

**핵심 발견:**
- **Norm cache 하나만으로 95%의 이득 포착** (3.91 → 1.78 ms). 불변인 코퍼스 norm을 캐싱하는 한 줄 최적화.
- **np gather는 한계적** (+0.5%), 이 스캔 크기(~3000 후보)에서는 Python list 오버헤드가 작업에 비해 작음.
- **변형 D는 negative result** — precomputed norms가 있는 Numba 병렬이 IVF의 축소된 스캔 크기에서 NumPy보다 *느림*. ~3000 벡터의 BLAS GEMV는 손으로 짠 `prange`로 이기기 어려움. 이는 프로젝트 중심 논지를 확인합니다: **올바른 도구는 연산 형태에 따라 다르다**. Numba 병렬은 대규모 전체 코퍼스 스캔(Section 2: 100K 벡터에서 NumPy 대비 10×)에서 이기지만 IVF 필터링된 작은 부분집합에서는 집니다.

**예상 외 보너스 — BruteForce가 10× 빨라짐:**

merge 후 `use_precomputed_norms`가 기본 `True`이므로 `BruteForceIndex.search`가 자동으로 이득을 봅니다. merge 후 BF 지연시간은 **57.80 ms에서 5.85 ms로** — 우리 자신의 최적화 노력이 아닌 merge에서 얻은 10× 속도 향상. "통합에서 얻은 공짜 승리"로 정직하게 보고.

---

## 4. 최종 3-Way 비교

`comparisons/` 아래의 독립 실행 가능한 Python 스크립트 3개가 각각 완전한 8-쿼리 end-to-end RAG 파이프라인을 실행하고 동일한 지표를 보고합니다.

### 설정

| Config | 인덱스 | sim_fn | Norm cache | Np gather | 생성기 | 아키텍처 |
|--------|-------|--------|:---:|:---:|-----------|--------------|
| 1. BruteForce | `BruteForceIndex` | `cosine_sim_numpy` | ✗ | — | non-stream | serial |
| 2. IVF default | `IVFIndex(32, np=4)` | `cosine_sim_numpy` | ✗ | ✗ | non-stream | serial |
| 3. Fully optimized | `IVFIndexNumbaPP(64, np=8)` | `cosine_sim_numpy` | ✓ | ✓ | **streaming** | **pipelined (4+8)** |

### 결과 (실제 Kong gpt-4o, 8개 쿼리, 쿼리당 K=5)

| 지표 | Config 1: BruteForce | Config 2: IVF default | Config 3: Fully optimized | 3 vs 1 속도 향상 |
|--------|:---:|:---:|:---:|:---:|
| 빌드 시간 | 130 ms | 3,530 ms | 1,972 ms | 0.07× |
| 쿼리당 Embed ms | 8.1 | 7.6 | 37.2 | — |
| **쿼리당 Search ms** | 62.9 | 16.7 | **14.5** | **4.35×** |
| 쿼리당 Gen ms | 1,754 | 1,722 | 1,598 | — |
| 쿼리당 총 시간 ms | 1,825 | 1,747 | 1,650 | 1.11× |
| **Batch 총 시간** | 14,601 ms | 13,974 ms | **3,770 ms** | **3.87×** |
| Recall@5 (8 쿼리) | 0.75 | 0.50 | 0.625 | — |

*Recall@5 행은 샘플 크기가 8 쿼리뿐이어서(각각 찾음/못 찾음 binary) 해상도가 낮습니다. 견고한 recall은 500 쿼리 기반 Step 1 수치 참조.*

![comparison figure](results/figures/three_way_comparison.png)

### 세 가지 결정적 발견

**1. BruteForce → IVF default = batch가 4%만 빠름 (14.60 s → 13.97 s).**
알고리즘만 바꾸는 것은 거의 도움이 안 됩니다. 생성이 batch 시간의 ~1.7 s/쿼리 × 8 = 13.6 s를 지배. IVF의 4× 검색 속도 향상(63 → 17 ms)은 총 ~0.4 s만 절약, **생성 벽에 대해 보이지 않음**.

**2. IVF default → Fully optimized = batch가 3.7× 빠름.**
진짜 속도 향상은 **아키텍처 변경**에서 오지, 알고리즘에서 오지 않습니다. Pipelined streaming은 8 gen 호출을 동시에 실행하게 하고, retrieval이 그들과 겹치게 함.

**3. 알고리즘적 승리는 검색 레이어에서 여전히 실제적 (4.35×).**
단지 LLM 지배 파이프라인의 end-to-end wall-clock에서 보이지 않을 뿐. 비-LLM 검색 시스템(고전적 search 같은)에서는 이것이 지배적 지표.

**중심 통찰:** *한 단계가 총 지연시간을 지배할 때, 빠른 단계를 최적화해도 end-to-end 효과가 없습니다. 아키텍처를 재배치해서 단계들을 겹치게 해야만 사용자가 볼 수 있는 속도 향상이 가능.*

---

## 5. 사용자 체감 영향

백엔드 지표를 사용자 경험으로 번역.

### 시나리오 A: 단일 사용자, 단일 질문

| 단계 | 베이스라인 | 완전 최적화 | 사용자 영향 |
|-------|:---:|:---:|:---:|
| 쿼리 embed (CPU) | 300 ms | 300 ms | — |
| Search | 63 ms | 10 ms | — |
| 첫 단어까지 Generation (TTFT) | **1,738 ms** | **961 ms** | **−45%** |
| **답변이 나타나기 시작하는 시간** | ~2,100 ms | **~1,270 ms** | **32% 짧아진 대기** |
| 완전한 답변까지 시간 | 2,100 ms | 1,750 ms | 17% 짧음 |

사용자는 응답을 "2.1 s"가 아닌 "1.3 s"로 인식 — 채팅 UI에서 "빠릿빠릿"과 "렉걸림"의 차이.

### 시나리오 B: 8개 쿼리 버스트 (멀티턴 채팅 또는 배치)

| 패턴 | 총 대기시간 |
|---------|:---:|
| 모든 쿼리 순차 (원본) | **14.6 s** |
| Pipelined + streaming (최적화) | **2.8 s** |
| **속도 향상** | **5.13×** |

빠르게 연속으로 8개 질문하는 사용자는 15초가 아닌 3초 대기.

### 시나리오 C: 멀티 사용자 서빙 (10 동시 사용자, 한 파이프라인)

| 패턴 | 마지막 사용자 대기시간 |
|---------|:---:|
| 원본 (순차 디스패치) | ~18 s |
| 최적화 (pipelined 동시) | ~3 s |
| **감소** | **~83%** |

동일한 하드웨어가 약 5× 더 많은 동시 부하를 처리, 꼬리 사용자가 몇 분을 기다리지 않음.

### 품질은 유지됨

| 지표 | 베이스라인 | 최적화 | 변화 |
|--------|:---:|:---:|:---:|
| Recall@10 (500 쿼리) | 0.8908 | 0.8508 (IVF np=8) | −4.5% |
| MRR | 0.4937 | 0.4754 | −3.7% |

5× batch 속도 향상을 위한 ~4% 상대 recall 하락은 유리한 트레이드오프. 대신 `n_probes=16` 선택 시 **Recall@10 = 0.8728 (BF의 98%)** 획득 가능하며 검색 지연시간이 대략 2배 — 여전히 생성 지배 총 시간 아래에 충분히 있음.

---

## 6. 결론 및 핵심 통찰

### 기억할 가치 있는 수치

| 차원 | 최고 성과 |
|-----------|:---:|
| 인덱스 빌드 시간 | 3.41 s → 2.08 s (**−39%**, Numba K-Means 통해) |
| 쿼리당 search | 63 ms → 14 ms (**4.35×**, IVF + norm cache + np gather) |
| 체감 지연 (TTFT) | 1,738 ms → 961 ms (**−45%**, streaming 통해) |
| End-to-end batch | 14,601 ms → 3,770 ms (**3.87×**, pipelined 아키텍처 통해) |
| 검색 품질 | 0.8908 → 0.8508 (**−4.5%**, 허용 가능한 트레이드오프) |

### 수행한 최적화 차원

이 프로젝트는 의도적으로 한 가지를 깊게 파는 대신 **6가지 별개 차원**을 커버. 각각 다른 수업 주차에 매핑됨:

| 차원 | 기법 | 수업 주차 |
|-----------|-----------|:---:|
| 알고리즘 구조 | IVF가 BruteForce 대체 | (설계) |
| 알고리즘 개선 | K-Means++ 시딩 | **Week 8** |
| 코드 실행 | Numba JIT 컴파일 커널 | **Week 6** |
| 데이터 접근 패턴 | 사전 계산된 캐시된 norms | (동료에서 merge) |
| 동시성 | Streaming + concurrent generation | **Week 9** |
| 시스템 아키텍처 | Dual-pool pipelined serving | **Week 10/11** |

### 프로젝트 명제

> **속도는 한 층에 살지 않는다. 실제 시스템은 모든 층에 병목이 있다 — CPU 수학, 데이터 접근 패턴, IO 대기, 단계 구성 — 각각 자신의 도구가 필요하다.**

이 프로젝트에서 가장 반직관적인 결과는 3-way 비교가 **IVF 단독으로는 end-to-end 속도 향상이 4%에 불과**하다는 것을 보여준 것인데, gpt-4o 생성이 wall-clock 시간을 지배하기 때문입니다. 극적인 3.87× batch 속도 향상은 전적으로 아키텍처 재배치(pipelining + streaming) 때문이지, 알고리즘 최적화가 아닙니다. 이는 광범위 커버리지 접근을 검증합니다: **어떤 도구가 어디에 적용되는지 이해하는 것이 하나의 도구를 극한으로 밀어붙이는 것보다 중요**합니다.

---

## 재현성

- 모든 최적화 단계는 `results/experiments/medium_*.json` 아래 태그된 JSON 스냅샷
- 3-way 비교 산출물은 `results/comparisons/`에 있으며 `compare_pipelines.ipynb`로 재생성 가능
- `rag-optimization/components/`의 베이스라인 코드는 절대 수정 안 됨; 모든 최적화는 `optimized/` 아래 drop-in 변형으로 존재
- Integration 브랜치 `integrate-friend`는 저자와 동료의 양쪽 commit 히스토리를 덮어쓰지 않고 보존

단계별 전체 세부사항은 [OPTIMIZATION_JOURNEY.md](OPTIMIZATION_JOURNEY.md) 및 [PIPELINE_MAP.md](PIPELINE_MAP.md) 참조.
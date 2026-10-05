[Previous](02-VLM-后训练.md) | [Contents](../../README.md) | [Next](04-Diffusion.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/projects.html#c=3)

# 3. <span style="color: rgb(36,91,219); background-color: inherit">VLM 应用</span>

## 3.1 <span style="color: rgb(36,91,219); background-color: inherit">商品理解与问答系统（RAG + SFT + RL）</span>

### 3.1.1 <span style="color: rgb(36,91,219); background-color: inherit">项目目标</span>

输入支持文字、商品图和图文混合问题。查询先带上租户、principal、商品范围和时间，经过权限过滤、多路召回、RRF、reranker 与证据压缩，再生成带 citations 的结构化答案。

库存、价格、优惠和预计送达时间从实时数据源读取；静态索引保存带版本、生效时间和来源定位的稳定证据。回答器基于当前召回证据生成答案，并在证据冲突或缺失时进入澄清或拒答分支。

| **<span style="color: rgb(36,91,219); background-color: inherit">环节</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">输入</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">输出</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">检查项</span>** |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 检索                                                                           | 文字、图片、身份与商品范围                                                                | 可访问的版本化证据块                                                                   | Recall@k、nDCG、ACL 泄漏                                                          |
| 生成                                                                           | 固定的 Top-K 证据                                                                 | answer、claims、citations                                                      | 答案与引用正确率                                                                      |
| 服务                                                                           | 协议结果与动态字段状态                                                                  | 回答、澄清或人工复核                                                                   | 模型、索引、策略版本                                                                    |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**`project.json` 绑定模型 revision、索引版本、策略版本、评测集和制品路径；`evaluate.py` 输出检索、回答、引用和系统指标。

```json
{
  "project_id": "product-rag-sft-grpo",
  "revisions": {"generator": "resolved_commit_sha", "embedding": "resolved_commit_sha", "reranker": "resolved_commit_sha"},
  "index": {"snapshot": "catalog_2026_07", "dense": "qwen3_vl_embedding", "sparse": "bm25", "fusion": "rrf"},
  "security": {"filter_stage": "pre_retrieval", "keys": ["tenant_id", "principal", "effective_at"]},
  "training": {"sft": "configs/sft.yaml", "grpo": "configs/grpo.yaml"},
  "evaluation": {"set": "eval/product_qa_test.jsonl", "predictions": "artifacts/predictions.jsonl", "report": "artifacts/eval_report.json"}
}
```

### 3.1.2 <span style="color: rgb(36,91,219); background-color: inherit">商品数据</span>

知识库的最小单位不是一整个商品页面，而是可独立引用的证据块。每个块同时保存商品、变体、租户、来源类型、来源版本、生效时间、过期时间、ACL、结构化 facts、媒体哈希和删除状态。这样才能回答“哪个变体、哪个版本、哪张图支持这条结论”。

<span style="color: rgb(100,37,208); background-color: inherit">来源优先级由字段语义决定：接口外观可以由图片支持，额定功率、兼容协议和保修范围通常要回到规格表或说明书。</span> <span style="color: rgb(216,57,49); background-color: inherit">把“图片一定是真相”写成统一规则，会在包装示意图、旧版商品图和商家误传图片上产生系统性错误。</span>

| **<span style="color: rgb(36,91,219); background-color: inherit">字段组</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">代表字段</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">作用</span>** |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| 业务主键                                                                          | product\_id、variant\_id                                                        | 避免把相似商品或不同容量混成同一证据                                                           |
| 权限与生命周期                                                                       | tenant\_id、acl、effective\_at、expires\_at、deleted                               | 决定证据能否进入检索候选                                                                 |
| 来源追踪                                                                          | source\_type、source\_revision、media.sha256                                     | 支持回放、更新和删除                                                                   |
| 可验证事实                                                                         | facts、text、media                                                               | 为结构化 claim 和图片引用提供依据                                                         |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**&#x8FC7;滤函数在召回之前检查租户、ACL、时间、删除状态、商品范围和结构化过滤条件。

```python
def is_eligible(chunk, query):
    if chunk["deleted"] or chunk["tenant_id"] != query["tenant_id"]:
        return False
    principals = set(query["principals"]) | {"public"}
    if not principals.intersection(chunk["acl"]):
        return False
    as_of = parse_time(query["as_of"])
    if parse_time(chunk["effective_at"]) > as_of:
        return False
    if chunk["expires_at"] and parse_time(chunk["expires_at"]) <= as_of:
        return False
    if query.get("product_scope") and chunk["product_id"] != query["product_scope"]:
        return False
    return all(chunk["facts"].get(k) == v for k, v in (query.get("filters") or {}).items())
```

### 3.1.3 <span style="color: rgb(36,91,219); background-color: inherit">多模态召回</span>

查询同时绑定文字、图片、租户身份、principal、查询时刻和商品范围。多模态检索使用 Qwen3-VL-Embedding 编码文字、商品图与混合输入，BM25 补足型号、接口名和精确属性词；`HashingEmbedder` 构造确定性的排序回归输入，模型实验保持 query、候选集和评测脚本不变。

<span style="color: rgb(46,161,33); background-color: inherit">dense 与 sparse 在同一批合格证据上运行，可以同时覆盖语义相似问题和精确型号问题。</span> <span style="color: rgb(216,57,49); background-color: inherit">先从全库取 Top-K 再删除越权结果仍然不安全，因为敏感文档已经参与相似度计算、排序和日志。</span>

| **<span style="color: rgb(36,91,219); background-color: inherit">检索通道</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">擅长的问题</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">主要风险</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">控制方法</span>** |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Dense                                                                          | 同义改写、图文混合查询                                                                     | 相似商品和变体混淆                                                                      | 商品范围、硬负样本、精排                                                                   |
| Sparse                                                                         | 型号、接口、精确属性词                                                                     | 关键词命中但语义错误                                                                     | 与 dense 融合并校验 facts                                                            |
| 权限过滤                                                                           | 租户、ACL、时间和删除状态                                                                  | 越权证据进入候选                                                                       | 在两个召回通道之前执行                                                                    |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**&#x48;ybridRetriever 先生成 eligible 集合，再在这个集合上计算 dense 与 BM25 排名。

```python
eligible = [chunk for chunk in self.chunks if is_eligible(chunk, query)]
if not eligible:
    return []

query_vector = self.embedder.embed_query(query)
chunk_vectors = self.embedder.embed_chunks(eligible)
dense_scores = [cosine(query_vector, vector) for vector in chunk_vectors]
sparse_scores = BM25(searchable_text(chunk) for chunk in eligible).scores(query["text"])

dense_ranking = self._rank(dense_scores, eligible, dense_limit)
sparse_ranking = self._rank(sparse_scores, eligible, sparse_limit)
```

### 3.1.4 <span style="color: rgb(36,91,219); background-color: inherit">融合与精排</span>

dense 分数和 BM25 分数不在同一量纲，直接加权需要额外标定。RRF 只使用每个通道中的相对名次，先形成稳定候选集；Qwen3-VL-Reranker 再读取 query 与候选的图文内容，完成更细的相关性判断。

<span style="color: rgb(100,37,208); background-color: inherit">精排训练最有价值的样本不是随机负例，而是同类商品中的相似接口、相近颜色、不同容量、不同适配机型和旧版本证据。</span>最终结果还按 product\_id 与 variant\_id 限制重复块数量，避免同一商品的多个近重复页面挤满上下文。

| **<span style="color: rgb(36,91,219); background-color: inherit">阶段</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">输入</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">职责</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">可定位问题</span>** |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| RRF                                                                          | dense 与 sparse 排名                                                            | 跨量纲合并候选                                                                      | 单路召回偏差                                                                          |
| Reranker                                                                     | query 与有限候选                                                                  | 图文相关性精排                                                                      | 相似商品、错误变体                                                                       |
| 去重配额                                                                         | 精排后的证据块                                                                      | 控制每个变体占位                                                                     | 上下文被重复块占满                                                                       |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**&#x52;RF 对每路结果去重，并以 rank 的倒数累计融合分数。

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
```

### 3.1.5 <span style="color: rgb(36,91,219); background-color: inherit">引用问答</span>

生成器默认使用 Qwen3-VL-4B-Instruct，只读取精排后的证据。输出采用 product-qa.v1 JSON：answer\_state 区分 grounded、not\_found 和 ambiguous；每条 claim 保存字段和值，并通过 citation\_ids 指向证据；citation 再绑定 chunk\_id、product\_id、source\_revision 和 media\_index。

<span style="color: rgb(46,161,33); background-color: inherit">严格协议让回答可以进入下游客服、审核或检索日志，而不是停留在一段无法校验的自然语言。</span> <span style="color: rgb(216,57,49); background-color: inherit">额外字段、尾随文本、未声明引用、错误商品 ID、错误来源版本和越界图片序号都会触发拒绝或人工复核。</span>

| **<span style="color: rgb(36,91,219); background-color: inherit">对象</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">核心字段</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">校验要求</span>** |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Answer                                                                       | answer\_state、answer                                                           | 状态合法；文本非空；not\_found 不携带事实                                                     |
| Claim                                                                        | field、value、citation\_ids                                                      | 每条事实至少绑定一个已声明引用                                                                |
| Citation                                                                     | chunk\_id、product\_id、source\_revision、media\_index                            | 必须来自当前检索上下文，且版本与图片序号一致                                                         |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**&#x5F15;用不仅检查 chunk\_id 是否存在，还要对齐商品、来源版本和图片索引。

```python
context_map = {context["chunk_id"]: context for context in contexts}
for citation in normalized_citations:
    context = context_map.get(citation["chunk_id"])
    if context is None:
        raise ContractError("citation is outside retrieved context")
    if citation["product_id"] != context["product_id"]:
        raise ContractError("citation product_id does not match retrieved context")
    if citation["source_revision"] != context["source_revision"]:
        raise ContractError("citation source_revision does not match retrieved context")
    if citation["media_index"] is not None and citation["media_index"] >= len(context.get("media", [])):
        raise ContractError("citation media_index is outside retrieved context")
```

### 3.1.6 <span style="color: rgb(36,91,219); background-color: inherit">SFT 训练</span>

SFT 的目标是让模型在给定 query、Top-K 文本与图片时，稳定生成符合 product-qa.v1 的答案。训练样本保存 query\_group，并把查询图片与证据图片完整带入 messages；同一问题模板、同一商品的改写和同一图片的裁剪版不能跨训练集与测试集。

<span style="color: rgb(100,37,208); background-color: inherit">SFT 阶段使用固定检索快照，模型学习的是“如何使用给定证据”，而不是同时适应不断变化的检索器。</span>样本同时覆盖可回答、检索为空、证据冲突、权限过滤后为空和动态字段过期，避免模型只会在完整上下文中复述答案。

| **<span style="color: rgb(36,91,219); background-color: inherit">样本类型</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">目标状态</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">训练价值</span>** |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| 证据完整                                                                           | grounded                                                                       | 学习字段抽取、回答与逐条引用                                                                 |
| 证据为空                                                                           | not\_found                                                                     | 学习拒绝使用参数记忆补写事实                                                                 |
| 证据冲突                                                                           | ambiguous                                                                      | 学习暴露冲突并请求澄清                                                                    |
| 动态字段过期                                                                         | 服务侧实时查询                                                                        | 防止静态索引回答库存与价格                                                                  |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**&#x6784;造器把 gold output 写入 assistant 消息，并把查询图片和证据图片合并到同一训练行。

```python
def build_sft_row(query, chunks):
    contexts = contexts_for_query(query, chunks)
    messages = build_messages(query, contexts)
    messages.append({
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": json.dumps(query["gold_output"], ensure_ascii=False),
        }],
    })
    images = list(query["images"])
    images.extend(media["path"] for context in contexts for media in context["media"])
    return {"query_id": query["query_id"], "query_group": query["query_group"],
            "messages": messages, "images": images}
```

### 3.1.7 <span style="color: rgb(36,91,219); background-color: inherit">GRPO 奖励</span>

GRPO 从已经完成的 SFT adapter 启动，继续冻结检索快照，只优化回答器。同一 query 采样多个 completion，奖励由 answer\_state、结构化字段、引用、合理拒答和简洁性组成；协议非法或生成无证据 claim 时，不再计算软分，直接进入硬门禁。

<span style="color: rgb(216,57,49); background-color: inherit">如果只奖励答案相似度，模型会忽略引用；如果只奖励引用数量，模型会复制更多 citation 骗分；如果检索器与生成器一起变化，离线收益也无法归因。</span> <span style="color: rgb(46,161,33); background-color: inherit">把字段、引用、拒答和长度拆开记录，才能检查模型到底学到了什么。</span>

| **<span style="color: rgb(36,91,219); background-color: inherit">奖励项</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">权重</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">约束目的</span>** |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| answer\_state                                                                 | **`0.25`**                                                                   | 区分可回答、无结果与证据冲突                                                                 |
| fields                                                                        | **`0.30`**                                                                   | 约束结构化事实和值                                                                      |
| citations                                                                     | **`0.25`**                                                                   | 同时计算引用 Precision 与 Recall                                                      |
| abstention                                                                    | **`0.15`**                                                                   | 奖励应该拒答时的正确拒答                                                                   |
| concision                                                                     | **`0.05`**                                                                   | 限制通过冗长文本堆砌信息                                                                   |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**&#x975E;法 JSON、协议错误、引用越界和 unsupported claim 统一返回硬门禁分数。

```python
try:
    prediction = parse_answer(completion, contexts).value
    gold_answer = validate_answer(gold, contexts)
except (ContractError, KeyError, TypeError, ValueError):
    return RewardBreakdown(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True)

if unsupported_claim_ids(prediction, contexts):
    return RewardBreakdown(-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True)
```

### 3.1.8 <span style="color: rgb(36,91,219); background-color: inherit">评测与消融</span>

评测必须把检索、生成、引用、拒答与系统成本拆开。先给生成器 gold context，得到 Oracle Context 上限；再固定生成器替换检索器。Oracle 高而端到端低，问题通常在召回、精排或权限过滤；两者都低，才优先检查输出协议、SFT 数据和生成能力。

核心对照包括 dense-only、sparse-only、RRF、RRF + reranker、SFT、SFT + GRPO。阈值在开发集选择；冻结测试集报告 bootstrap 置信区间、问题类型与商品类目切片，并保存逐样本错误文件。

|       | **<span style="color: rgb(36,91,219); background-color: inherit">指标</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">回答的问题</span>** |
| ----- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 召回    | Recall@K、MRR@K、nDCG@K                                                        | 正确证据是否进入候选以及排在什么位置                                                              |
| 安全    | ACL Leak Rate                                                                | 越权、过期或删除块是否进入候选                                                                 |
| 生成    | Schema Valid、Field Exact Match、Grounded Claim                                | 答案是否可解析且被证据支持                                                                   |
| 引用与拒答 | Citation Precision/Recall、Abstention F1                                      | 引用是否准确覆盖，拒答是否合理                                                                 |
| 系统    | Oracle Context Gap、P50/P95、图片与 token 预算                                      | 损失来自哪一层，以及成本是否可接受                                                               |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**&#x6743;限查询的标准答案为空时，空检索结果计为正确；评测报告分别统计授权过滤和召回错误。

```python
relevant = set(query["relevant_chunk_ids"])
if not relevant:
    empty_result_score = 1.0 if not ids else 0.0
    recalls.append(empty_result_score)
    reciprocal_ranks.append(empty_result_score)
    ndcgs.append(empty_result_score)
    continue

recalls.append(1.0 if relevant.intersection(ids) else 0.0)
first_rank = next((i for i, chunk_id in enumerate(ids, start=1)
                   if chunk_id in relevant), None)
```

### 3.1.9 <span style="color: rgb(36,91,219); background-color: inherit">部署与更新</span>

在线请求先鉴权和判断是否涉及动态字段，再执行召回、精排、上下文压缩和生成。模型输出通过协议与引用校验后才能返回；证据冲突进入澄清，非法输出和 unsupported claim 进入人工复核，动态数据源不可用时不降级为静态猜测。

索引更新采用新 collection 构建、影子评测和 alias 原子切换。日志保存 query\_id、身份摘要、索引版本、模型 revision、候选 ID、精排分数、最终引用与失败路由。<span style="color: rgb(46,161,33); background-color: inherit">这组信息能把每次错误归到数据、召回、精排、生成、协议或服务流程，而不是只看到一条“回答错了”。</span>

| **<span style="color: rgb(36,91,219); background-color: inherit">失败状态</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">服务动作</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">排查重点</span>** |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| 无合格证据                                                                          | 返回 not\_found                                                                  | 权限、时间、删除状态与召回覆盖                                                                |
| 证据冲突                                                                           | 请求澄清                                                                           | 来源版本、字段优先级与商品变体                                                                |
| 协议非法                                                                           | 人工复核                                                                           | JSON、字段集合、尾随文本与长度                                                              |
| unsupported claim                                                              | 人工复核                                                                           | claim、facts 与 citation 的一致性                                                    |
| 动态数据源不可用                                                                       | 请求澄清或稍后重试                                                                      | 实时接口、as\_of 与降级策略                                                              |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**&#x670D;务层把模型异常转换成明确动作，不把解析失败包装成正常答案。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .citations import unsupported_claim_ids
from .contracts import ContractError, parse_answer
from .retrieval import HybridRetriever


@dataclass(frozen=True)
class ServiceResult:
    action: str
    reason: str
    output: dict[str, Any] | None
    context_ids: tuple[str, ...]


def answer_query(
    query: dict[str, Any],
    retriever: HybridRetriever,
    generate: Callable[[dict[str, Any], list[dict[str, Any]]], Any],
    requires_dynamic_data: bool = False,
    dynamic_lookup: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> ServiceResult:
    if requires_dynamic_data:
        if dynamic_lookup is None or dynamic_lookup(query) is None:
            return ServiceResult("clarify", "dynamic_source_unavailable", None, ())
    hits = retriever.search(query)
    contexts = [hit.chunk for hit in hits]
    context_ids = tuple(context["chunk_id"] for context in contexts)
    if not contexts:
        return ServiceResult(
            "answer",
            "no_eligible_context",
            {
                "schema_version": "product-qa.v1",
                "answer_state": "not_found",
                "answer": "当前可访问资料中没有足够信息。",
                "claims": [],
                "citations": [],
            },
            (),
        )
    try:
        output = parse_answer(generate(query, contexts), contexts).value
    except (ContractError, TypeError, ValueError):
        return ServiceResult("manual_review", "invalid_model_output", None, context_ids)
    if unsupported_claim_ids(output, contexts):
        return ServiceResult("manual_review", "unsupported_claim", output, context_ids)
    action = "clarify" if output["answer_state"] == "ambiguous" else "answer"
    return ServiceResult(action, "validated_output", output, context_ids)
```

### 3.1.10 `简历书写`

> **<span style="color: rgb(36,91,219); background-color: inherit">项目名称：商品理解与问答系统</span>**
>
> * 使用商品文本、图片、版本和租户权限构建可更新知识库，在召回前完成 ACL 过滤，再执行 Dense 与 BM25 双路召回、RRF 融合和多模态 reranker 精排；
>
> * 基于 Qwen3-VL 完成带证据、干扰项、引用和拒答样本的 LoRA SFT，并在固定检索上下文上使用 GRPO 优化事实正确、引用合法和信息不足时的拒答行为；
>
> * 将评测拆为 Recall@k、nDCG、Oracle Context、答案正确率、引用正确率和拒答准确率，保存索引版本、候选商品、最终引用及完整请求链路。

### 3.1.11 `面试官问`

1. **<span style="color: rgb(36,91,219); background-color: inherit">权限过滤为什么放在召回前？</span>**
   回答要点：候选集先按 principal、租户和版本过滤，dense、sparse 与 rerank 都在授权集合内执行；候选与引用写入同一条审计记录。

2. **<span style="color: rgb(36,91,219); background-color: inherit">为什么同时保留 Dense 和 BM25？</span>**
   回答要点：Dense 处理语义改写和图文匹配，BM25 补足型号、规格和精确词项；`rrf_fuse()` 在不强行对齐分数尺度的情况下合并名次。

3. **<span style="color: rgb(36,91,219); background-color: inherit">怎样区分检索错误和生成错误？</span>**
   回答要点：分别报告 Recall@k、nDCG 和 Oracle Context 下的答案指标。真实检索失败但 Oracle Context 正确时，问题在召回或重排。

4. **<span style="color: rgb(36,91,219); background-color: inherit">SFT 在 RAG 中学什么？</span>**
   回答要点：学习根据候选证据输出固定 JSON、逐条引用和拒答，不负责替代索引记忆最新商品信息。

5. **<span style="color: rgb(36,91,219); background-color: inherit">GRPO 为什么固定检索上下文？</span>**
   回答要点：先固定候选文档再比较多个回答，reward 才能稳定归因到格式、事实、引用和拒答；检索策略单独做消融。

6. **<span style="color: rgb(36,91,219); background-color: inherit">引用正确怎样验收？</span>**
   回答要点：先校验 citation 对应候选商品和可访问版本，再用 claim-evidence 对齐检查引用内容是否支持答案中的具体结论。

## 3.2 <span style="color: rgb(36,91,219); background-color: inherit">多模态无人机 Agent</span>

### 3.2.1 <span style="color: rgb(36,91,219); background-color: inherit">项目目标</span>

Planner 读取任务意图、相机观测和遥测快照，输出离散任务计划。FlightPlan 先过 Schema 和 Safety Gate，状态机再逐步下发 MAVSDK 指令；PX4 继续处理状态估计、姿态控制、geofence 和 failsafe。

测试按 TraceVehicle、PX4 SITL、HIL 和真机分层运行。每次任务保存 observation、门禁结果、状态转换、MAVSDK 指令、遥测回读和最终状态。

Planner 的动作集合固定为 `takeoff、goto、inspect、hold、rtl、land`。PWM、姿态角速度和任意 MAVLink 命令不在模型接口中，人工急停直接由执行层处理。

| **<span style="color: rgb(36,91,219); background-color: inherit">组件</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">处理</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">输出</span>** |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 多模态 Planner                                                                  | 理解任务和场景，生成高层步骤                                                               | `FlightPlan` JSON                                                            |
| Safety Gate                                                                  | 检查状态、权限、速度、高度和 geofence                                                      | allow/deny 与 reason code                                                     |
| 状态机与 MAVSDK                                                                  | 处理顺序、幂等、超时和指令下发                                                              | ACK、遥测与状态转换                                                                  |
| PX4                                                                          | 状态估计、控制器、围栏与 failsafe                                                        | 飞行状态与故障动作                                                                    |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**`project.json` 绑定场景版本、Planner revision、Safety Gate 配置和实验日志；报告按 TraceVehicle、SITL、HIL、真机分别汇总。

```json
{
  "project_id": "multimodal-drone-agent",
  "planner": {"model_revision": "resolved_commit_sha", "schema": "flight_plan.v1"},
  "safety_gate": {"config": "configs/safety_gate.yaml", "policy_version": "safety_2026_07"},
  "execution": {"adapter": "MAVSDK", "autopilot": "PX4", "idempotency_key": "mission_id+step_id"},
  "evaluation": {
    "trace_vehicle": "artifacts/trace_vehicle_report.json",
    "px4_sitl": "artifacts/sitl_missions.jsonl",
    "hil": "artifacts/hil_runs.jsonl",
    "flight": "artifacts/flight_runs.jsonl"
  }
}
```

### 3.2.2 <span style="color: rgb(36,91,219); background-color: inherit">观测合同</span>

相机帧描述外部场景，遥测描述飞行器自身状态，任务意图描述操作者想完成什么。三者必须绑定到同一个 `observation_id`。Planner 看到的是带时间戳的快照，而不是一个可以无限期复用的事实。

图像保存 SHA-256，遥测保存采集时间、连接、解锁、在空中、电量、位置有效性、经纬度、高度和飞行模式。执行前重新读取遥测；若时间戳不一致或年龄超过门槛，旧计划立即失效并触发重规划。<span style="color: rgb(216,57,49); background-color: inherit">不能用图像中的地面纹理推断 GPS 正常，也不能用模型常识补出电量。</span>

| **<span style="color: rgb(36,91,219); background-color: inherit">输入</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">最小字段</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">失效条件</span>** |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| 图像                                                                           | observation\_id、SHA-256、采集时刻                                                   | 哈希不一致、解码失败、与遥测不同步                                                              |
| 遥测                                                                           | timestamp、连接、电量、位置、高度、模式                                                       | 过期、断链、位置无效、数值越界                                                                |
| 任务意图                                                                         | mission\_id、目标、操作者、批准状态                                                        | 意图变更、权限撤销、批准过期                                                                 |
| 环境约束                                                                         | home、geofence、限高、限速                                                            | 版本变更或未随任务保存                                                                    |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**&#x89C2;测对象只保存 Planner 所需的语义字段，安全判断使用独立遥测对象，避免模型输出反向污染状态。

```python
@dataclass(frozen=True)
class Observation:
    observation_id: str
    image_sha256: str
    telemetry_timestamp_ms: int
    mission_intent: str

def planner_prompt(obs: Observation) -> str:
    return (
        "Return one drone-plan.v1 JSON object. "
        "Never emit actuator, PWM, shell, or MAVLink commands. "
        f"observation_id={obs.observation_id}; "
        f"telemetry_timestamp_ms={obs.telemetry_timestamp_ms}"
    )
```

### 3.2.3 <span style="color: rgb(36,91,219); background-color: inherit">任务规划</span>

Qwen3-VL 接收任务意图、图像和精简遥测摘要，输出版本化的 `drone-plan.v1`。Schema 采用字段白名单，拒绝额外字段；每个计划最多 16 步，step ID 不能重复。`goto` 和 `inspect` 必须给坐标与相对高度，所有数值必须有限。

<span style="color: rgb(100,37,208); background-color: inherit">结构化输出还承担安全约束：</span>它把模型自由度压缩到可枚举动作空间，使门禁、重放、消融和错误归因都有稳定输入。模型生成一段“向前飞一点”无法验收，生成带坐标、高度、超时和 step ID 的计划才可以。

| **<span style="color: rgb(36,91,219); background-color: inherit">字段</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">约束</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">解决的问题</span>** |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| schema\_version                                                              | 固定为 drone-plan.v1                                                            | 防止客户端与模型协议漂移                                                                    |
| observation\_id                                                              | 必须对应当前图像与遥测                                                                  | 定位计划使用了哪次观测                                                                     |
| telemetry\_timestamp\_ms                                                     | 执行时与实时遥测精确匹配                                                                 | 拒绝过期计划                                                                          |
| steps                                                                        | 1 至 16 步，动作白名单，ID 唯一                                                         | 限制循环、越权与重复执行                                                                    |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**&#x6A21;型只返回 JSON；解释文字、Markdown 代码围栏和未知参数都视为协议失败。

```json
{
  "schema_version": "drone-plan.v1",
  "mission_id": "inspection-001",
  "observation_id": "obs-001",
  "telemetry_timestamp_ms": 1000,
  "steps": [{
    "step_id": "s1",
    "action": "takeoff",
    "relative_altitude_m": 8.0,
    "timeout_s": 30.0
  }]
}
```

### 3.2.4 <span style="color: rgb(36,91,219); background-color: inherit">安全门禁</span>

Safety Gate 是模型与执行器之间的独立策略层。它按固定顺序检查重复 step、连接、遥测新鲜度、快照绑定、人工批准、电量、位置有效性、速度、相对高度和 geofence。每次拒绝返回稳定的 `reason_code`，不靠自然语言解释决定后续路由。

低电量并不意味着一律拒绝所有动作。继续巡检应该阻断，但 `hold、rtl、land` 是恢复动作，必须保留。<span style="color: rgb(46,161,33); background-color: inherit">门禁因此对普通动作和恢复动作采用不同规则，避免把返航路径一起封死。</span>

| **<span style="color: rgb(36,91,219); background-color: inherit">检查</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">拒绝码</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">处理</span>** |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 遥测过期                                                                         | stale\_telemetry                                                              | 停止执行并重新采集观测                                                                  |
| 起飞未批准                                                                        | approval\_required                                                            | 等待人工确认，不自动重试                                                                 |
| 低电继续运动                                                                       | battery\_too\_low                                                             | 改走 Hold、RTL 或 Land                                                           |
| 目标越界                                                                         | outside\_geofence                                                             | 拒绝该 step，保留原围栏                                                               |
| 重复 step                                                                      | duplicate\_step                                                               | 不再次下发指令                                                                      |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**&#x95E8;禁先做状态检查，再计算目标点与 home 的球面距离。模型无法修改 policy，也无法把自己的估计写回 telemetry。

```python
if now_ms - telemetry.timestamp_ms > policy.max_telemetry_age_ms:
    return Decision(False, "stale_telemetry")
if plan.telemetry_timestamp_ms != telemetry.timestamp_ms:
    return Decision(False, "plan_telemetry_mismatch")
if step.action is Action.TAKEOFF and not human_approved:
    return Decision(False, "approval_required")
if step.action in MOTION and telemetry.battery_remaining < policy.min_motion_battery:
    return Decision(False, "battery_too_low")
if target_distance_m > policy.geofence.radius_m:
    return Decision(False, "outside_geofence")
```

### 3.2.5 <span style="color: rgb(36,91,219); background-color: inherit">状态机与执行</span>

Safety Gate 放行动作后，状态机继续校验 `idle、airborne、holding、returning、landed、aborted` 之间的合法转换，随后才调用 MAVSDK。非法转换直接写入 reason code，并进入 Hold、RTL 或人工接管分支。

起飞、定点飞行、Hold、RTL 和 Land 走 MAVSDK Action API。连续 setpoint 交给独立 Offboard 控制器；进入 Offboard 前先发送 setpoint 流，并持续提供 proof-of-life，失联动作由 `COM_OF_LOSS_T` 和 `COM_OBL_RC_ACT` 配置。

| **<span style="color: rgb(36,91,219); background-color: inherit">动作</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">执行接口</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">安全责任</span>** |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Takeoff                                                                      | Action.set\_takeoff\_altitude、arm、takeoff                                      | 人工批准、位置和电量先过门禁                                                                 |
| Goto / Inspect                                                               | Action.goto\_location                                                          | 状态机、高度与 geofence 再检查                                                           |
| Hold / RTL / Land                                                            | Action 恢复接口                                                                    | 低电或异常时仍保持可达                                                                    |
| 连续 Offboard                                                                  | 独立实时控制器                                                                        | setpoint 频率、模式监控、失联 failsafe                                                   |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**&#x5148;预演状态转换，确认合法后才 dispatch；调用成功再提交状态，异常则进入 aborted 并尝试 Hold。

```python
class MissionExecutor:
    def __init__(self, vehicle: VehiclePort, gate: SafetyGate) -> None:
        self.vehicle = vehicle
        self.gate = gate
        self.state_machine = MissionStateMachine()
        self.executed_step_ids: set[str] = set()

    async def execute_step(
        self,
        plan: MissionPlan,
        step: PlanStep,
        telemetry: Telemetry,
        *,
        now_ms: int,
        human_approved: bool,
    ) -> ExecutionResult:
        decision = self.gate.evaluate(
            plan,
            step,
            telemetry,
            now_ms=now_ms,
            human_approved=human_approved,
            executed_step_ids=frozenset(self.executed_step_ids),
        )
        if not decision.allowed:
            return ExecutionResult(step.step_id, "blocked", decision.reason_code)
        try:
            self.state_machine.next_state(step.action)
            await asyncio.wait_for(self._dispatch(step, telemetry), timeout=step.timeout_s)
            self.state_machine.accept(step.action)
            self.executed_step_ids.add(step.step_id)
            return ExecutionResult(step.step_id, "completed", "allowed")
        except (TimeoutError, RuntimeError, ValueError):
            self.state_machine.abort()
            try:
                await self.vehicle.hold()
            except RuntimeError:
                pass
            return ExecutionResult(step.step_id, "failed", "executor_error")

    async def _dispatch(self, step: PlanStep, telemetry: Telemetry) -> None:
        if step.action is Action.TAKEOFF:
            await self.vehicle.takeoff(step.relative_altitude_m or 0.0)
        elif step.action in {Action.GOTO, Action.INSPECT}:
            absolute_altitude_m = telemetry.home_absolute_altitude_m + (step.relative_altitude_m or 0.0)
            await self.vehicle.goto(
                step.latitude_deg or 0.0,
                step.longitude_deg or 0.0,
                absolute_altitude_m,
            )
        elif step.action is Action.HOLD:
            await self.vehicle.hold()
        elif step.action is Action.RTL:
            await self.vehicle.rtl()
        elif step.action is Action.LAND:
            await self.vehicle.land()
```

### 3.2.6 <span style="color: rgb(36,91,219); background-color: inherit">闭环恢复</span>

无人机任务不是一次生成整条计划后盲目跑到底。每个 step 执行前都要重新采集遥测、检查观测版本并过门禁；执行完成后记录 step ID 和状态转换。图像目标消失、遥测变旧、飞行模式被地面站切换或 MAVSDK 返回异常时，当前计划失效。

重试先区分“确认未下发”和“下发结果未知”。前者在新遥测下重新规划；后者先读取飞行模式、位置和幂等键，再决定继续、补偿、Hold、RTL、Land 或人工接管。

| **<span style="color: rgb(36,91,219); background-color: inherit">事件</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">系统动作</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">禁止做法</span>** |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| 遥测过期                                                                         | 丢弃计划并重新观测                                                                      | 沿用旧坐标继续执行                                                                      |
| MAVSDK 超时                                                                    | 查询模式和位置，进入 Hold/人工处理                                                           | 无状态确认地无限重试                                                                     |
| 重复 step ID                                                                   | 幂等拒绝                                                                           | 再次下发起飞或移动                                                                      |
| 飞行模式被切换                                                                      | 停止当前计划并尊重 PX4/GCS                                                              | 强制抢回 Offboard                                                                  |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**&#x73;tep ID 在执行成功后持久化；再次出现同一 ID 时，门禁在任何车辆调用之前返回拒绝。

```python
if step.step_id in executed_step_ids:
    return Decision(False, "duplicate_step")

result = await executor.execute_step(
    plan,
    step,
    latest_telemetry,
    now_ms=clock.now_ms(),
    human_approved=approval_store.is_valid(plan.mission_id),
)
if result.status == "completed":
    executed_step_ids.add(step.step_id)
```

### 3.2.7 <span style="color: rgb(36,91,219); background-color: inherit">Planner 训练</span>

Planner 如果需要训练，先用人工任务、专家计划、PX4 SITL 回放和失败案例构造 SFT 数据。输入保存任务意图、图像哈希、遥测快照与 policy 版本，输出保存专家 `drone-plan.v1` 或明确拒绝原因。相邻视频帧、同一地图和同一任务模板按场景分组切分，避免测试集只是在复述训练轨迹。

安全门禁、状态机和 PX4 failsafe 不参与学习。也不把在线强化学习放进飞行闭环。若要做偏好优化，只能在冻结离线场景上比较计划的协议正确性、任务可行性和保守拒绝，奖励不能覆盖硬规则。

| **<span style="color: rgb(36,91,219); background-color: inherit">样本类型</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">目标输出</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">主要风险</span>** |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| 正常巡检                                                                           | 短计划、合法目标、恢复动作                                                                  | 模板记忆替代视觉理解                                                                     |
| 目标歧义                                                                           | Hold 或请求人工确认                                                                   | 强行选择错误目标                                                                       |
| 状态异常                                                                           | 拒绝继续运动，优先恢复                                                                    | 模型忽略低电或定位无效                                                                    |
| 越权提示                                                                           | 协议级拒绝                                                                          | 输出 PWM、shell 或任意 MAVLink                                                       |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**&#x8BAD;练样本保存 provenance 与拒绝标签；相同 `scene_group` 只能落在一个数据分区。

```json
{
  "sample_id": "sitl-scene-014-frame-006",
  "scene_group": "sitl-scene-014",
  "image_sha256": "<sha256>",
  "telemetry_timestamp_ms": 1000,
  "policy_version": "safety-v1",
  "mission_intent": "检查一号设备外观",
  "assistant_plan": {"schema_version": "drone-plan.v1"},
  "review_status": "expert_verified"
}
```

### 3.2.8 <span style="color: rgb(36,91,219); background-color: inherit">SITL 测试</span>

SITL、failsafe 参数和故障注入脚本都绑定 PX4 revision。电池故障在 PX4 仿真层注入，TraceVehicle 中手工构造的低电 JSON 归在协议测试。

任务集固定 PX4 固件 revision、仿真世界、机型、home、风场、任务脚本和随机种子。每个场景保存 Oracle Plan 与允许的恢复路径，并注入 GPS 丢失、电量下降、MAVLink 中断、遥测延迟、目标误检、目标越界和重复回调。

| **<span style="color: rgb(36,91,219); background-color: inherit">场景</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">期望系统行为</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">需要保存的证据</span>** |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 正常起飞与巡检                                                                      | 计划合法、逐步执行、落地                                                                     | 计划、遥测、ACK、状态迁移                                                                    |
| GPS/位置失效                                                                     | 停止依赖位置的运动，进入 PX4 恢复                                                              | 估计器状态、模式切换、reason code                                                            |
| Offboard/MAVLink 中断                                                          | 触发配置的失联 failsafe                                                                 | 断链时刻、COM\_OF\_LOSS\_T、最终模式                                                        |
| 目标越界                                                                         | Safety Gate 在下发前拒绝                                                               | 目标坐标、围栏版本、outside\_geofence                                                       |

**<span style="color: rgb(222,120,2); background-color: inherit">代码运行：</span>**&#x5148;启动 PX4 SITL，再启动只监听环回 UDP 端点的 MAVSDK 适配器。串口和远端飞行器连接由独立的真机配置与安全开关控制。

```powershell
docker run --rm -it -p 14550:14550/udp px4io/px4-sitl:latest

# Python 侧使用 MAVSDK-Python 官方示例端点
vehicle = MavsdkVehicle("udpin://0.0.0.0:14540")
await vehicle.connect()

# 非 SITL 端点默认抛出 RuntimeError
MavsdkVehicle("serial:///dev/ttyACM0:57600")
```

### 3.2.9 <span style="color: rgb(36,91,219); background-color: inherit">评测与消融</span>

评测拆成协议、安全、任务和系统四层：Schema 合法率衡量输出协议，门禁拒绝与误杀衡量计划安全，任务成功率衡量完整执行，系统指标继续定位感知、状态过期、执行器和 PX4 failsafe。每个总指标都附带场景与 reason code 分桶。

Baseline 从规则 Planner 开始，再比较纯文本 Planner、图文 Planner 和带 SFT 的 Planner。安全消融包括移除遥测时间戳绑定、移除 Safety Gate、移除状态机恢复，但这些实验只能在离线回放或 SITL 运行。

| **<span style="color: rgb(36,91,219); background-color: inherit">层级</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">指标</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">场景切片</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">评测制品</span>** |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| 协议层                                                                          | Schema 合法率、动作白名单命中                                                           | 缺字段、越界参数、非法动作                                                                  | 协议错误清单                                                                         |
| 安全层                                                                          | 危险动作阻断、恢复动作误杀                                                                | 低电量、旧遥测、geofence                                                               | reason code 分布                                                                 |
| 任务层                                                                          | 成功、超时、恢复、重复执行                                                                | 巡检、返航、降落、链路中断                                                                  | SITL 任务日志                                                                      |
| 系统层                                                                          | P50/P95、遥测年龄、MAVSDK 错误                                                       | 负载、网络抖动、相机延迟                                                                   | 时序 trace                                                                       |
| 真机层                                                                          | HIL、受控场地、安全事件                                                                | 机型、天气、场地、人工接管                                                                  | 飞行与安全记录                                                                        |

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现：</span>**`evaluate_fixtures.py` 按场景汇总 reason code、状态转换和最终动作，并保存逐场景 trace。

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/evaluate_fixtures.py `
  --scenarios tests/fixtures `
  --output artifacts/fixture_report.json

# artifacts/fixture_report.json
{
  "report_type": "trace_vehicle",
  "scenario_count": 6,
  "passed_count": 6,
  "checks": ["schema", "safety_gate", "transition", "reason_code"],
  "trace_dir": "artifacts/traces"
}
```

### 3.2.10 `简历书写`

> **<span style="color: rgb(36,91,219); background-color: inherit">项目名称：多模态无人机 Agent</span>**
>
> * 将机载图像、任务意图和实时遥测绑定为版本化 observation，由 VLM Planner 生成六类受限高层动作，避免模型直接输出底层飞控指令；
>
> * 设计 Safety Gate、幂等状态机和 MAVSDK 执行适配层，对低电、过期遥测、越界、断链、超时和重复执行设置独立拦截与恢复分支；
>
> * 使用 TraceVehicle、PX4 SITL、HIL 和真机日志分层验证任务成功、门禁拒绝、故障恢复、人工接管和安全事件，保留每次任务的 observation、action 与状态迁移轨迹。

### 3.2.11 `面试官问`

1. **<span style="color: rgb(36,91,219); background-color: inherit">为什么 Planner 不能直接调用 Offboard？</span>**
   回答要点：模型只输出高层动作，`safety.py` 和 `executor.py` 负责地理围栏、遥测新鲜度、状态转换和副作用控制。

2. **<span style="color: rgb(36,91,219); background-color: inherit">Agent 和 PX4 的 geofence 怎样分工？</span>**
   回答要点：Agent 门禁负责任务级拒绝、路径解释和 reason code；PX4 围栏负责飞控侧最终保护，两层分别拦截任务决策和执行故障。

3. **<span style="color: rgb(36,91,219); background-color: inherit">MAVSDK 超时后怎样恢复？</span>**
   回答要点：状态机读取遥测、飞行模式和幂等键确认当前状态，再选择继续、补偿、Hold 或人工接管。

4. **<span style="color: rgb(36,91,219); background-color: inherit">TraceVehicle 与 SITL 分别覆盖什么？</span>**
   回答要点：TraceVehicle 覆盖协议、门禁和状态转换；SITL 加入飞控动力学、通信时序、任务执行与 failsafe。

5. **<span style="color: rgb(36,91,219); background-color: inherit">SITL 之后为什么还要 HIL？</span>**
   回答要点：HIL 用真实飞控硬件暴露时钟、链路、传感器和算力约束；通过后仍需独立安全评审才能进入真机。

6. **<span style="color: rgb(36,91,219); background-color: inherit">如何定位一次任务失败？</span>**
   回答要点：按 observation、planner、safety gate、state machine、MAVSDK、PX4 六层回放日志，先确认失败层，再修改对应代码和回归场景。

---

[Previous](02-VLM-后训练.md) | [Contents](../../README.md) | [Next](04-Diffusion.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/projects.html#c=3)

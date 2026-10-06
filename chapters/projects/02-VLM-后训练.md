[Previous](01-基础知识.md) | [Contents](../../README.md) | [Next](03-VLM-应用.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/projects.html#c=2)

# 2. <span style="color: rgb(36,91,219); background-color: inherit">VLM 后训练</span>

## 2.1 <span style="color: rgb(36,91,219); background-color: inherit">多模态理解（SFT）</span>

### 2.1.1 <span style="color: rgb(36,91,219); background-color: inherit">项目目标</span>

输入是一张商品图或同一商品的多视图，输出是固定 JSON。每个属性都要带图片索引和可见证据；图片里看不清的重量、容量、功能和材质保留空值，交给人工复核。

`data_pipeline.py` 负责数据清洗和商品级切分，`contracts.py` 校验 JSON 与证据索引，`qwen3vl_adapter.py` 跑 Zero-Shot 基线，LoRA 配置交给 LLaMA-Factory，`evaluation.py` 和 `service.py` 分别处理离线指标与线上路由。

| **<span style="color: rgb(36,91,219); background-color: inherit">代码</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">核心职责</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">产出</span>** |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| `project.json`                                                               | 绑定模型、数据和评测配置                                                                   | 运行指纹与制品索引                                                                    |
| `contracts.py`                                                               | 校验商品字段、证据和 decision                                                            | 合法 JSON 或明确错误码                                                               |
| `data_pipeline.py`                                                           | 构建数据、分组切分并检查泄漏                                                                 | train、validation、test 清单                                                     |
| `qwen3vl_adapter.py`                                                         | 封装 Qwen3-VL Zero-Shot 推理                                                       | 结构化预测文件                                                                      |
| `evaluation.py` / `service.py`                                               | 计算指标并执行复核路由                                                                    | 评测报告与审计日志                                                                    |

`project.json` 保存模型 revision、数据快照、训练配置、预测文件和评测报告路径。一次实验对应一份配置指纹，训练和推理都从这份文件取版本。

```text
{
  "project_id": "vlm-understanding-sft",
  "model": {"id": "Qwen/Qwen3-VL-4B-Instruct", "revision": "resolved_commit_sha"},
  "dataset": {
    "manifest": "artifacts/data_manifest.json",
    "split_key": "group_id",
    "media_hash": "sha256"
  },
  "training": {
    "framework": "LLaMA-Factory",
    "method": "LoRA SFT",
    "config": "configs/qwen3vl_lora_sft.yaml"
  },
  "evaluation": {
    "predictions": "artifacts/test_predictions.jsonl",
```


```json
{
  "project_id": "vlm-understanding-sft",
  "model": {"id": "Qwen/Qwen3-VL-4B-Instruct", "revision": "resolved_commit_sha"},
  "dataset": {
    "manifest": "artifacts/data_manifest.json",
    "split_key": "group_id",
    "media_hash": "sha256"
  },
  "training": {
    "framework": "LLaMA-Factory",
    "method": "LoRA SFT",
    "config": "configs/qwen3vl_lora_sft.yaml"
  },
  "evaluation": {
    "predictions": "artifacts/test_predictions.jsonl",
    "report": "artifacts/eval_report.json",
    "metrics": ["field_em", "unsupported_attribute_rate", "evidence_coverage", "review_rate"]
  },
  "serving": {"engine": "vLLM", "contract": "vlm_product.schema.v1"}
}
```

### 2.1.2 <span style="color: rgb(36,91,219); background-color: inherit">输出协议</span>

训练前先固定机器可校验的输出合同。`product_type` 看不清时写 `unknown`；颜色或材质看不清时使用空数组；没有区域标注时采用 `image-level` 证据并令 `bbox=null`。每个非空字段都指向输入图片索引。必需字段明确且证据完整时返回 `accept`，语义不确定时返回 `review`，JSON、枚举或证据索引错误时返回 `reject`。

```text
{
  "schema_version": "1.0",
  "product_type": "chair",
  "attributes": {
    "color": ["black"],
```


```json
{
  "schema_version": "1.0",
  "product_type": "chair",
  "attributes": {
    "color": ["black"],
    "material": ["wood"]
  },
  "visible_text": [],
  "evidence": [
    {"field": "product_type", "media_index": 0, "support": "image_level"},
    {"field": "attributes.color", "media_index": 0, "support": "image_level"},
    {"field": "attributes.material", "media_index": 0, "support": "image_level"}
  ],
  "decision": "accept"
}
```

| **<span style="color: rgb(36,91,219); background-color: inherit">字段</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">允许值</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">校验规则</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">失败路由</span>** |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| product\_type                                                                | 版本化类别或 unknown                                                                | 非 unknown 必须有证据                                                                | review / reject                                                                |
| attributes                                                                   | color、material 字符串数组                                                          | 拒绝额外字段、重复值和错误类型                                                                | reject                                                                         |
| visible\_text                                                                | 可读文字数组                                                                        | 非空时必须有图片级证据                                                                    | review                                                                         |
| evidence                                                                     | field、media\_index、image\_level                                               | 索引必须落在本次输入图片范围                                                                 | reject                                                                         |
| decision                                                                     | accept / review / reject                                                      | 未知必需字段不能 accept                                                                | manual\_review                                                                 |

> **<span style="color: rgb(36,91,219); background-color: inherit">证据机制</span>**
>
> 💡 <span style="color: rgb(100,37,208); background-color: inherit">训练时</span>，证据约束能阻止目录标题中的“真皮”“防水”等不可见事实被直接抄进答案；<span style="color: rgb(100,37,208); background-color: inherit">评测时</span>，它把字段正确和视觉支持分开；<span style="color: rgb(100,37,208); background-color: inherit">上线时</span>，缺证据结果可以转人工，而不是直接写入商品库。

### 2.1.3 <span style="color: rgb(36,91,219); background-color: inherit">数据设计</span>

ABO 有 147,702 个商品和 398,212 张目录图，同一商品通常包含主图、侧图和细节图，适合按商品切分多视图样本。目录元数据先作为候选属性，重量、容量、功能、品牌和部分材质再由标注员检查图片是否可见。

| **<span style="color: rgb(36,91,219); background-color: inherit">数据</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">用途</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">处理</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">位置</span>** |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| ABO 图片与元数据                                                                   | 生成候选属性与商品分组                                                                  | 保存许可快照，人工确认可见字段                                                              | 候选池                                                                          |
| 人工复核样本                                                                       | 领域 SFT 与商品测试                                                                 | 双人抽检，保留冲突和 unknown                                                           | train/validation/test                                                        |
| COCO Caption                                                                 | 通用描述回归                                                                       | 使用原任务 split 与评分脚本                                                            | 外部回归                                                                         |
| TextVQA                                                                      | 图中文字读取与推理                                                                    | 使用原答案协议                                                                      | 外部回归                                                                         |
| A-OKVQA                                                                      | 世界知识型 VQA                                                                    | 分别跑 direct answer 和 multiple choice                                          | 外部回归                                                                         |
| POPE                                                                         | 对象存在性幻觉                                                                      | 固定 polling 与采样设置                                                             | 外部回归                                                                         |

> **<span style="color: rgb(36,91,219); background-color: inherit">数据授权</span>**
>
> ❌ 下载任务保存随包 LICENSE、来源 URL、下载时间、压缩包 SHA-256、许可证 SHA-256 和署名文本。训练清单只接收授权状态明确的 asset\_id；授权变化时按媒体哈希定位并移除相关样本。

### 2.1.4 <span style="color: rgb(36,91,219); background-color: inherit">数据治理</span>

数据流水线先从目录元数据生成候选记录，再做视觉确认。写入训练集前检查图片解码、媒体 SHA-256、来源、许可证、目标 Schema 和商品分组；目录字段只提供候选值，最终标签来自当前图片。

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">文件层</span>**：拒绝丢失、损坏和哈希变化的图片；保留原始字节哈希。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">样本层</span>**：检查图片数与 `<image>` 数一致、消息角色交替、assistant 最后一轮是合法 JSON。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">商品层</span>**：对带盐的 `group_id` 哈希做稳定切分；主图、侧图、细节图与 360 视图不跨集合。
>
> 🥛 4. **<span style="color: rgb(36,91,219); background-color: inherit">近重复层</span>**：精确重复按 SHA-256 合并，感知近重复使用固定版本的图像哈希或 Embedding 聚类，跨 split 的簇进入人工复核。
>
> 5. **<span style="color: rgb(36,91,219); background-color: inherit">审计层</span>**：输出接受、拒绝和待复核计数，保存数据快照、split salt 与构建脚本 commit。

```python
import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
```


```python
import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_media_files(record: dict[str, Any], data_root: Path) -> list[str]:
    errors: list[str] = []
    root = data_root.resolve()
    images = record.get("images", [])
    hashes = record.get("media_sha256", [])
    for index, relative in enumerate(images):
        path = (root / str(relative)).resolve()
        if not path.is_relative_to(root):
            errors.append(f"images[{index}] escapes data_root")
            continue
        if not path.is_file():
            errors.append(f"images[{index}] does not exist: {relative}")
            continue
        if index >= len(hashes):
            continue
        actual = sha256_file(path)
        if actual != hashes[index]:
            errors.append(f"images[{index}] SHA-256 mismatch")
    return errors
```

| **<span style="color: rgb(36,91,219); background-color: inherit">泄漏来源</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">为什么会高估</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">泄漏检查</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">验证证据</span>** |
| ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| 同商品不同角度                                                                        | 外观、背景和拍摄风格高度相似                                                                   | group\_id 稳定切分                                                                 | group 泄漏测试                                                                     |
| 原图与压缩图                                                                         | 像素变化但语义几乎相同                                                                      | 感知近重复聚类                                                                        | 跨 split 簇清单                                                                    |
| 同一文件复用                                                                         | 模型直接见过相同字节                                                                       | media\_sha256 全局检查                                                             | 哈希泄漏测试                                                                         |
| 先看 test 再改规则                                                                   | 评测口径被结果反向污染                                                                      | 冻结 manifest 与评测器                                                               | 报告与 commit                                                                     |

### 2.1.5 <span style="color: rgb(36,91,219); background-color: inherit">数据格式</span>

训练样本使用 ShareGPT 多模态格式：`messages` 保存 `role/content`，`images` 保存媒体路径；消息中的每个 `<image>` 按出现顺序对应一张图片。

```text
dataset_info.json：数据注册
JSON
```


```json
{
  "vlm_product_train": {
    "file_name": "sample/train.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },
  "vlm_product_eval": {
    "file_name": "sample/eval_sft.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

训练记录除了 LLaMA-Factory 消费的 messages 和 images，还保留 sample\_id、group\_id、split、media\_sha256、source 与 review\_required。这些额外字段供项目校验和审计使用，不改变模型看到的消息内容。

```text
{"sample_id":"demo_train_001","group_id":"product_demo_001","split":"train","images":["sample/assets/demo_product.ppm"],"media_sha256":["bfb01feb120b746a4e277cf1f4dadd71b3bdb58c05e9e2df1877910cd6ae6a9b"],"messages":[{"role":"user","content":"<image>\n只根据图片提取商品类型、颜色、材质和可见文字。无法从图中确认的字段使用空数组；不要根据常识补写。严格按 vlm_product.schema.v1 输出 JSON。"},{"role":"assistant","content":"{\"schema_version\":\"1.0\",\"product_type\":\"unknown\",\"attributes\":{\"color\":[\"black\",\"white\"],\"material\":[]},\"visible_text\":[],\"evidence\":[{\"field\":\"attributes.color\",\"media_index\":0,\"support\":\"image_level\"}],\"decision\":\"review\"}"}],"source":{"dataset":"synthetic_format_demo","snapshot_id":"demo-v1","license_id":"CC0-1.0","source_uri":"local-generated"},"review_required":false}
```


> ✅ **<span style="color: rgb(36,91,219); background-color: inherit">校验门禁</span>**
>
> <span style="color: rgb(46,161,33); background-color: inherit">非 messages/role/content 格式、角色顺序错误、图片占位符数量不一致、assistant 答案含图片占位符、媒体哈希错误、来源字段不完整、未经审核的训练记录、group 跨 split 和相同媒体跨 split 都会被拒绝。</span>

### 2.1.6 <span style="color: rgb(36,91,219); background-color: inherit">基线推理</span>

微调前先冻结一个可复现的 Zero-Shot 基线。实现统一指向 `Qwen/Qwen3-VL-4B-Instruct`，使用 `AutoModelForImageTextToText` 和 `AutoProcessor`，强制传入不可变模型 revision，并在解码前裁掉输入 token。

| **<span style="color: rgb(36,91,219); background-color: inherit">固定变量</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">Zero-Shot 与 LoRA 共用</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">保存内容</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">变化项</span>** |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| 模型与处理器                                                                         | 同一 model\_id、revision、chat template                                                           | 模型卡、commit、依赖版本                                                                | 仅 adapter                                                                     |
| 视觉输入                                                                           | 同一图片、顺序、像素上限                                                                                  | 媒体 SHA-256、预处理参数                                                               | 无                                                                             |
| 生成                                                                             | do\_sample=False、相同 max tokens                                                                | 原始文本、解析 JSON、错误                                                                | 无                                                                             |
| 评测                                                                             | 冻结测试集与同一评测代码                                                                                  | 逐样本预测与聚合报告                                                                     | 无                                                                             |

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class InferenceConfig:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    revision: str = ""
    max_new_tokens: int = 512


class Qwen3VLAdapter:
    """Lazy Transformers adapter following the official Qwen3-VL chat path."""

    def __init__(self, config: InferenceConfig) -> None:
        if not config.revision or config.revision.startswith("REPLACE_WITH"):
            raise ValueError("an immutable reviewed model revision is required")
        self.config = config
```


```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class InferenceConfig:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    revision: str = ""
    max_new_tokens: int = 512


class Qwen3VLAdapter:
    """Lazy Transformers adapter following the official Qwen3-VL chat path."""

    def __init__(self, config: InferenceConfig) -> None:
        if not config.revision or config.revision.startswith("REPLACE_WITH"):
            raise ValueError("an immutable reviewed model revision is required")
        self.config = config
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            dtype="auto",
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
        )

    def generate(self, image_paths: Sequence[Path], prompt: str) -> str:
        if not image_paths:
            raise ValueError("at least one image is required")
        if self._model is None or self._processor is None:
            self.load()
        content = [
            {"type": "image", "image": path.resolve().as_uri()}
            for path in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        generated_ids = self._model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.config.max_new_tokens,
        )
        trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
```

Zero-Shot 适配器固定 `transformers>=4.57.0` 和 `qwen-vl-utils==0.0.14`，使用 `apply_chat_template(..., tokenize=True, return_dict=True)` 构造输入。模型按不可变 revision 加载，预测文件记录 processor 与权重版本。

### 2.1.7 <span style="color: rgb(36,91,219); background-color: inherit">LoRA SFT</span>

> 📌 LoRA 从 4B Instruct、`qwen3_vl_nothink`、rank 8、`lora_target: all`、学习率 1e-4、单卡 batch 1、梯度累积 8 和三轮训练起步。学习率、轮数、rank 与 target modules 在 Validation 上逐项比较；每次运行同时记录模型 revision、LLaMA-Factory commit、数据快照和许可证哈希。

```text
qwen3vl_lora_sft.yaml：SFT 配置
YAML
```


```yaml
### model
model_name_or_path: Qwen/Qwen3-VL-4B-Instruct
# Replace this placeholder with a reviewed immutable model commit before training.
model_revision: REPLACE_WITH_REVIEWED_MODEL_COMMIT
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset_dir: data
dataset: vlm_product_train
eval_dataset: vlm_product_eval
template: qwen3_vl_nothink
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/qwen3-vl-4b/lora/product-sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

| **<span style="color: rgb(36,91,219); background-color: inherit">记录项</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">内容</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">排查用途</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">制品</span>** |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| 版本                                                                            | 模型、processor、代码、LLaMA-Factory commit                                         | 定位权重与依赖变化                                                                      | 环境清单                                                                         |
| 数据                                                                            | 快照哈希、许可哈希、split salt、统计                                                      | 定位样本和切分变化                                                                      | 数据清单                                                                         |
| 优化                                                                            | loss、学习率、梯度范数、有效 batch                                                       | 区分不收敛、溢出与数据异常                                                                  | 训练日志                                                                         |
| 系统                                                                            | 硬件、吞吐、峰值显存、总时长                                                               | 对齐运行环境与调度配置                                                                    | 压测报告                                                                         |
| 选型                                                                            | Validation 主指标、幻觉、复核率与切片                                                     | 比较 checkpoint 与随机种子                                                            | 选型表                                                                          |

LLaMA-Factory 和 Qwen finetune 是两条独立训练入口。前者使用 `lora_target`，后者使用 `tune_mm_vision、tune_mm_mlp、tune_mm_llm`；实验配置只保留其中一套参数体系。

> **<span style="color: rgb(36,91,219); background-color: inherit">LoRA 加载</span>**
>
> 📌 `lora_target: all` 可能生成视觉塔和连接器的 adapter 权重。服务启动时枚举实际加载的模块，并用固定图像样本做 adapter 开/关回归；视觉输出没有变化时阻断该版本。

### 2.1.8 <span style="color: rgb(36,91,219); background-color: inherit">评测与消融</span>

CIDEr 计算 Caption 相似度，POPE 检查对象存在性幻觉；商品测试集另外计算 Schema Valid Rate、逐字段 Exact Match、无依据属性率、证据覆盖率、unknown 处理和系统成本。

|      | **<span style="color: rgb(36,91,219); background-color: inherit">指标</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">切片</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">评测输出</span>** |
| ---- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 结构   | Schema Valid Rate、缺失预测数                                                      | 输出长度、图片数量、未知类别                                                               | `schema_report.json`                                                           |
| 字段   | 逐字段 Exact Match、Macro-F1                                                     | 头部/长尾类别、单图/多图                                                                | `field_metrics.json`                                                           |
| 证据   | 无依据属性率、证据覆盖率、复核率                                                             | 反光、遮挡、小字、不可见属性诱导                                                             | `evidence_slices.json`                                                         |
| 外部诊断 | COCO、TextVQA、A-OKVQA、POPE                                                    | 各任务原始 split                                                                  | `benchmark_report.json`                                                        |
| 系统   | TTFT、P50/P95、吞吐、显存、失败率                                                       | 图片数量、像素、并发、输出长度                                                              | `load_test.json`                                                               |

```python
from collections import Counter
from typing import Any, Iterable

from .contracts import validate_prediction


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    if isinstance(value, list):
        return sorted(normalize_value(item) for item in value)
```


```python
from collections import Counter
from typing import Any, Iterable

from .contracts import validate_prediction


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    if isinstance(value, list):
        return sorted(normalize_value(item) for item in value)
    return value


def get_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _is_known(value: Any) -> bool:
    return normalize_value(value) not in (None, "unknown", [], ["unknown"])


def evaluate_records(
    gold_records: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    gold_by_id = {str(item["sample_id"]): item for item in gold_records}
    pred_by_id = {str(item["sample_id"]): item["prediction"] for item in predictions}
    counts = Counter()
    field_hits = Counter()
    field_totals = Counter()
    unsupported = 0
    predicted_attributes = 0
    evidence_hits = 0
    evidence_required = 0

    for sample_id, gold in gold_by_id.items():
        counts["samples"] += 1
        prediction = pred_by_id.get(sample_id)
        if prediction is None:
            counts["missing_predictions"] += 1
            continue
        allowed_media = set(gold.get("allowed_evidence_media", []))
        media_count = max(allowed_media, default=0) + 1
        if not validate_prediction(prediction, media_count):
            counts["schema_valid"] += 1
        counts[f"decision_{prediction.get('decision', 'missing')}"] += 1

        observable = gold.get("observable_fields", {})
        for field, expected in observable.items():
            actual = get_path(prediction, field)
            field_totals[field] += 1
            if normalize_value(actual) == normalize_value(expected):
                field_hits[field] += 1
            if field.startswith("attributes.") and _is_known(actual):
                predicted_attributes += 1
                if not _is_known(expected):
                    unsupported += 1

        evidence_by_field: dict[str, set[int]] = {}
        for item in prediction.get("evidence", []):
            evidence_by_field.setdefault(str(item.get("field")), set()).add(item.get("media_index"))
        for field, actual in ((field, get_path(prediction, field)) for field in observable):
            if not _is_known(actual):
                continue
            evidence_required += 1
            if evidence_by_field.get(field, set()) & allowed_media:
                evidence_hits += 1

    samples = counts["samples"]
    predicted = samples - counts["missing_predictions"]
    return {
        "sample_count": samples,
        "missing_prediction_count": counts["missing_predictions"],
        "schema_valid_rate": counts["schema_valid"] / samples if samples else 0.0,
        "field_exact_match": {
            field: field_hits[field] / total for field, total in sorted(field_totals.items())
        },
        "unsupported_attribute_rate": unsupported / predicted_attributes if predicted_attributes else 0.0,
        "evidence_coverage": evidence_hits / evidence_required if evidence_required else 0.0,
        "manual_review_rate": counts["decision_review"] / predicted if predicted else 0.0,
    }
```

| **<span style="color: rgb(36,91,219); background-color: inherit">消融</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">唯一主变量</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">要回答的问题</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">成本指标</span>** |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Zero-Shot vs LoRA                                                            | 是否加载 adapter                                                                    | SFT 是否改善结构与领域字段                                                                  | 延迟、显存                                                                          |
| 弱标签 vs 审核标签                                                                  | 标签审核状态                                                                          | 元数据噪声是否诱发属性幻觉                                                                    | 审核工时                                                                           |
| 单图 vs 多图                                                                     | 同商品图片数                                                                          | 多视角收益是否大于串扰                                                                      | 视觉 token                                                                       |
| 低 vs 高像素上限                                                                   | image\_max\_pixels                                                              | 小字与纹理收益是否值得成本                                                                    | 显存、吞吐                                                                          |
| 自由文本 vs JSON                                                                 | 输出合同                                                                            | 结构约束是否降低不可解析率                                                                    | 复核率                                                                            |

样例预测检查字段解析、证据路径和路由分支；冻结测试集使用独立预测文件，输出字段、证据、复核率和错误切片。COCO、TextVQA、A-OKVQA 和 POPE 分别按各自的 split 与评分脚本运行，与商品测试集分栏展示。

### 2.1.9 <span style="color: rgb(36,91,219); background-color: inherit">部署与兜底</span>

vLLM 负责底座与 LoRA 推理，应用层处理请求大小、图片数量、JSON 提取、Schema、证据、复核路由和审计日志。服务固定 `vLLM>=0.11.0`，启动参数显式传入模型 revision 与 adapter 路径，每个请求最多四张图片并关闭视频输入。

```text
#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_REVISION:?Set MODEL_REVISION to a reviewed immutable commit}"
```


```bash
#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_REVISION:?Set MODEL_REVISION to a reviewed immutable commit}"
: "${ADAPTER_PATH:?Set ADAPTER_PATH to the released LoRA directory}"

vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --revision "${MODEL_REVISION}" \
  --enable-lora \
  --lora-modules "product=${ADAPTER_PATH}" \
  --limit-mm-per-prompt '{"image": 4, "video": 0}'
```

模型返回后，业务层再次校验。格式或证据非法直接拒绝；模型主动 review/reject 进入人工队列；只有合法 accept 才能自动写库。

```python
from dataclasses import dataclass
from typing import Any

from .contracts import validate_prediction


@dataclass(frozen=True)
```


```python
from dataclasses import dataclass
from typing import Any

from .contracts import validate_prediction


@dataclass(frozen=True)
class RouteResult:
    destination: str
    reasons: tuple[str, ...]


def route_prediction(prediction: dict[str, Any], media_count: int) -> RouteResult:
    errors = validate_prediction(prediction, media_count)
    if errors:
        return RouteResult("reject", tuple(errors))
    if prediction.get("decision") != "accept":
        return RouteResult("manual_review", (f"model decision is {prediction.get('decision')}",))
    return RouteResult("accept", ())
```

| **<span style="color: rgb(36,91,219); background-color: inherit">失败桶</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">典型现象</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">处理</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">是否重试</span>** |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 不可见属性                                                                         | 根据常识猜材质、容量或功能                                                                  | 增加 unknown 负样本；转人工                                                           | 否                                                                              |
| OCR / 反光                                                                      | 包装小字漏读或误读                                                                      | 保留高像素；必要时串联 OCR                                                              | 条件重试                                                                           |
| 多图冲突                                                                          | 不同视角颜色或部件不一致                                                                   | 保留逐图证据并转人工                                                                   | 否                                                                              |
| 格式/证据非法                                                                       | JSON 错、额外字段、越界索引                                                               | 拒绝并保存原始输出                                                                    | 有限修复                                                                           |
| 瞬时系统错误                                                                        | 超时、临时显存不足、下游中断                                                                 | 指数退避和熔断                                                                      | 有限                                                                             |

> **<span style="color: rgb(36,91,219); background-color: inherit">媒体安全</span>**
>
> ❗ 服务显式设置 `--limit-mm-per-prompt`。远程媒体开启域名白名单并关闭重定向；默认链路由应用层下载、扫描和存储图片，再把受控文件交给推理服务。

### 2.1.10 `简历书写`

> **<span style="color: rgb(36,91,219); background-color: inherit">项目名称：多模态商品理解系统</span>**
>
> * 负责商品多视图理解链路，输入主图、细节图和包装图，统一输出类目、颜色、材质、OCR 文本及对应证据位置，处理单图信息缺失和多图属性冲突；
>
> 🎁 * 基于 Qwen3-VL-4B-Instruct 搭建 Zero-Shot 基线，使用 LLaMA-Factory 和 LoRA 完成 SFT，使模型稳定生成约定 JSON，并将属性预测约束在可见证据范围内；
>
> * 按商品 group\_id 切分数据，评测字段 Exact Match、无依据属性率、证据覆盖率和人工复核率，服务端通过 Schema 校验、置信度阈值和人工复核处理低置信样本。

### 2.1.11 `面试官问`

1. **<span style="color: rgb(36,91,219); background-color: inherit">商品元数据为什么还要人工确认？</span>**
   回答要点：元数据描述商品事实，图像标签描述当前照片中的可见证据。`data_pipeline.py` 先把元数据作为候选字段，再由人工确认可见性并写入目标 JSON。

2. **<span style="color: rgb(36,91,219); background-color: inherit">为什么按商品而不是按图片切分？</span>**
   回答要点：同一商品的主图、侧图和细节图高度相似。`stable_group_split()` 按 `group_id` 切分，避免近重复图片跨训练集和测试集。

3. **<span style="color: rgb(36,91,219); background-color: inherit">怎样证明提升来自 LoRA SFT？</span>**
   回答要点：固定底座 revision、processor、图片、prompt、像素上限和解码参数，只替换 adapter，并保存 Zero-Shot 与 SFT 的逐样本预测。

4. **<span style="color: rgb(36,91,219); background-color: inherit">为什么不能只用 Caption 指标？</span>**
   回答要点：Caption 相似度不能验证字段 JSON、证据覆盖和不可见属性。主评测应使用字段 Exact Match、unsupported attribute rate、证据覆盖率和复核路由。

5. **<span style="color: rgb(36,91,219); background-color: inherit">unknown 为什么不是失败？</span>**
   回答要点：图片无法确认时保持空值可以阻止属性幻觉。需要同时评估 unknown 的精确率、召回率、人工复核量和错误放行率。

6. **<span style="color: rgb(36,91,219); background-color: inherit">怎样验证 LoRA SFT 的收益？</span>**
   回答要点：固定数据快照、模型 revision、推理参数和评测脚本，对比 Zero-Shot 与 LoRA SFT 的字段准确率、无依据属性率、证据覆盖率、复核率和系统开销。

训练日志按 step 记录 loss、学习率、梯度范数和 checkpoint；评测报告保存逐样本预测、错误切片、模型 revision 与推理参数。

![](../../images/视觉多模态项目-image-33.png)

![](../../images/视觉多模态项目-image-34.png)

## 2.2 <span style="color: rgb(36,91,219); background-color: inherit">多模态</span>**<span style="color: rgb(36,91,219); background-color: inherit">商品审核系统（SFT + RL）</span>**

### 2.2.1 <span style="color: rgb(36,91,219); background-color: inherit">项目目标</span>

`contracts.py` 定义输出，`datasets.py` 构建 SFT/GRPO 样本，`rewards.py` 计算分项奖励，`policy.py` 执行硬规则，`service.py` 根据模型输出与策略结果选择放行、拒绝或人工复核。

输入包含标题、类目、属性、OCR 文本和一到多张商品图，输出固定 JSON：`pass/reject/review`、风险代码、图片序号、证据区域和策略条款。SFT 先把格式和审核口径训稳，GRPO 再优化风险召回、证据质量和规则遵循。

| **<span style="color: rgb(36,91,219); background-color: inherit">代码</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">核心职责</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">关键约束</span>** |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| `contracts.py`                                                               | 定义 pass/reject/review、风险和证据                                                    | Schema 失败直接进入 review                                                           |
| `policy.py`                                                                  | 执行版本化硬规则                                                                       | 硬违规不允许被模型分数覆盖                                                                  |
| `datasets.py`                                                                | 构建 SFT 与 GRPO 样本                                                               | 按商品分组并绑定 `policy_version`                                                      |
| `rewards.py`                                                                 | 计算格式、规则、证据和代价奖励                                                                | 零方差组不更新                                                                        |
| `service.py`                                                                 | 组合模型、策略和人工复核                                                                   | 模型 confidence 不直接决定放行                                                          |

```text
{
  "project_id": "product-audit-sft-grpo",
  "model": {"id": "Qwen/Qwen3-VL-4B-Instruct", "revision": "resolved_commit_sha"},
  "data": {"snapshot": "audit_v3", "split_key": "group_id", "policy_version": "policy_2026_07"},
  "sft": {"trainer": "TRL SFTTrainer", "adapter": "PEFT LoRA", "config": "configs/sft.yaml"},
  "grpo": {"trainer": "TRL GRPOTrainer", "reward": ["schema", "risk", "evidence", "policy"]},
  "evaluation": {"predictions": "artifacts/test_predictions.jsonl", "report": "artifacts/audit_report.json"},
  "serving": {"engine": "vLLM", "gate": "hard_policy", "fallback": "manual_review"}
}
```


### 2.2.2 <span style="color: rgb(36,91,219); background-color: inherit">审核协议</span>

先固定协议，再开始标注和训练。输出包含六个一级字段；多余字段、尾随解释、越界图片序号、非法坐标、重复风险代码或未声明规则引用统一进入协议错误分支。模型输出错误路由到人工复核，商品风险由独立规则继续判定。

| **<span style="color: rgb(36,91,219); background-color: inherit">字段</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">约束</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">审核含义</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">失败路由</span>** |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| decision                                                                     | pass / reject / review                                                       | 最终业务动作，不是自然语言建议                                                                | manual\_review                                                                 |
| risk\_codes                                                                  | 非空字符串、去重                                                                     | 用于策略统计和分风险评测                                                                   | invalid\_output                                                                |
| evidence                                                                     | 风险、图片序号、证据类型、区域                                                              | 让人工能回看模型依据                                                                     | manual\_review                                                                 |
| policy\_refs                                                                 | 规则 ID 去重                                                                     | 绑定当时生效的策略口径                                                                    | policy\_conflict                                                               |
| explanation                                                                  | 最多 300 字                                                                     | 辅助复核，不能覆盖结构化字段                                                                 | truncate/retry                                                                 |

区域坐标统一采用 `[x1,y1,x2,y2]` 的 0 到 1 归一化值；整图证据使用 `image_level` 且 `bbox=null`。通过样本的风险与证据字段保持为空，拒绝样本至少包含一个风险代码。业务置信度由独立校准器根据模型分数、证据完整度和规则信号计算。

```text
output_schema.json：严格输出结构
JSON
```


```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "audit-output.v1",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "decision", "risk_codes", "evidence", "policy_refs", "explanation"],
  "properties": {
    "schema_version": {"const": "audit-output.v1"},
    "decision": {"enum": ["pass", "reject", "review"]},
    "risk_codes": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
    "evidence": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["risk_code", "media_index", "support", "bbox", "policy_rule_id"],
        "properties": {
          "risk_code": {"type": "string"},
          "media_index": {"type": "integer", "minimum": 0},
          "support": {"enum": ["image_level", "region"]},
          "bbox": {
            "oneOf": [
              {"type": "null"},
              {"type": "array", "prefixItems": [{"type": "number"}, {"type": "number"}, {"type": "number"}, {"type": "number"}], "items": false}
            ]
          },
          "policy_rule_id": {"type": ["string", "null"]}
        }
      }
    },
    "policy_refs": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
    "explanation": {"type": "string", "maxLength": 300}
  }
}
```

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


DECISIONS = {"pass", "reject", "review"}
SUPPORT_TYPES = {"image_level", "region"}
OUTPUT_KEYS = {
    "schema_version",
    "decision",
    "risk_codes",
    "evidence",
    "policy_refs",
    "explanation",
}
EVIDENCE_KEYS = {"risk_code", "media_index", "support", "bbox", "policy_rule_id"}


class ContractError(ValueError):
    """Raised when model output cannot be consumed safely."""


@dataclass(frozen=True)
class ParsedOutput:
    value: dict[str, Any]
    source_text: str


def completion_text(completion: Any) -> str:
    """Normalize TRL chat completions and plain strings."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append(item["content"])
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    if isinstance(completion, dict):
        content = completion.get("content")
        if isinstance(content, str):
            return content
    raise ContractError("unsupported completion type")


def parse_first_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object while rejecting non-whitespace trailing text."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start < 0:
        raise ContractError("JSON object not found")
    try:
        value, end = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON: {exc.msg}") from exc
    if text[start + end :].strip():
        raise ContractError("trailing text after JSON object")
    if not isinstance(value, dict):
        raise ContractError("top-level output must be an object")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ContractError(f"{name} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ContractError(f"{name} must not contain duplicates")
    return value


def _bbox(value: Any, support: str) -> list[float] | None:
    if support == "image_level":
        if value is not None:
            raise ContractError("image-level evidence must use a null bbox")
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ContractError("region evidence requires a four-number bbox")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise ContractError("bbox coordinates must be numbers")
    coords = [float(x) for x in value]
    x1, y1, x2, y2 = coords
    if not all(0.0 <= x <= 1.0 for x in coords) or not (x1 < x2 and y1 < y2):
        raise ContractError("bbox must be normalized and ordered")
    return coords


def validate_audit_output(value: dict[str, Any], image_count: int | None = None) -> dict[str, Any]:
    if set(value) != OUTPUT_KEYS:
        missing = sorted(OUTPUT_KEYS - set(value))
        extra = sorted(set(value) - OUTPUT_KEYS)
        raise ContractError(f"output keys mismatch; missing={missing}, extra={extra}")
    if value["schema_version"] != "audit-output.v1":
        raise ContractError("unsupported schema_version")
    if value["decision"] not in DECISIONS:
        raise ContractError("invalid decision")
    risks = _string_list(value["risk_codes"], "risk_codes")
    refs = _string_list(value["policy_refs"], "policy_refs")
    explanation = value["explanation"]
    if not isinstance(explanation, str) or len(explanation) > 300:
        raise ContractError("explanation must be a string no longer than 300 characters")
    evidence = value["evidence"]
    if not isinstance(evidence, list):
        raise ContractError("evidence must be a list")
    normalized_evidence: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict) or set(item) != EVIDENCE_KEYS:
            raise ContractError("evidence keys mismatch")
        if item["risk_code"] not in risks:
            raise ContractError("evidence risk_code is not declared in risk_codes")
        media_index = item["media_index"]
        if isinstance(media_index, bool) or not isinstance(media_index, int) or media_index < 0:
            raise ContractError("media_index must be a non-negative integer")
        if image_count is not None and media_index >= image_count:
            raise ContractError("media_index is outside the request")
        support = item["support"]
        if support not in SUPPORT_TYPES:
            raise ContractError("invalid evidence support type")
        rule_id = item["policy_rule_id"]
        if rule_id is not None and (not isinstance(rule_id, str) or not rule_id):
            raise ContractError("policy_rule_id must be a non-empty string or null")
        if rule_id is not None and rule_id not in refs:
            raise ContractError("evidence policy_rule_id is not declared in policy_refs")
        normalized_evidence.append({**item, "bbox": _bbox(item["bbox"], support)})
    if value["decision"] == "pass" and (risks or evidence or refs):
        raise ContractError("pass output must not declare risks, evidence, or policy refs")
    if value["decision"] == "reject" and not risks:
        raise ContractError("reject output requires at least one risk code")
    return {**value, "risk_codes": risks, "policy_refs": refs, "evidence": normalized_evidence}


def parse_audit_output(completion: Any, image_count: int | None = None) -> ParsedOutput:
    text = completion_text(completion)
    value = validate_audit_output(parse_first_json_object(text), image_count=image_count)
    return ParsedOutput(value=value, source_text=text)
```

### 2.2.3 <span style="color: rgb(36,91,219); background-color: inherit">数据设计</span>

每条审核样本保存 `sample_id、product_id`、标题、类目、属性、OCR、媒体哈希、来源、授权、采集时间、策略版本和仲裁结果。标签包含 decision、风险、证据和条款，图片与政策口径按同一时间版本对齐。

| **<span style="color: rgb(36,91,219); background-color: inherit">数据层</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">进入条件</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">主要风险</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">用途</span>** |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| 真实审核样本                                                                        | 脱敏、授权明确、双人仲裁                                                                   | 策略口径和人审偏差                                                                      | 训练/测试                                                                        |
| 困难负样本                                                                         | 规则命中但人工判定安全                                                                    | 误杀被简单规则放大                                                                      | 校准/消融                                                                        |
| 合成反事实                                                                         | 记录生成脚本与父样本                                                                     | 伪影让结果虚高                                                                        | 训练/诊断                                                                        |
| 冻结测试集                                                                         | 真实样本、商品级去重                                                                     | 反复调参污染                                                                         | 最终报告                                                                         |

切分键必须是 `product_id`，不能是图片文件名。同一商品的主图、细节图、裁剪图、换背景图和改标题版本必须进入同一 split。精确重复用 SHA-256 拦截；感知近重复需要固定版本的图像向量或感知哈希，再对跨 split 簇人工抽检。

> **<span style="color: rgb(36,91,219); background-color: inherit">数据切分</span>**
>
> 合成二维码、联系方式叠字和图文冲突进入训练集或长尾测试切片，不混入冻结业务测试集。策略变更生成新的 `policy_version` 和标签快照，历史测试集继续使用原口径。

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import validate_audit_output
from .provenance import grouped_split, sha256_file


REQUIRED_RECORD_KEYS = {
    "sample_id",
    "product_id",
    "title",
    "category",
    "attributes",
    "ocr_tokens",
    "media",
    "policy_version",
    "ground_truth",
    "source",
    "license",
    "collected_at",
    "adjudication",
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: record must be an object")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def validate_record(record: dict[str, Any], root: str | Path) -> None:
    if set(record) != REQUIRED_RECORD_KEYS:
        raise ValueError(f"record keys mismatch for {record.get('sample_id', '<unknown>')}")
    media = record["media"]
    if not isinstance(media, list) or not media:
        raise ValueError("media must contain at least one image")
    for item in media:
        if set(item) != {"path", "sha256"}:
            raise ValueError("media item keys mismatch")
        image_path = Path(root) / item["path"]
        if not image_path.is_file():
            raise ValueError(f"missing media file: {image_path}")
        if sha256_file(image_path) != item["sha256"]:
            raise ValueError(f"media hash mismatch: {image_path}")
    validate_audit_output(record["ground_truth"], image_count=len(media))
    adjudication = record["adjudication"]
    if not isinstance(adjudication, dict) or adjudication.get("status") != "resolved":
        raise ValueError("sample must have resolved adjudication")


def assign_splits(records: Iterable[dict[str, Any]], seed: int = 42) -> dict[str, str]:
    return {record["sample_id"]: grouped_split(record["product_id"], seed=seed) for record in records}


def assert_no_group_leakage(records: Iterable[dict[str, Any]], splits: dict[str, str]) -> None:
    group_to_split: dict[str, str] = {}
    for record in records:
        split = splits[record["sample_id"]]
        previous = group_to_split.setdefault(record["product_id"], split)
        if previous != split:
            raise ValueError(f"product group leaked across splits: {record['product_id']}")


def build_prompt(record: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for item in record["media"]:
        content.append({"type": "image", "image": item["path"]})
    request = {
        "title": record["title"],
        "category": record["category"],
        "attributes": record["attributes"],
        "ocr_tokens": record["ocr_tokens"],
        "policy_version": record["policy_version"],
    }
    content.append({"type": "text", "text": json.dumps(request, ensure_ascii=False)})
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "审核商品并只输出 audit-output.v1 JSON。证据必须绑定图片序号与策略规则。",
                }
            ],
        },
        {"role": "user", "content": content},
    ]


def build_sft_row(record: dict[str, Any]) -> dict[str, Any]:
    messages = build_prompt(record)
    messages.append(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": json.dumps(record["ground_truth"], ensure_ascii=False)}
            ],
        }
    )
    return {
        "sample_id": record["sample_id"],
        "product_id": record["product_id"],
        "messages": messages,
        "images": [item["path"] for item in record["media"]],
    }


def build_grpo_row(record: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "prompt": build_prompt(record),
        "images": [item["path"] for item in record["media"]],
        "ground_truth": json.dumps(record["ground_truth"], ensure_ascii=False),
        "policy_json": json.dumps(policy, ensure_ascii=False),
        "context_json": json.dumps(
            {
                "title": record["title"],
                "category": record["category"],
                "attributes": record["attributes"],
                "ocr_tokens": record["ocr_tokens"],
                "image_count": len(record["media"]),
            },
            ensure_ascii=False,
        ),
    }
```

### 2.2.4 <span style="color: rgb(36,91,219); background-color: inherit">策略引擎</span>

大模型擅长融合多图、OCR 和上下文，但不适合独自承担所有审核规则。手机号、外部联系方式、禁售类目、必填属性等确定性条件由策略引擎先执行；模型负责语义模糊、跨图关联和证据定位。这样既保留多模态能力，也让高风险规则可以审计、灰度和快速回滚。

| **<span style="color: rgb(36,91,219); background-color: inherit">阶段</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">职责</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">冲突处理</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">日志</span>** |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| 预规则                                                                          | 确定性禁限售与格式检查                                                                  | reject 直接短路模型                                                                  | 规则 ID                                                                        |
| 多模态模型                                                                        | 语义风险与视觉证据                                                                    | 输出非法进入复核                                                                       | 原始输出                                                                         |
| 后规则                                                                          | 协议、条款与决策一致性                                                                  | 规则优先，不能被覆盖                                                                     | 冲突原因                                                                         |
| 人工复核                                                                         | 处理模糊与新风险                                                                     | 结论回流需重新仲裁                                                                      | 人审版本                                                                         |

策略文件使用不可变版本和 effective\_at。离线报告必须打印 policy\_version；线上日志还要记录输入摘要、命中规则、模型 revision、适配器版本与最终路由。否则发生投诉时无法还原当时为什么拒绝。

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PolicyHit:
    rule_id: str
    risk_code: str
    severity: str
    decision: str


@dataclass(frozen=True)
class PolicyResult:
    policy_version: str
    decision: str
    hits: tuple[PolicyHit, ...]


def load_policy(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        policy = json.load(stream)
    required = {"schema_version", "policy_version", "effective_at", "rules"}
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError("policy keys mismatch")
    if policy["schema_version"] != "audit-policy.v1" or not isinstance(policy["rules"], list):
        raise ValueError("unsupported policy schema")
    ids = [rule.get("id") for rule in policy["rules"] if isinstance(rule, dict)]
    if len(ids) != len(policy["rules"]) or len(ids) != len(set(ids)):
        raise ValueError("policy rule ids must be present and unique")
    return policy


def _get_path(context: dict[str, Any], dotted_path: str) -> Any:
    current: Any = context
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _matches(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "equals":
        return actual == expected
    if operator == "in":
        return isinstance(expected, list) and actual in expected
    if operator == "contains_any":
        if not isinstance(expected, list):
            return False
        values = actual if isinstance(actual, list) else [actual]
        normalized = [str(value).casefold() for value in values if value is not None]
        return any(str(needle).casefold() in value for needle in expected for value in normalized)
    raise ValueError(f"unsupported policy operator: {operator}")


def evaluate_hard_policy(context: dict[str, Any], policy: dict[str, Any]) -> PolicyResult:
    hits: list[PolicyHit] = []
    for rule in policy["rules"]:
        if _matches(_get_path(context, rule["field"]), rule["operator"], rule["value"]):
            hits.append(
                PolicyHit(
                    rule_id=rule["id"],
                    risk_code=rule["risk_code"],
                    severity=rule["severity"],
                    decision=rule["decision"],
                )
            )
    decisions = {hit.decision for hit in hits}
    decision = "reject" if "reject" in decisions else "review" if "review" in decisions else "pass"
    return PolicyResult(policy_version=policy["policy_version"], decision=decision, hits=tuple(hits))
```

### 2.2.5 <span style="color: rgb(36,91,219); background-color: inherit">SFT 冷启动</span>

底座使用 Qwen3-VL-4B-Instruct，SFT 由 TRL SFTTrainer 驱动，adapter 使用 PEFT LoRA。图像通过 `AutoProcessor` 展开，模型由 `AutoModelForImageTextToText` 加载；模型 revision 固定为 commit SHA。

SFT 先解决三件事：稳定生成严格 JSON，学会风险代码与条款映射，建立证据绑定习惯。训练集不要只堆违规样本，还要包含安全商品、困难负样本、信息不足的 review 样本和多图矛盾样本。对 VLM 默认设置 `max_length=None`，除非已经证明截断不会删掉图像 token。

| **<span style="color: rgb(36,91,219); background-color: inherit">变量</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">起始设置</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">记录项</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">选择依据</span>** |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 底座                                                                           | Qwen3-VL-4B-Instruct                                                           | model\_id、commit SHA                                                          | Schema 与风险基线                                                                   |
| 参数高效训练                                                                       | QLoRA，q\_proj/v\_proj                                                          | rank、alpha、dropout                                                            | 开发集指标与峰值显存                                                                     |
| 长度                                                                           | `max_length=None`                                                              | 图像 token 与文本长度分布                                                              | 截断率与长样本指标                                                                      |
| 模型选择                                                                         | Risk Macro-F1 + High-risk FNR                                                  | Schema、证据、误杀                                                                  | 多指标 Pareto 对比                                                                  |

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_config(path: str, override_revision: str | None) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if override_revision:
        config["model_revision"] = override_revision
    if config.get("model_revision") in {None, "", "REQUIRED_COMMIT_SHA"}:
        raise ValueError("pass --model-revision with an immutable commit SHA")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-revision")
    args = parser.parse_args()
    config = _load_config(args.config, args.model_revision)
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("install requirements-ml.txt before training") from exc
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if config["bf16"] else torch.float16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        config["model_id"], revision=config["model_revision"], quantization_config=quantization
    )
    processor = AutoProcessor.from_pretrained(config["model_id"], revision=config["model_revision"])
    dataset = load_dataset("json", data_files=config["dataset_path"], split="train")
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        output_dir=config["output_dir"],
        seed=config["seed"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=config["gradient_checkpointing"],
        bf16=config["bf16"],
        max_length=config["max_length"],
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
    processor.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()
```

学习率、训练轮数和 LoRA rank 先按显存预算设置，再根据开发集 loss、Schema Valid Rate、高风险召回、误杀率和训练稳定性逐项选择。

### 2.2.6 <span style="color: rgb(36,91,219); background-color: inherit">GRPO 奖励</span>

SFT 收敛后再接 GRPO。每个 prompt 采样一组回答，按组内相对 reward 计算优势，不训练单独的 critic；reward 拆成 Schema、风险、证据、策略和长度分量，硬违规先门控再聚合。

| **<span style="color: rgb(36,91,219); background-color: inherit">分量</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">权重</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">计算</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">目的</span>** |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Schema 门控                                                                    | 硬门                                                                           | 非法 JSON、额外字段、越界证据直接 -1                                                       | 可消费                                                                          |
| 策略门控                                                                         | 硬门                                                                           | 漏掉确定性 reject 或规则引用时直接 -1                                                     | 合规                                                                           |
| 决策代价                                                                         | 0.35                                                                         | 漏审、误杀、转复核使用不同代价矩阵                                                            | 业务成本                                                                         |
| 风险 F1                                                                        | 0.30                                                                         | 预测与人工风险集合的 F1                                                                | 多标签                                                                          |
| 证据                                                                           | 0.20                                                                         | 风险、图片序号、类型和 IoU 联合匹配                                                         | 可核验                                                                          |
| 条款                                                                           | 0.10                                                                         | 策略引用集合 F1                                                                    | 可审计                                                                          |
| 解释                                                                           | 0.05                                                                         | 只奖励短且非空的解释，不奖励文采                                                             | 抑制冗长                                                                         |

同一 prompt 的多条 rollout 如果得到完全相同的 reward，组内标准差为零，此时优势全部置零。继续用极小 epsilon 强行归一化会放大数值噪声，却没有提供任何偏好信息。正式训练还要分别监控门控率、各分量均值和奖励方差，避免单一格式奖励压过真实审核目标。

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .contracts import ContractError, parse_audit_output, validate_audit_output
from .policy import evaluate_hard_policy


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    schema: float
    hard_policy: float
    decision: float
    risk_f1: float
    evidence: float
    policy_refs: float
    explanation: float
    gated: bool


WEIGHTS = {
    "decision": 0.35,
    "risk_f1": 0.30,
    "evidence": 0.20,
    "policy_refs": 0.10,
    "explanation": 0.05,
}


def _set_f1(predicted: list[str], gold: list[str]) -> float:
    pred_set, gold_set = set(predicted), set(gold)
    if not pred_set and not gold_set:
        return 1.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def bbox_iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 1.0 if left is None and right is None else 0.0
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1, ix2, iy2 = max(lx1, rx1), max(ly1, ry1), min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _evidence_score(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    matched_gold: set[int] = set()
    matches = 0.0
    for item in predicted:
        candidates: list[tuple[float, int]] = []
        for index, target in enumerate(gold):
            if index in matched_gold:
                continue
            if item["risk_code"] != target["risk_code"] or item["media_index"] != target["media_index"]:
                continue
            if item["support"] != target["support"]:
                continue
            candidates.append((bbox_iou(item["bbox"], target["bbox"]), index))
        if candidates:
            score, index = max(candidates)
            if score >= 0.5:
                matched_gold.add(index)
                matches += 1.0
    precision = matches / len(predicted)
    recall = matches / len(gold)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _decision_score(prediction: str, gold: str) -> float:
    if prediction == gold:
        return 1.0
    # Missing a reject is more costly than sending a safe item to review.
    costs = {
        ("pass", "reject"): -1.0,
        ("review", "reject"): -0.5,
        ("reject", "pass"): -0.75,
        ("review", "pass"): 0.25,
        ("pass", "review"): 0.0,
        ("reject", "review"): -0.25,
    }
    return costs.get((prediction, gold), -0.5)


def _violates_policy(prediction: dict[str, Any], context: dict[str, Any], policy: dict[str, Any]) -> bool:
    result = evaluate_hard_policy(context, policy)
    if result.decision == "reject" and prediction["decision"] != "reject":
        return True
    required_rules = {hit.rule_id for hit in result.hits if hit.decision == "reject"}
    return not required_rules.issubset(set(prediction["policy_refs"]))


def score_completion(
    completion: Any,
    ground_truth: dict[str, Any],
    policy: dict[str, Any],
    context: dict[str, Any],
) -> RewardBreakdown:
    try:
        prediction = parse_audit_output(completion, image_count=context.get("image_count")).value
        gold = validate_audit_output(ground_truth, image_count=context.get("image_count"))
    except (ContractError, KeyError, TypeError, ValueError):
        return RewardBreakdown(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True)
    if _violates_policy(prediction, context, policy):
        return RewardBreakdown(-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True)
    decision = _decision_score(prediction["decision"], gold["decision"])
    risk_f1 = _set_f1(prediction["risk_codes"], gold["risk_codes"])
    evidence = _evidence_score(prediction["evidence"], gold["evidence"])
    policy_refs = _set_f1(prediction["policy_refs"], gold["policy_refs"])
    explanation = 1.0 if 1 <= len(prediction["explanation"]) <= 120 else 0.0
    total = (
        WEIGHTS["decision"] * decision
        + WEIGHTS["risk_f1"] * risk_f1
        + WEIGHTS["evidence"] * evidence
        + WEIGHTS["policy_refs"] * policy_refs
        + WEIGHTS["explanation"] * explanation
    )
    return RewardBreakdown(total, 1.0, 1.0, decision, risk_f1, evidence, policy_refs, explanation, False)


def product_audit_reward(
    completions: list[Any],
    ground_truth: list[str],
    policy_json: list[str],
    context_json: list[str],
    **_: Any,
) -> list[float]:
    """TRL GRPO reward adapter; dataset columns are passed by name."""
    if not (len(completions) == len(ground_truth) == len(policy_json) == len(context_json)):
        raise ValueError("reward inputs must have equal lengths")
    scores: list[float] = []
    for completion, gold_text, policy_text, context_text in zip(
        completions, ground_truth, policy_json, context_json
    ):
        result = score_completion(
            completion=completion,
            ground_truth=json.loads(gold_text),
            policy=json.loads(policy_text),
            context=json.loads(context_text),
        )
        scores.append(result.total)
    return scores
```

### 2.2.7 <span style="color: rgb(36,91,219); background-color: inherit">训练配置</span>

GRPO 样本保留 `image/images、ground_truth、policy_json、context_json`。reward 函数直接接收完整审核上下文，训练从 SFT adapter 继续；`num_generations` 大于 1，组内 reward 方差为零时跳过该组更新。

| **<span style="color: rgb(36,91,219); background-color: inherit">记录项</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">内容</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">诊断作用</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">制品</span>** |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| 版本                                                                            | 代码、模型、TRL、transformers                                                       | 定位依赖漂移                                                                         | 环境锁文件                                                                        |
| 数据                                                                            | 快照、策略、split seed                                                             | 定位口径与泄漏变化                                                                      | 数据清单                                                                         |
| 优化                                                                            | reward 分量、KL、长度、有效 batch                                                     | 识别 reward hacking                                                              | 训练曲线                                                                         |
| 系统                                                                            | GPU、显存、吞吐、总时长                                                                | 分离算法与硬件变量                                                                      | 压测清单                                                                         |

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_config(path: str, override_revision: str | None) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if override_revision:
        config["model_revision"] = override_revision
    if config.get("model_revision") in {None, "", "REQUIRED_COMMIT_SHA"}:
        raise ValueError("pass --model-revision with an immutable commit SHA")
    if not Path(config["sft_adapter_path"]).exists():
        raise FileNotFoundError("SFT adapter path does not exist")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-revision")
    args = parser.parse_args()
    config = _load_config(args.config, args.model_revision)
    try:
        import torch
        from datasets import load_dataset
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("install requirements-ml.txt before training") from exc
    from product_audit.rewards import product_audit_reward

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if config["bf16"] else torch.float16,
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        config["model_id"], revision=config["model_revision"], quantization_config=quantization
    )
    model = PeftModel.from_pretrained(base_model, config["sft_adapter_path"], is_trainable=True)
    processor = AutoProcessor.from_pretrained(config["model_id"], revision=config["model_revision"])
    dataset = load_dataset("json", data_files=config["dataset_path"], split="train")
    training_args = GRPOConfig(
        output_dir=config["output_dir"],
        seed=config["seed"],
        learning_rate=config["learning_rate"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_generations=config["num_generations"],
        max_completion_length=config["max_completion_length"],
        temperature=config["temperature"],
        beta=config["beta"],
        bf16=config["bf16"],
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=product_audit_reward,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
    processor.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()
```

> **<span style="color: rgb(36,91,219); background-color: inherit">训练框架</span>**
>
> Qwen3-VL 先走 TRL 的 VLM GRPO 入口。切换 VeRL 时单独验证 rollout、图像预处理、LoRA 加载和 checkpoint 转换，四项通过后再替换训练调度层。

### 2.2.8 <span style="color: rgb(36,91,219); background-color: inherit">评测与消融</span>

商品审核不能只报 Accuracy。大多数商品可能安全，全部预测 pass 也能得到很高准确率，却会漏掉真正重要的高风险商品。评测拆成结构、风险、证据、策略、自动化覆盖和校准六层，并按类目、风险代码、图片数量、OCR 密度、策略版本切片。

| **<span style="color: rgb(36,91,219); background-color: inherit">指标</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">定义</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">主要风险</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">评测输出</span>** |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Schema Valid Rate                                                            | 严格合同一次解析成功率                                                                  | 服务解析失败                                                                         | Schema 错误分布                                                                    |
| Risk Macro-F1                                                                | 各风险代码 F1 宏平均                                                                 | 长尾被高频类掩盖                                                                       | 逐类 P/R/F1                                                                      |
| High-risk FNR                                                                | 高风险未判 reject 的比例                                                             | 漏审                                                                             | 高风险漏审清单                                                                        |
| Auto-reject FPR                                                              | 合规商品被自动拒绝的比例                                                                 | 误杀                                                                             | 误杀切片                                                                           |
| Evidence P/R                                                                 | 风险、图片、区域联合匹配                                                                 | 伪证据                                                                            | 证据错误桶                                                                          |
| Policy Violation                                                             | 模型结论违反硬规则的比例                                                                 | 规则绕过                                                                           | reason code 分布                                                                 |
| ECE                                                                          | 外部置信度的校准误差                                                                   | 自动阈值失效                                                                         | 校准曲线                                                                           |

最少完成四组消融：仅规则、仅 SFT、SFT+GRPO、去除证据奖励。更完整的实验再加入去除硬规则门控、去除条款奖励、不同 num\_generations 和不同高风险代价。所有模型在同一冻结集、同一策略版本、同一解析器上比较，并给出 bootstrap 置信区间与错误样例。

```python
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

from .contracts import ContractError, validate_audit_output
from .policy import evaluate_hard_policy


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def expected_calibration_error(
    confidences: list[float], correct: list[bool], bins: int = 10
) -> float | None:
    if not confidences:
        return None
    if len(confidences) != len(correct) or bins <= 0:
        raise ValueError("invalid calibration inputs")
    total = len(confidences)
    error = 0.0
    for bin_index in range(bins):
        lower, upper = bin_index / bins, (bin_index + 1) / bins
        indices = [
            index
            for index, value in enumerate(confidences)
            if lower <= value <= upper and (bin_index == bins - 1 or value < upper)
        ]
        if not indices:
            continue
        accuracy = sum(1.0 for index in indices if correct[index]) / len(indices)
        confidence = sum(confidences[index] for index in indices) / len(indices)
        error += len(indices) / total * abs(accuracy - confidence)
    return error


def evaluate(
    gold_records: Iterable[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    high_risk_codes: set[str],
) -> dict[str, Any]:
    records = list(gold_records)
    schema_valid = 0
    risk_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    high_risk_total = high_risk_missed = safe_total = safe_auto_rejected = 0
    manual_review = policy_violations = 0
    confidences: list[float] = []
    confidence_correct: list[bool] = []
    for record in records:
        gold = validate_audit_output(record["ground_truth"], image_count=len(record["media"]))
        envelope = predictions.get(record["sample_id"], {})
        raw_prediction = envelope.get("output")
        try:
            prediction = validate_audit_output(raw_prediction, image_count=len(record["media"]))
            schema_valid += 1
        except (ContractError, TypeError, KeyError):
            prediction = {
                "decision": "review",
                "risk_codes": [],
                "evidence": [],
                "policy_refs": [],
                "explanation": "invalid output",
            }
        gold_risks, predicted_risks = set(gold["risk_codes"]), set(prediction["risk_codes"])
        for risk in gold_risks | predicted_risks:
            if risk in gold_risks and risk in predicted_risks:
                risk_stats[risk]["tp"] += 1
            elif risk in predicted_risks:
                risk_stats[risk]["fp"] += 1
            else:
                risk_stats[risk]["fn"] += 1
        if gold_risks & high_risk_codes:
            high_risk_total += 1
            if prediction["decision"] != "reject":
                high_risk_missed += 1
        if gold["decision"] == "pass":
            safe_total += 1
            if prediction["decision"] == "reject":
                safe_auto_rejected += 1
        if prediction["decision"] == "review":
            manual_review += 1
        context = {
            "title": record["title"],
            "category": record["category"],
            "attributes": record["attributes"],
            "ocr_tokens": record["ocr_tokens"],
        }
        hard_result = evaluate_hard_policy(context, policy)
        if hard_result.decision == "reject" and prediction["decision"] != "reject":
            policy_violations += 1
        confidence = envelope.get("external_confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and 0 <= confidence <= 1:
            confidences.append(float(confidence))
            confidence_correct.append(prediction["decision"] == gold["decision"])
    f1_by_risk: dict[str, float] = {}
    for risk, values in sorted(risk_stats.items()):
        precision = safe_divide(values["tp"], values["tp"] + values["fp"])
        recall = safe_divide(values["tp"], values["tp"] + values["fn"])
        f1_by_risk[risk] = safe_divide(2 * precision * recall, precision + recall)
    macro_f1 = sum(f1_by_risk.values()) / len(f1_by_risk) if f1_by_risk else 0.0
    return {
        "sample_count": len(records),
        "schema_valid_rate": safe_divide(schema_valid, len(records)),
        "risk_macro_f1": macro_f1,
        "risk_f1": f1_by_risk,
        "high_risk_false_negative_rate": safe_divide(high_risk_missed, high_risk_total),
        "auto_reject_false_positive_rate": safe_divide(safe_auto_rejected, safe_total),
        "manual_review_rate": safe_divide(manual_review, len(records)),
        "policy_violation_rate": safe_divide(policy_violations, len(records)),
        "ece": expected_calibration_error(confidences, confidence_correct),
    }
```

### 2.2.9 <span style="color: rgb(36,91,219); background-color: inherit">部署与兜底</span>

在线链路采用“硬规则 → 多模态模型 → 协议校验 → 外部校准 → 决策路由”。硬规则 reject 直接短路模型；模型输出非法、与 review 规则冲突或外部置信度低时进入人工复核。只有校准集上达到阈值的 pass/reject 才允许自动执行。

外部置信度可以来自冻结校准集上的温度缩放、专门的正确性判别器或分风险校准模型，但不能直接使用模型在答案中写出的数字。上线时同时监控协议失败率、风险分布、规则冲突率、人工复核率、各风险的 P50/P95 延迟和输入图片数量；任何策略切换都要支持按版本回滚。

| **<span style="color: rgb(36,91,219); background-color: inherit">异常</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">默认动作</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">原因</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">监控</span>** |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 硬规则 reject                                                                   | 直接拒绝                                                                           | 确定性红线不能被模型覆盖                                                                 | 规则命中率                                                                        |
| JSON/Schema 错误                                                               | 人工复核                                                                           | 系统错误不是商品违规                                                                   | 协议失败率                                                                        |
| 规则与模型冲突                                                                      | 人工复核                                                                           | 保留审核证据                                                                       | 冲突率                                                                          |
| 低外部置信度                                                                       | 人工复核                                                                           | 控制自动化风险                                                                      | 覆盖/风险曲线                                                                      |
| 模型超时                                                                         | 重试后复核                                                                          | 避免静默放行                                                                       | P95/超时率                                                                      |

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from .contracts import ContractError, parse_audit_output
from .policy import PolicyResult, evaluate_hard_policy


@dataclass(frozen=True)
class ServiceDecision:
    action: str
    reason: str
    policy_version: str
    output: dict[str, Any] | None
    hard_policy: dict[str, Any]


def _serialize_policy(result: PolicyResult) -> dict[str, Any]:
    return {
        "policy_version": result.policy_version,
        "decision": result.decision,
        "hits": [asdict(hit) for hit in result.hits],
    }


def review_product(
    request: dict[str, Any],
    policy: dict[str, Any],
    model_generate: Callable[[dict[str, Any]], Any],
    external_confidence: Callable[[dict[str, Any], dict[str, Any]], float],
    auto_action_threshold: float = 0.95,
) -> ServiceDecision:
    if not 0.0 <= auto_action_threshold <= 1.0:
        raise ValueError("auto_action_threshold must be between zero and one")
    hard_result = evaluate_hard_policy(request, policy)
    serialized = _serialize_policy(hard_result)
    if hard_result.decision == "reject":
        return ServiceDecision("reject", "hard_policy_reject", policy["policy_version"], None, serialized)
    try:
        prediction = parse_audit_output(
            model_generate(request), image_count=len(request.get("images", []))
        ).value
    except (ContractError, TypeError, ValueError):
        return ServiceDecision("manual_review", "invalid_model_output", policy["policy_version"], None, serialized)
    if hard_result.decision == "review" and prediction["decision"] == "pass":
        return ServiceDecision(
            "manual_review", "policy_model_conflict", policy["policy_version"], prediction, serialized
        )
    confidence = external_confidence(request, prediction)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        return ServiceDecision(
            "manual_review", "invalid_external_confidence", policy["policy_version"], prediction, serialized
        )
    if prediction["decision"] == "review" or confidence < auto_action_threshold:
        return ServiceDecision(
            "manual_review", "low_confidence_or_model_review", policy["policy_version"], prediction, serialized
        )
    return ServiceDecision(prediction["decision"], "calibrated_model_action", policy["policy_version"], prediction, serialized)
```

服务压测固定 Qwen3-VL revision、LoRA adapter、图片数量、最大像素和输出长度，记录 P50/P95、有效吞吐、峰值显存、Schema 失败率与人工复核率。

### 2.2.10 `简历书写`

> **<span style="color: rgb(36,91,219); background-color: inherit">项目名称：多模态商品审核系统</span>**
>
> * 负责商品标题、属性、OCR 和多图联合审核，输出 pass、reject、review、风险代码、证据位置和命中策略，覆盖信息冲突、证据不足与高风险类目；
>
> * 使用 SFT 建立结构化审核协议，再用 GRPO 优化规则遵循、风险识别和证据定位；强制拦截规则由独立 Policy Engine 执行，不依赖模型自由判断；
>
> * 按风险等级评测召回、误杀、漏审、证据覆盖和人工复核量，同时压测 P50/P95 延迟、吞吐和峰值显存，低置信及策略冲突样本统一转人工复核。

### 2.2.11 `面试官问`

1. **<span style="color: rgb(36,91,219); background-color: inherit">为什么商品审核不能只看 Accuracy？</span>**
   回答要点：类别分布和错误代价不对称。评测代码按风险切片报告 Macro-F1、高风险召回、误杀率、漏审率和人工复核率。

2. **<span style="color: rgb(36,91,219); background-color: inherit">模型输出的 confidence 能直接放行吗？</span>**
   回答要点：不能。`service.py` 使用外部校准阈值、证据完整性和硬策略共同路由，模型自报 confidence 只作为输入信号。

3. **<span style="color: rgb(36,91,219); background-color: inherit">策略更新后怎样避免标签污染？</span>**
   回答要点：每条样本绑定 `policy_version`；新策略先触发重标和回归评测，旧标签不能直接混入新版本训练。

4. **<span style="color: rgb(36,91,219); background-color: inherit">GRPO 零方差组为什么不更新？</span>**
   回答要点：组内 reward 全相同时标准差接近零，优势项没有排序信息。奖励代码记录该组并跳过更新，避免制造伪梯度。

5. **<span style="color: rgb(36,91,219); background-color: inherit">误杀、漏审和复核怎样权衡？</span>**
   回答要点：按风险等级配置不同代价和阈值，高风险优先召回，低风险控制误杀，不确定样本进入人工复核。

6. **<span style="color: rgb(36,91,219); background-color: inherit">为什么 SFT 后还要 GRPO？</span>**
   回答要点：SFT 先学会输出协议和基础审核；GRPO 在多候选上继续优化规则遵循、证据质量和错误代价，硬规则仍不交给模型学习。

---

[Previous](01-基础知识.md) | [Contents](../../README.md) | [Next](03-VLM-应用.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/projects.html#c=2)

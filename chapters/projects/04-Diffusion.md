[Previous](03-VLM-应用.md) | [Contents](../../README.md) | [Next](05-统一理解生成.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/projects.html#c=4)

# 4. <span style="color: rgb(36,91,219); background-color: inherit">Diffusion</span>

## 4.1 <span style="color: rgb(36,91,219); background-color: inherit">可控图像生成系统（ControlNet + LoRA）</span>

### 4.1.1 <span style="color: rgb(36,91,219); background-color: inherit">项目目标</span>

输入是一张商品 Canny 边缘图和一条营销描述，输出 1024×1024 商品图。SDXL 负责生成，ControlNet 约束轮廓与版式，LoRA 适配商品域或视觉风格；推理记录模型 revision、ControlNet、LoRA、seed 和采样参数。

ControlNet 和 LoRA 分别验收：ControlNet 衡量位置与轮廓保持，LoRA 衡量商品类目或视觉风格适配。联合实验使用 ControlNet 开/关 × LoRA 开/关的 2×2 对照，拆解结构控制、风格适配和交互项。

数据校验、LoRA/ControlNet 训练、联合推理和结构评测共用一份运行配置。配置里固定数据清单、模型 revision、seed、control scale 与 LoRA scale，评测脚本写出逐样本结果和错误切片。

| **<span style="color: rgb(36,91,219); background-color: inherit">方案</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">模块</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">作用</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">指标</span>** |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Base                                                                         | SDXL                                                                         | 通用生成基线                                                                       | 文本一致性、质量、失败类型                                                                |
| Control-only                                                                 | SDXL + ControlNet                                                            | 空间与轮廓控制                                                                      | Edge F1、轮廓错位率                                                                |
| LoRA-only                                                                    | SDXL + LoRA                                                                  | 商品域或风格适配                                                                     | 盲测偏好、过拟合切片                                                                   |
| Joint                                                                        | SDXL + ControlNet + LoRA                                                     | 同时控制结构与风格                                                                    | 2×2 ablation、成本、安全                                                           |

### 4.1.2 <span style="color: rgb(36,91,219); background-color: inherit">模型方案</span>

ControlNet 冻结原扩散模型，在编码路径旁增加可训练副本，再通过零初始化卷积注入控制残差。第一种控制条件使用 Canny，轮廓、占位和版式都能通过 Edge F1 与错位率检查。

LoRA 不复制一套完整 SDXL，而是在注意力层加入低秩更新，只优化少量适配参数。它适合管理多个商品域或品牌版本，也便于下线某个数据许可受影响的适配器。<span style="color: rgb(100,37,208); background-color: inherit">主线采用“预训练 Canny ControlNet + 自训 LoRA”</span>：先复用成熟控制器，再把训练预算放在已授权业务风格上。只有控制器跨分桶稳定失效时，才进入可选的 ControlNet 微调。

> **<span style="color: rgb(36,91,219); background-color: inherit">参数冲突怎么判断</span>**
>
> ControlNet scale 过低会漏控，过高会把断边、背景纹理和僵硬轮廓一起复制；LoRA scale 过低时风格不足，过高时可能记忆训练图、污染触发词或把结构推偏。因此不能凭单张样例选参数，而要在同一批 prompt、condition 与 seed 上扫描二维网格。

### 4.1.3 <span style="color: rgb(36,91,219); background-color: inherit">数据设计</span>

生产数据只接受自有拍摄、内部设计资产或许可范围明确的第三方素材。每条记录必须能通过 `license_id` 回到授权台账；“网页可以访问”不是训练许可。涉及商标、人物肖像或合作方素材时，还要记录用途、地域、期限、可否生成衍生内容以及删除流程。

目标图、条件图和 caption 必须一一对应。切分不能按单张图片随机抽样，而要按商品主体、同一视频来源或拍摄批次形成 `capture_group`，同一组只能进入一个 split。这样可以防止相邻帧、连拍和轻微裁剪图跨集合泄漏，把“记住训练图”误判成泛化。

```json
{"sample_id":"sku_001","image":"images/sku_001.png","conditioning_image":"conditions/sku_001.png","text":"studio product photo of an authorized red package on a clean background","subject_id":"sku_red","capture_group":"shoot_202607_batch01","license_id":"LIC-2026-001","split":"train","condition_type":"canny","sha256_image":"<64-hex>","sha256_conditioning":"<64-hex>"}
```

| **<span style="color: rgb(36,91,219); background-color: inherit">字段</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">作用</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">错误时的风险</span>** |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| capture\_group                                                               | 主体/拍摄批次级切分                                                                   | 近重复图泄漏到 Test                                                                     |
| license\_id                                                                  | 回溯授权与删除请求                                                                    | 无法判断样本是否合法                                                                       |
| condition\_type                                                              | 选择条件预处理与指标                                                                   | 不同任务共用错误指标                                                                       |
| SHA-256                                                                      | 冻结目标图与条件图内容                                                                  | 训练后文件被静默替换                                                                       |

### 4.1.4 <span style="color: rgb(36,91,219); background-color: inherit">条件图</span>

Canny 条件图不是一次性的中间文件，而是数据版本的一部分。低/高阈值、缩放方式、裁剪区域和输出尺寸都要固定。条件图与目标图必须采用同一裁剪并保持相同尺寸；否则模型会在训练时学习到系统性错位，推理时再高的 ControlNet scale 也只能放大错误。

```powershell
python scripts/build_canny.py `
  --input-dir data/raw/images `
  --output-dir data/processed/conditions `
  --low 100 `
  --high 200

python scripts/validate_manifest.py `
  --manifest data/manifest.jsonl `
  --root data
```

质量检查先做机械门禁：路径不能越过数据根目录，文件必须存在，目标图与条件图尺寸一致，哈希匹配，`sample_id` 唯一，`capture_group` 不跨 split。再做抽样目检：边缘是否断裂、背景纹理是否过密、商品内部文字是否被当成强边缘、透明区域是否出现伪轮廓。修改 Canny 阈值后必须重算条件图与哈希，并作为独立实验版本评测。

### 4.1.5 <span style="color: rgb(36,91,219); background-color: inherit">LoRA 训练</span>

LoRA 使用训练集目标图和 caption，并由最终 manifest 生成 Diffusers ImageFolder。第一轮搜索点设为 UNet attention LoRA、rank 16、学习率 1e-4、3000 steps、BF16 与 gradient checkpointing；text encoder、rank、步数和学习率在 Validation 上做单变量比较。

```powershell
python scripts/prepare_lora_imagefolder.py `
  --manifest data/manifest.jsonl `
  --root data `
  --output data/lora_imagefolder

python scripts/print_train_command.py `
  --stage lora `
  --config configs/lora_train.json `
  --diffusers-root D:\src\diffusers
```

训练前填入 `configs/lora_train.json` 的 Diffusers commit、基础模型 revision 和数据版本。每个 checkpoint 使用相同验证 prompt 与 seed 生成对照图，记录欠拟合、训练图记忆、颜色偏置、固定视角和触发词污染；收益归因对齐配置、日志、checkpoint 和样本级评分。

### 4.1.6 <span style="color: rgb(36,91,219); background-color: inherit">ControlNet 基线</span>

第一阶段直接加载 `diffusers/controlnet-canny-sdxl-1.0`，按商品类型、背景复杂度和边缘密度分桶扫描 scale。低密度条件图主要观察漏轮廓，高密度条件图主要观察背景纹理被过度复制。先调整阈值、裁剪和 scale，再决定是否需要训练控制器。

ControlNet 自训练设置三项准入条件：业务条件分布与预训练 Canny 存在稳定差异；失败跨多个 seed 和数据分桶重复出现；数据错位、坏边缘和参数问题已经排除。微调从预训练 Canny ControlNet 初始化，学习率以 1e-5 为搜索起点，围绕 10000 steps 保存中间 checkpoint 并在 Validation 上选型。

```powershell
python scripts/build_hf_controlnet_dataset.py `
  --manifest data/manifest.jsonl `
  --root data `
  --output data/hf_dataset `
  --hub-id ORG/PRIVATE_CONTROLNET_DATASET

python scripts/print_train_command.py `
  --stage controlnet `
  --config configs/controlnet_train.json `
  --diffusers-root D:\src\diffusers
```

<span style="color: rgb(216,57,49); background-color: inherit">如果项目最终只复用了预训练 ControlNet，就不能在简历中写“训练了 ControlNet”。</span>可以写“构建并评测 ControlNet 控制链路”，并把自训练设计明确标成备用方案。

### 4.1.7 <span style="color: rgb(36,91,219); background-color: inherit">联合推理</span>

推理端使用 `StableDiffusionXLControlNetPipeline`。先加载固定 revision 的 SDXL 与 ControlNet，再用 `load_lora_weights()` 注册 LoRA，用 `set_adapters()` 设置适配器权重。每次请求固定两种 scale、prompt、negative prompt、采样步数、分辨率和 seed，生成图片旁边写出同名 `.run.json`。

```python
def generate(
    *,
    config_path: Path,
    control_image_path: Path,
    lora_path: str,
    output_path: Path,
) -> dict[str, Any]:
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("install requirements-ml.txt before model inference") from exc

    config = load_config(config_path)
    dtype_name = str(config.get("dtype", "float16"))
    dtype = getattr(torch, dtype_name)
    controlnet = ControlNetModel.from_pretrained(
        config["controlnet_model"],
        revision=config.get("controlnet_revision"),
        torch_dtype=dtype,
    )
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        config["base_model"],
        revision=config.get("base_revision"),
        controlnet=controlnet,
        torch_dtype=dtype,
    ).to(config.get("device", "cuda"))
    pipeline.load_lora_weights(
        lora_path,
        weight_name=config.get("lora_weight_name", "pytorch_lora_weights.safetensors"),
        adapter_name="brand",
    )
    pipeline.set_adapters(["brand"], adapter_weights=[float(config["lora_scale"])])

    control_image = Image.open(control_image_path).convert("RGB").resize(
        (int(config["width"]), int(config["height"]))
    )
    generator = torch.Generator(device=config.get("device", "cuda")).manual_seed(int(config["seed"]))
    result = pipeline(
        prompt=config["prompt"],
        negative_prompt=config["negative_prompt"],
        image=control_image,
        controlnet_conditioning_scale=float(config["controlnet_conditioning_scale"]),
        guidance_scale=float(config["guidance_scale"]),
        num_inference_steps=int(config["num_inference_steps"]),
        generator=generator,
        width=int(config["width"]),
        height=int(config["height"]),
    ).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    run_record = make_run_record(
        config,
        seed=int(config["seed"]),
        base_revision=str(config.get("base_revision", "UNPINNED")),
        controlnet_revision=str(config.get("controlnet_revision", "UNPINNED")),
        lora_revision=str(config.get("lora_revision", "LOCAL_UNPINNED")),
    )
    output_path.with_suffix(".run.json").write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_record
```

```powershell
python scripts/generate.py `
  --config configs/inference.json `
  --control-image data/eval/conditions/sku_101.png `
  --lora artifacts/lora `
  --output outputs/sku_101_seed_20260729.png
```

服务层至少返回 `request_id`、输出 URI、配置哈希、三个模型 revision、seed、延迟与安全决策。线上不要把 LoRA 永久 fuse 进共享模型后再切换品牌；独立适配器更容易热切换、回滚和下线。缓存键必须包含条件图哈希、prompt、模型版本、两种 scale 与 seed，防止不同请求误命中同一结果。

### 4.1.8 <span style="color: rgb(36,91,219); background-color: inherit">评测与消融</span>

评测按结构、文本、风格、人工、工程和安全六层展开。Canny 方案对生成图重新提边，用带像素容差的 Precision、Recall、F1 衡量结构遵循；姿态使用 PCK，分割使用 mIoU，深度使用 AbsRel/RMSE，各控制类型采用对应的几何指标。

| **<span style="color: rgb(36,91,219); background-color: inherit">评测层</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">指标/方法</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">回答的问题</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">主要限制</span>** |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 结构                                                                            | Edge P/R/F1 + 像素容差                                                              | 轮廓是否跟随条件图                                                                       | 衡量文本与图像的语义一致性                                                                  |
| 文本                                                                            | CLIP 类相似度                                                                       | 语义是否匹配 prompt                                                                   | 不证明结构/身份                                                                       |
| 分布                                                                            | FID/KID/CMMD                                                                    | 生成集与真实集距离                                                                       | 依赖样本量与预处理                                                                      |
| 风格/主体                                                                         | 冻结视觉编码器 + 分桶                                                                    | LoRA 是否学到目标域                                                                    | 可能奖励近重复                                                                        |
| 人工                                                                            | 双盲偏好 + 缺陷标签                                                                     | 素材是否可用                                                                          | 需要随机化与区间                                                                       |
| 工程/安全                                                                         | P95、显存、失败率、错放/误拒                                                                | 能否稳定上线                                                                          | 必须注明硬件与并发                                                                      |

必做消融是 Base、Control-only、LoRA-only、Joint。Joint 方案扫描 `control_scale ∈ {0.5, 0.8, 1.0}` 与 `lora_scale ∈ {0.5, 0.7, 0.9}`，固定同一组 prompt、condition、seed、步数和分辨率。之后再分别比较 LoRA rank、训练步数、Canny 阈值、分辨率与采样步数。

```powershell
python scripts/evaluate_edges.py `
  --prediction-edge outputs/sku_101.edge.png `
  --reference-edge data/eval/conditions/sku_101.png `
  --tolerance 1.5

python -m unittest discover -s tests -v
```

失败样本按原因归档：断边或背景纹理过密；ControlNet 漏控或过约束；LoRA 欠拟合、过拟合、触发词污染；风格与结构冲突；prompt 和条件图语义冲突；基础模型在文字、细小 Logo、人脸、手部或多对象关系上的限制；以及授权、肖像和商标风险。修复时先用相同 seed 和配置重放，再单独关闭 LoRA 或 ControlNet，最后才修改数据或训练。

### 4.1.9 `简历书写`

> **<span style="color: rgb(36,91,219); background-color: inherit">项目名称：可控图像生成系统</span>**
>
> * 基于 SDXL 搭建商品图可控生成链路，使用 Canny ControlNet 约束轮廓与构图，使用 LoRA 学习授权商品域的主体和风格特征；
>
> * 按拍摄批次切分训练与测试数据，统一原图和条件图的裁剪增强，完成 Base、Control-only、LoRA-only、Joint 四组 ablation，并扫描 ControlNet 与 LoRA 权重；
>
> * 从结构一致性、主体或风格保持、人工盲测和安全门禁四个维度评测生成结果，保存固定模型 revision、逐样本分数和失败样本切片。

### 4.1.10 `面试官问`

1. **<span style="color: rgb(36,91,219); background-color: inherit">为什么不用全量微调 SDXL？</span>**
   回答要点：项目需要维护多个授权商品域。LoRA 权重更小，切换、回滚和删除受影响版本更直接；全量微调必须额外证明收益足以覆盖训练和权重治理成本。

2. **<span style="color: rgb(36,91,219); background-color: inherit">ControlNet 和 LoRA 冲突怎样定位？</span>**
   回答要点：固定 prompt、condition 与 seed，依次运行 Base、Control-only、LoRA-only 和 Joint，再扫描两种 scale，先定位单模块问题，再判断联合冲突。

3. **<span style="color: rgb(36,91,219); background-color: inherit">为什么不能只看 CLIPScore？</span>**
   回答要点：它不能证明边缘忠实、主体保持、品牌风格和内容安全。还要结合 Edge F1、身份或风格指标、人工盲测和安全门禁。

4. **<span style="color: rgb(36,91,219); background-color: inherit">什么时候值得自训 ControlNet？</span>**
   回答要点：预训练控制器在跨分桶、跨 seed 验证集上持续失败，并且阈值、裁剪、尺寸对齐和 scale 已经排除后再训练。

5. **<span style="color: rgb(36,91,219); background-color: inherit">授权样本删除怎样处理？</span>**
   回答要点：根据 `license_id` 和媒体哈希定位样本，冻结受影响 LoRA revision，重建清单并重新训练或回滚。

6. **<span style="color: rgb(36,91,219); background-color: inherit">实验怎样保证可追溯？</span>**
   回答要点：保存数据清单哈希、模型与 ControlNet revision、LoRA 哈希、完整配置、环境指纹、逐样本输出和评测脚本。

## 4.2 <span style="color: rgb(36,91,219); background-color: inherit">低成本大规模图像生成服务</span>

### 4.2.1 <span style="color: rgb(36,91,219); background-color: inherit">项目目标</span>

请求进入网关后先做 Schema 校验，再按模型 revision、dtype、尺寸、steps、scheduler、guidance 和 LoRA 集合分桶。队列检查 deadline、显存预算和租户隔离；OOM 时有界拆批，结果通过质量门禁后写入 exact cache。`load_test.py` 记录吞吐、P50/P95、峰值显存、质量通过率和单图成本。

服务加载 4.1 训练好的基础模型、ControlNet 和 LoRA，把并发请求合并成兼容 batch。调度目标是每 GPU 小时产出的合格图片数，同时约束 P95、OOM、质量通过率和租户隔离。

| **<span style="color: rgb(36,91,219); background-color: inherit">指标</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">计算</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">同时记录</span>** |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 有效吞吐                                                                         | 质量门禁通过图片数 / 聚合 GPU 小时                                                        | 分辨率、steps、batch size、拒绝输出                                                      |
| 端到端时延                                                                        | 接收请求到最终状态的 P50/P95/P99                                                       | 排队、推理、质量门禁和编码耗时                                                                |
| 可靠性                                                                          | OOM batch 尝试率、超时率、重试成本                                                       | 首次失败与最终状态                                                                      |
| 质量                                                                           | 固定提示集上的门禁通过率                                                                 | 失败类型、steps 与 scheduler                                                         |
| 成本                                                                           | 全部尝试 GPU 时间 / 合格输出数                                                          | 缓存命中、重试和拒绝成本                                                                   |

压测使用单请求基线与动态 batch 对照，联合汇总有效吞吐、OOM 尝试率、质量门禁通过率、排队时间和单图成本。

### 4.2.2 <span style="color: rgb(36,91,219); background-color: inherit">服务架构</span>

在线主链路分成入口、调度和执行三层。入口层负责 schema、租户配额、模型白名单和幂等；调度层负责 exact cache、兼容分桶、deadline 与 batch 工作量；执行层只接收已经确定模型版本、shape 和 adapter 集合的 batch，并把结果交给质量与安全门禁。

| **<span style="color: rgb(36,91,219); background-color: inherit">阶段</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">输入</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">关键动作</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">终止状态</span>** |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Admission                                                                    | 完整 GenerationRequest                                                         | 合同、策略、队列和显存估算校验                                                                | rejected                                                                       |
| Cache                                                                        | tenant + generation hash                                                     | 只读取已通过门禁的完全一致结果                                                                | cache\_hit                                                                     |
| Scheduler                                                                    | 未命中请求                                                                        | 兼容分桶、短等待窗口和 deadline 检查                                                        | deadline\_exceeded                                                             |
| Worker                                                                       | 同构 micro-batch                                                               | Diffusers 推理、OOM 有界拆批                                                          | generated / oom\_failed                                                        |
| Guard                                                                        | 生成图片与运行元数据                                                                   | 质量、安全和策略版本校验                                                                   | guard\_rejected                                                                |

HTTP 层不是“一次请求就立刻调用一次 pipeline”。`AsyncBatchingGateway` 把并发请求收集到一个很短的窗口，再交给同步控制面统一分桶；如果队列已满或请求超过自己的 deadline，入口直接返回明确状态，不让已经失去业务价值的请求继续占用 GPU。

```python
async def generate(self, request: GenerationRequest) -> ServiceResponse:
    now_ms = int(time.monotonic() * 1000)
    if request.submitted_at_ms == 0:
        request = replace(request, submitted_at_ms=now_ms)
    future = asyncio.get_running_loop().create_future()
    self.queue.put_nowait(_Pending(request, future))
    return await asyncio.wait_for(
        asyncio.shield(future),
        timeout=request.deadline_ms / 1000.0,
    )
```

### 4.2.3 <span style="color: rgb(36,91,219); background-color: inherit">请求合同</span>

图像能否复现，取决于权重、adapter、scheduler、shape、steps、seed、提示词和运行精度共同确定的输入，而不是只记一个 prompt。请求还要携带 tenant、策略版本、idempotency key 和 deadline，分别服务于隔离、审核、传输重试和时延控制。

| **<span style="color: rgb(36,91,219); background-color: inherit">字段组</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">示例</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">为什么必须记录</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">进入缓存键</span>** |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 身份与传输                                                                         | request\_id、idempotency\_key                                                 | 定位调用和抑制重复提交                                                                       | 否                                                                               |
| 隔离与策略                                                                         | tenant\_id、policy\_version                                                   | 防止跨租户复用，策略升级后自动失效                                                                 | 是                                                                               |
| 模型产物                                                                          | model revision、adapter SHA-256                                               | revision 名称可能被覆盖，哈希绑定实际文件                                                         | 是                                                                               |
| 采样参数                                                                          | seed、steps、scheduler、guidance                                                | 任一变化都可能改变输出                                                                       | 是                                                                               |
| 时限                                                                            | submitted\_at、deadline\_ms                                                   | 控制队列生命周期                                                                          | 否                                                                               |

```json
{
  "request_id": "req-001",
  "idempotency_key": "order-1001-main",
  "tenant_id": "tenant-a",
  "prompt": "studio product photo of a red running shoe",
  "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
  "model_revision": "PIN_A_COMMIT_SHA_BEFORE_GPU_RUN",
  "width": 1024,
  "height": 1024,
  "steps": 30,
  "seed": 41001,
  "scheduler": "euler",
  "guidance_scale": 6.5,
  "policy_version": "image-policy-v1",
  "deadline_ms": 30000,
  "adapters": []
}
```

`GenerationRequest.cache_key()` 对影响生成结果和隔离边界的字段做规范化 JSON，再计算 SHA-256；它主动排除 request ID、idempotency key、提交时间和 deadline。因此同一租户的传输重试可以命中缓存，而模型 revision、adapter 文件、seed 或策略版本变化一定得到新键。

```python
def cache_key(self) -> str:
    canonical = json.dumps(
        self.generation_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

> **<span style="color: rgb(36,91,219); background-color: inherit">为什么主链路不用语义缓存</span>**
>
> 完全一致缓存按模型 revision、adapter 哈希、prompt、negative prompt、seed、尺寸、steps、scheduler 和 guidance 逐字段匹配。近似复用涉及品牌、版权与审核授权，作为独立产品能力设计。

### 4.2.4 <span style="color: rgb(36,91,219); background-color: inherit">动态批处理</span>

动态 batch 先按完整兼容键分组：基础模型 revision、dtype、width、height、steps、scheduler、guidance、输出格式和 LoRA 集合保持一致；prompt、negative prompt 与 seed 允许逐样本变化。

```python
def compatibility_key(request: GenerationRequest) -> tuple[object, ...]:
    adapters = tuple(
        (a.adapter_id, a.revision, a.artifact_sha256, a.scale)
        for a in request.adapters
    )
    return (
        request.model_id, request.model_revision, request.dtype,
        request.width, request.height, request.steps,
        request.scheduler, request.guidance_scale,
        request.output_format, adapters,
    )
```

batch 控制同时使用 `max_batch_size` 和 `max_batch_work_units`。四个 512×512、20 steps 请求与四个 1536×1536、50 steps 请求的计算量差异很大，因此用 `ceil(megapixels × steps)` 构造轻量工作量代理，再通过 GPU 压测校准系数。

> **<span style="color: rgb(36,91,219); background-color: inherit">一次具体调度</span>**
>
> 队列里同时出现 A、B 两个 1024×1024、30 steps、Euler 请求，以及 C 一个 768×1024 请求。A 与 B 进入同一桶；C 因 shape 不同进入另一桶。A/B 桶达到 preferred batch size 时立即出队，C 等到 `max_delay_ms` 后单独执行，不能为了凑 batch 无限等待。

等待窗口是在吞吐和尾延迟之间换取收益。调参顺序应是：先定单请求基线，再逐步增大 max batch 和 delay；一旦 P95 超出预算就停止。请求已经达到 deadline 时直接标记 `deadline_exceeded`，不会进入 GPU。

### 4.2.5 <span style="color: rgb(36,91,219); background-color: inherit">显存管理</span>

入口的显存估算采用保守容量模型：模型常驻显存、batch 激活、adapter 占用和安全余量相加。单请求通过准入后，batch 成形时还会按 batch size 再估一次；若预计超过预算，先拆成更小 batch，再考虑执行。

$$Memory_{peak}=Memory_{model}+c_{act}\cdot MP\cdot B+Memory_{adapter}+Headroom$$

```python
    def _fit_vram(self, batch: MicroBatch) -> list[MicroBatch]:
        estimate = self.admission.estimate_peak_mb(batch.requests[0], batch_size=len(batch.requests))
        if estimate <= self.admission.policy.gpu_budget_mb or len(batch.requests) == 1:
            return [batch]
        midpoint = len(batch.requests) // 2
        return self._fit_vram(MicroBatch(batch.requests[:midpoint])) + self._fit_vram(
            MicroBatch(batch.requests[midpoint:])
        )

    def _execute_batch(self, batch: MicroBatch, *, retries_left: int, now_ms: int) -> list[ServiceResponse]:
        try:
            result = self.backend.generate(batch)
            self.metrics.record_batch_attempt(result.elapsed_ms, len(batch.requests), oom=False)
        except BackendOOM as exc:
            self.metrics.record_batch_attempt(exc.elapsed_ms, len(batch.requests), oom=True)
            if retries_left > 0 and len(batch.requests) > 1:
                midpoint = len(batch.requests) // 2
                left = MicroBatch(batch.requests[:midpoint])
                right = MicroBatch(batch.requests[midpoint:])
                return self._execute_batch(left, retries_left=retries_left - 1, now_ms=now_ms) + self._execute_batch(
                    right, retries_left=retries_left - 1, now_ms=now_ms
                )
            failed: list[ServiceResponse] = []
            for request in batch.requests:
                failed.append(
                    ServiceResponse(request.request_id, "oom_failed", request.cache_key(), None, str(exc), None)
                )
                self._record_request(
                    request,
                    now_ms,
                    "oom_failed",
                    cache_hit=False,
                    queue_wait_ms=max(0, now_ms - request.submitted_at_ms),
                )
            return failed
```

在线 OOM 仍可能来自 allocator 碎片、attention backend 或未建模的 adapter。服务捕获 CUDA OOM 后记录这次失败尝试的 batch size 和 GPU 时间，再按二分法拆 batch；重试次数由 `oom_split_retries` 限制。单请求仍 OOM 时返回 `oom_failed`，禁止无限重试。

| **<span style="color: rgb(36,91,219); background-color: inherit">资源</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">驻留键</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">淘汰或隔离原则</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">风险</span>** |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 基础 pipeline                                                                  | model revision + dtype + compile profile                                      | 按显存预算控制 worker 池，不在请求中临时下载                                                        | 冷启动、重复常驻                                                                     |
| LoRA                                                                         | adapter revision + SHA-256                                                    | 只从批准目录加载；动态与固定 worker 分池                                                          | 串租户、重复编译                                                                     |
| 结果缓存                                                                         | tenant + generation hash                                                      | 只写门禁通过的输出，原子提交                                                                    | 策略污染、跨租户命中                                                                   |

`torch.cuda.empty_cache()` 在 OOM 异常后释放未使用的缓存块。容量控制依次执行入口拒绝、batch 显存复检、有界拆批和失败清理。

### 4.2.6 <span style="color: rgb(36,91,219); background-color: inherit">推理优化</span>

推理优化按稳定 shape 的算力池和动态 adapter 的灵活池拆开。PyTorch 2.x 下 Diffusers 默认可使用 SDPA；attention backend、channels-last、compile、量化和 offload 分别做单变量消融。

| **<span style="color: rgb(36,91,219); background-color: inherit">手段</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">主要收益</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">适用条件</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">必须复测</span>** |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| FP16/BF16                                                                    | 降低权重与计算开销                                                                      | 硬件原生支持对应 dtype                                                                 | 数值稳定与质量                                                                        |
| SDPA/attention backend                                                       | 降低 attention 显存或延迟                                                             | dtype、设备和 kernel 支持                                                            | graph break、回退路径                                                               |
| torch.compile                                                                | 稳定 workload 下减少框架开销                                                            | 固定 shape、固定 adapter 目标更合适                                                      | 冷启动、重新编译、缓存占用                                                                  |
| VAE slicing                                                                  | 多图解码降低峰值显存                                                                     | batch 大于 1 时更有意义                                                               | 解码延迟                                                                           |
| VAE tiling                                                                   | 高分辨率降低峰值显存                                                                     | 单图 VAE 解码仍超预算                                                                  | 分块色调和接缝                                                                        |
| model CPU offload                                                            | 降低 GPU 常驻显存                                                                    | 显存不足且能接受传输开销                                                                   | P95 与 PCIe 瓶颈                                                                  |

```json
{
  "model": {
    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "model_revision": "PIN_A_COMMIT_SHA_BEFORE_GPU_RUN",
    "dtype": "float16",
    "compile_unet": false,
    "lora_mode": "dynamic",
    "vae_slicing": true,
    "vae_tiling": false,
    "model_cpu_offload": false
  }
}
```

真实 backend 从受控模型仓加载 LoRA，并在加载前复算文件 SHA-256。固定 LoRA worker 可以在预热后编译；动态 LoRA worker 默认不编译，避免 adapter target、rank 或 shape 变化触发重新编译。

> offload 增加主机与设备传输，tiling 可能改变图像，量化影响质量，compile 的首次执行和 shape 变化产生额外开销。每个开关在同硬件、同模型、同请求集上单独评测，再组合进入最终配置。
>
> 优化开关逐项消融

### 4.2.7 <span style="color: rgb(36,91,219); background-color: inherit">队列与扩缩容</span>

服务维护的是每个兼容桶的等待，而不是只有一个全局 FIFO。热门桶达到 preferred batch size 可以立即执行；冷门桶达到 max delay 也必须放行。长 steps、大分辨率或冷门 adapter 可单独设置队列和并发配额，避免拖慢常规流量。

容量估算可用 Little 定律检查是否自洽：$$L=\lambda W$$。这里的 $$\lambda$$ 必须是质量门禁后的可持续到达率，$$W$$ 是端到端时延；用峰值到达率或只用 kernel 延迟会低估所需并发。

| **<span style="color: rgb(36,91,219); background-color: inherit">触发信号</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">优先动作</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">不要直接做</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">观察指标</span>** |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| queue wait P95 上升                                                              | 检查桶分布，再扩对应 worker 池                                                            | 不分模型和 adapter 地全局扩容                                                             | 各桶深度、超时率                                                                       |
| GPU 利用率低且队列有量                                                                  | 增大兼容 batch 或短幅增加 delay                                                         | 牺牲 deadline 无限等 batch                                                           | batch size、E2E P95                                                             |
| OOM 尝试率上升                                                                      | 收紧 batch/shape 预算并校准估算                                                         | 依靠重试掩盖容量错误                                                                      | 失败 batch 分布、显存峰值                                                               |
| adapter 冷启动增多                                                                  | 按访问频率建立固定/动态池                                                                  | 把所有 LoRA 永久常驻一张卡                                                                | 加载时间、命中率、驻留显存                                                                  |

多 GPU 下，每个 worker 自己维护 pipeline 与 adapter 状态，路由层按兼容键和 revision 选择 worker。不能假设进程内缓存跨 worker 共享；exact cache 应落在对象存储或共享缓存中，并保持 tenant namespace。发布新模型时先启动、验哈希、预热和 smoke test，再接流量，旧 revision 等在途请求结束后下线。

### 4.2.8 <span style="color: rgb(36,91,219); background-color: inherit">压测与成本</span>

FakeBackend 生成确定性的 batch 时间、OOM 和门禁结果，用于回归调度状态机。GPU backend 固定 GPU、驱动、CUDA、PyTorch、Diffusers、容器摘要、模型 commit、adapter 哈希、提示集和质量门禁版本，记录吞吐、延迟、显存与质量。

| **<span style="color: rgb(36,91,219); background-color: inherit">组别</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">Batch</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">Cache</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">Compile</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">显存策略</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">回答的问题</span>** |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| B0                                                                           | 1                                                                               | off                                                                             | off                                                                               | 常驻 GPU                                                                         | 单请求基线                                                                           |
| B1                                                                           | dynamic                                                                         | off                                                                             | off                                                                               | 常驻 GPU                                                                         | 只测 micro-batch                                                                  |
| B2                                                                           | dynamic                                                                         | exact                                                                           | off                                                                               | 常驻 GPU                                                                         | 缓存真实复用                                                                          |
| B3                                                                           | dynamic                                                                         | off                                                                             | on                                                                                | 固定 shape/adapter                                                               | 稳态编译收益                                                                          |
| B4                                                                           | dynamic                                                                         | off                                                                             | off                                                                               | VAE slicing                                                                    | 多图解码显存                                                                          |
| B5                                                                           | dynamic                                                                         | off                                                                             | off                                                                               | VAE tiling                                                                     | 大图显存与接缝                                                                         |
| B6                                                                           | dynamic                                                                         | off                                                                             | off                                                                               | model offload                                                                  | 显存与传输交换                                                                         |

有效吞吐采用

$$Throughput_{valid}=N_{accepted}/T_{GPU-hour}$$

单张有效图片 GPU 成本采用

$$Cost_{image}=T_{GPU-hour}\cdot Price_{GPU-hour}/N_{accepted}$$

失败 batch、OOM 重试和质量拒绝消耗的 GPU 时间都留在分子中。

```python
gpu_seconds = sum(
    attempt.elapsed_ms for attempt in self.batch_attempts
) / 1000.0
total_gpu_cost = gpu_seconds / 3600.0 * gpu_hour_price
cost_per_accepted = (
    total_gpu_cost / accepted_outputs
    if accepted_outputs else None
)
```

如果云厂商按实例墙钟时间计费，还要补充空闲、模型预热和滚动发布重叠时间。CUDA 事件时间适合解释 kernel 与 batch，但不能单独代替真实账单。

### 4.2.9 <span style="color: rgb(36,91,219); background-color: inherit">故障演练</span>

上线前至少演练四类失败：单 batch OOM、单请求 OOM、队列超时和 adapter 哈希不匹配。每次演练都要验证终止状态、重试次数、成本记录、缓存未污染以及旧 revision 能否继续服务。

| **<span style="color: rgb(36,91,219); background-color: inherit">故障</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">预期降级</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">必须留痕</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">上线阻断条件</span>** |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| batch OOM                                                                    | 预算内二分拆批                                                                        | 原 batch、失败 GPU 时间、重试树                                                          | 无限重试或漏记成本                                                                        |
| 单请求 OOM                                                                      | oom\_failed，释放在途资源                                                             | shape、steps、adapter 和显存                                                        | worker 反复崩溃                                                                      |
| 队列超时                                                                         | deadline\_exceeded，不进 GPU                                                      | 入队、出队与桶深度                                                                      | 超时后仍执行                                                                           |
| 权重哈希不符                                                                       | worker not ready                                                               | 期望与实际 SHA-256                                                                  | 继续加载或降级到未知权重                                                                     |
| 质量门禁拒绝                                                                       | guard\_rejected，不写缓存                                                           | 门禁版本和原因                                                                        | 把拒绝输出计入有效吞吐                                                                      |

调度回归覆盖缓存键、shape 兼容、work budget、deadline、入口准入、batch 显存复检、OOM 有界拆批和质量拒绝；`load_test.py` 对 GPU backend 生成容量、质量与成本报告。

`capacity_report.py` 读取 GPU 压测结果，汇总硬件、请求分布、吞吐、时延、峰值显存和质量样本。调度回归使用 `--allow-simulated` 读取 FakeBackend 结果，两类报告使用不同的 `report_type`。

### 4.2.10 `简历书写`

> **<span style="color: rgb(36,91,219); background-color: inherit">项目名称：低成本大规模图像生成服务</span>**
>
> * 基于 Diffusers 搭建异步推理服务，按模型、分辨率、采样参数和 LoRA 组合执行兼容 micro-batching，并为请求设置显存预算与 deadline；
>
> * 实现租户隔离 exact cache、LoRA 动态加载、显存准入和 OOM 有界拆批，分别管理固定 adapter 池与动态 adapter 池的 compile 策略；
>
> * 在固定 GPU 和请求分布下压测有效吞吐、E2E P95、峰值显存、质量通过率和单图成本，容量统计只计入通过质量门禁的图片。

### 4.2.11 `面试官问`

1. **<span style="color: rgb(36,91,219); background-color: inherit">为什么不能只报 QPS？</span>**
   回答要点：分辨率、steps、缓存命中和质量拒绝会改变单请求计算量。主指标应是固定请求分布下的合格图片数/GPU-hour，并同时报告 E2E P95。

2. **<span style="color: rgb(36,91,219); background-color: inherit">adapter 为什么必须绑定 SHA-256？</span>**
   回答要点：revision 名称可能被覆盖，文件哈希才能绑定实际权重；worker 加载日志和缓存键都使用该哈希。

3. **<span style="color: rgb(36,91,219); background-color: inherit">compile 为什么不与动态 LoRA 默认同开？</span>**
   回答要点：adapter 的 target、rank 或 shape 变化可能触发重新编译并污染 P95。固定模型、adapter 和 shape 的流量更适合专用编译池。

4. **<span style="color: rgb(36,91,219); background-color: inherit">OOM 为什么按 batch 尝试统计？</span>**
   回答要点：一次失败 batch 会消耗 GPU 时间并影响多条请求，只看最终请求失败率会掩盖重试成本和容量模型偏差。

5. **<span style="color: rgb(36,91,219); background-color: inherit">吞吐提升怎样验收？</span>**
   回答要点：固定硬件、请求混合、模型 revision 和质量门禁，对比基线与优化方案的原始日志、有效图片数、P95、显存和拒绝率。

系统在固定 GPU 型号和质量门禁下优化每 GPU 小时的合格图片数。请求携带租户、策略版本、模型与 LoRA 版本、尺寸、步数和种子，用于计费、复现、审计和删除。

---

[Previous](03-VLM-应用.md) | [Contents](../../README.md) | [Next](05-统一理解生成.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/projects.html#c=4)

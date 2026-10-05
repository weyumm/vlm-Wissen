# 视觉多模态知识库

围绕视觉基础与 Vision-Language Model（VLM）的中文讲义、面试题与项目实战，共四个分册。

## 在线阅读

从 [`index.html`](index.html) 进入总目录，再跳到对应分册。每一册是**独立页面**，浏览器同一时间只需处理其中一册，避免单页过大。

| 分册 | 页面 | 内容 |
|---|---|---|
| 视觉多模态讲义（上） | [`lecture-1.html`](lecture-1.html) | 视觉基础、骨干网络、对比学习、图像分割、VLM（2 章 / 13 节 / 277 图） |
| 视觉多模态讲义（下） | [`lecture-2.html`](lecture-2.html) | GAN、自编码器、流模型、扩散模型、统一理解生成模型 UMM（3 章 / 15 节 / 303 图） |
| 视觉多模态面试题 | [`interview.html`](interview.html) | 模型原理、对比分析、架构细节、训练微调、生成推理、真题（7 章 / 22 节 / 74 图） |
| 视觉多模态项目 | [`projects.html`](projects.html) | LLaMA-Factory 环境、多模态数据、VLM 后训练与 Diffusion 实战（5 章 / 11 节 / 36 图） |

阅读页提供：章节目录与滚动高亮、正文全文搜索、`$$LaTeX$$` 公式渲染、图片懒加载与点击放大、阅读进度、深色模式。

## 本地预览

必须通过静态服务器打开（浏览器会阻止 `file://` 下的 `fetch`）：

```bash
python -m http.server 8000
```

随后访问 `http://localhost:8000`。

## 目录结构

```
index.html              总目录（入口）
lecture-1.html          讲义（上）阅读页
lecture-2.html          讲义（下）阅读页
interview.html          面试题阅读页
projects.html           项目实战阅读页
assets/site.css         共享样式
assets/parts.js         四个分册的元信息（单一数据源）
assets/reader.js        阅读页渲染逻辑（Markdown → HTML → 公式 → 目录/搜索）
assets/hub.js           总目录卡片渲染
assets/theme.js         深色模式与讲义配色的深色映射
assets/vendor/          本地化的 marked / DOMPurify / KaTeX（无 CDN 依赖）
images/                 690 张配图，以所属分册名前缀区分
视觉多模态讲义（上）.md  各分册 Markdown 原文
视觉多模态讲义（下）.md
视觉多模态面试题.md
视觉多模态项目.md
```

网页直接读取仓库中的 Markdown 与图片，**不依赖任何 CDN 或构建步骤**，离线可用。新增分册时只需放入 `.md` 与图片，并在 `assets/parts.js` 中追加一项。

## GitHub Pages

在 GitHub 仓库的 **Settings → Pages** 中，将 **Build and deployment** 设为 **Deploy from a branch**，分支选 `main`、目录选 `/ (root)` 并保存。发布完成后，GitHub 会提供站点地址。仓库根目录已放置 `.nojekyll`，跳过 Jekyll 处理。

## 许可

讲义与配图版权归原作者所有。第三方库见 `assets/vendor/` 下的 LICENSE 文件：
[marked](assets/vendor/LICENSE-marked.md)（MIT）、[DOMPurify](assets/vendor/LICENSE-dompurify.txt)（MPL-2.0 / Apache-2.0）、[KaTeX](assets/vendor/katex/LICENSE.txt)（MIT）。

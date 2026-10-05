# 视觉多模态知识库

围绕视觉基础与 Vision-Language Model（VLM）的中文讲义、面试题与项目实战，共四个分册、17 个章节。

## 在线阅读

从 [`index.html`](index.html) 进入总目录，再跳到对应分册。每一册是**独立页面**，而且**一次只渲染一章**——浏览器同一时间只需要处理一章的内容，不会因为整册过大而吃满显存。

| 分册 | 页面 | 章节 |
|---|---|---|
| 视觉多模态讲义（上） | [`lecture-1.html`](lecture-1.html) | 2 章 · 277 张配图 |
| 视觉多模态讲义（下） | [`lecture-2.html`](lecture-2.html) | 3 章 · 303 张配图 |
| 视觉多模态面试题 | [`interview.html`](interview.html) | 7 章 · 74 张配图 |
| 视觉多模态项目 | [`projects.html`](projects.html) | 5 章 · 36 张配图 |

阅读页提供：章节切换与上一章/下一章、本节目录与滚动高亮、正文全文搜索、`$$LaTeX$$` 公式渲染、图片懒加载与点击放大、阅读进度、深色模式。

## 章节

讲义原本是单个大 Markdown（最大 780 KB），已按一级标题拆成独立章节文件，便于在 GitHub 上直接阅读和预览。

### 视觉多模态讲义（上）

1. [视觉基础](chapters/lecture-1/01-视觉基础.md)
2. [Vision-Language Model](chapters/lecture-1/02-Vision-Language-Model.md)

### 视觉多模态讲义（下）

1. [GAN & AE & Flow](chapters/lecture-2/01-GAN--AE--Flow.md)
2. [Diffusion Model](chapters/lecture-2/02-Diffusion-Model.md)
3. [UMM 统一理解生成模型](chapters/lecture-2/03-UMM-统一理解生成模型.md)

### 视觉多模态面试题

1. [模型原理](chapters/interview/01-模型原理.md)
2. [对比分析](chapters/interview/02-对比分析.md)
3. [模型细节、架构与应用](chapters/interview/03-模型细节架构与应用.md)
4. [训练与微调](chapters/interview/04-训练与微调.md)
5. [生成与推理](chapters/interview/05-生成与推理.md)
6. [未来发展](chapters/interview/06-未来发展.md)
7. [面试真题与算法刷题](chapters/interview/07-面试真题与算法刷题.md)

### 视觉多模态项目

1. [基础知识](chapters/projects/01-基础知识.md)
2. [VLM 后训练](chapters/projects/02-VLM-后训练.md)
3. [VLM 应用](chapters/projects/03-VLM-应用.md)
4. [Diffusion](chapters/projects/04-Diffusion.md)
5. [统一理解生成](chapters/projects/05-统一理解生成.md)

每个章节文件首尾各有一行 `[Previous] | [Contents] | [Next] | [Visual website]` 导航。

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
chapters/<分册>/NN-章节名.md   拆分后的章节 Markdown（17 个）
assets/site.css         共享样式
assets/parts.js         四个分册的展示信息（标题、简介、标签）
assets/chapters.js      由 build-docs.py 生成：分册 → 章节文件清单
assets/reader.js        阅读页渲染逻辑（Markdown → HTML → 公式 → 目录/搜索/章节切换）
assets/hub.js           总目录卡片渲染
assets/theme.js         深色模式与讲义配色的深色映射
assets/vendor/          本地化的 marked / DOMPurify / KaTeX（无 CDN 依赖）
images/                 690 张配图，以所属分册名前缀区分
视觉多模态讲义（上）.md  各分册 Markdown 原文（未拆分的原始文件）
视觉多模态讲义（下）.md
视觉多模态面试题.md
视觉多模态项目.md
build-docs.py           从原文重新生成 chapters/ 与 assets/chapters.js
```

网页直接读取仓库中的 Markdown 与图片，**不依赖任何 CDN 或构建步骤**，离线可用。

## 重新生成章节

改动了根目录下的 `.md` 原文后，重新切分一次：

```bash
python build-docs.py
```

它会按一级标题拆分，重写章节里的图片路径为 `../../images/`，生成 `assets/chapters.js`，并输出 `chapters/INDEX.md`。

## GitHub Pages

在 GitHub 仓库的 **Settings → Pages** 中，将 **Build and deployment** 设为 **Deploy from a branch**，分支选 `main`、目录选 `/ (root)` 并保存。发布完成后，GitHub 会提供站点地址。仓库根目录已放置 `.nojekyll`，跳过 Jekyll 处理。

## 许可

讲义与配图版权归原作者所有。第三方库见 `assets/vendor/` 下的 LICENSE 文件：
[marked](assets/vendor/LICENSE-marked.md)（MIT）、[DOMPurify](assets/vendor/LICENSE-dompurify.txt)（MPL-2.0 / Apache-2.0）、[KaTeX](assets/vendor/katex/LICENSE.txt)（MIT）。

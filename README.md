# 视觉多模态知识库

围绕视觉基础与 Vision-Language Model（VLM）的中文讲义、面试题与项目实战，共四个分册、40 个小节。

## 在线阅读

从 [`index.html`](index.html) 进入总目录，再跳到对应分册。每册**独立页面**、**一次只渲染一个小节**，公式也只在滚近视口时才排版——首屏 DOM 只有约两千个节点，几百毫秒即可阅读。

| 分册 | 页面 | 章节 |
|---|---|---|
| 视觉多模态讲义（上） | [`lecture-1.html`](lecture-1.html) | 13 节 · 277 张配图 |
| 视觉多模态讲义（下） | [`lecture-2.html`](lecture-2.html) | 15 节 · 303 张配图 |
| 视觉多模态面试题 | [`interview.html`](interview.html) | 7 节 · 74 张配图 |
| 视觉多模态项目 | [`projects.html`](projects.html) | 5 节 · 36 张配图 |

阅读页提供：章节切换与上一章/下一章、本节目录与滚动高亮、正文全文搜索、`$$LaTeX$$` 公式渲染、图片懒加载与点击放大、阅读进度、深色模式。

## 章节

原本四个 300–780 KB 的大 Markdown，已按二级标题切成 40 个小节文件（单个最大约 117 KB），在 GitHub 上点开即读。完整清单见 [`chapters/INDEX.md`](chapters/INDEX.md)。

- [视觉多模态讲义（上）· 13 节](chapters/INDEX.md#视觉多模态讲义上)
- [视觉多模态讲义（下）· 15 节](chapters/INDEX.md#视觉多模态讲义下)
- [视觉多模态面试题 · 7 节](chapters/INDEX.md#视觉多模态面试题)
- [视觉多模态项目 · 5 节](chapters/INDEX.md#视觉多模态项目)

每个小节文件首尾各有一行 `[Previous] | [Contents] | [Next] | [Visual website]` 导航。

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
chapters/<分册>/NN-小节名.md   拆分后的小节 Markdown（40 个）
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

它会按一级/二级标题拆分（超过 110 KB 的小节沿二级标题再切，碎块自动合并），重写章节里的图片路径为 `../../images/`，生成 `assets/chapters.js`，并输出 `chapters/INDEX.md`。

## GitHub Pages

在 GitHub 仓库的 **Settings → Pages** 中，将 **Build and deployment** 设为 **Deploy from a branch**，分支选 `main`、目录选 `/ (root)` 并保存。发布完成后，GitHub 会提供站点地址。仓库根目录已放置 `.nojekyll`，跳过 Jekyll 处理。

## 许可

讲义与配图版权归原作者所有。第三方库见 `assets/vendor/` 下的 LICENSE 文件：
[marked](assets/vendor/LICENSE-marked.md)（MIT）、[DOMPurify](assets/vendor/LICENSE-dompurify.txt)（MPL-2.0 / Apache-2.0）、[KaTeX](assets/vendor/katex/LICENSE.txt)（MIT）。

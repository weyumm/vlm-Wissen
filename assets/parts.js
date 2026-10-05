/* Single source of truth for the four parts of the knowledge base.
   Shared by the hub page (cards) and the reading pages (nav + hero). */
window.VLM_PARTS = [
  {
    key: 'lecture-1',
    page: 'lecture-1.html',
    md: '视觉多模态讲义（上）.md',
    label: '视觉多模态讲义（上）',
    short: '讲义（上）',
    eyebrow: '讲义 · 第一部分',
    desc: '从视觉表征与经典骨干网络出发，梳理对比学习、图像分割，并走进 Vision-Language Model 的发展脉络。',
    pills: ['视觉基础', '骨干网络', '对比学习', '图像分割', 'VLM'],
    meta: ['2 章', '13 节', '277 张配图']
  },
  {
    key: 'lecture-2',
    page: 'lecture-2.html',
    md: '视觉多模态讲义（下）.md',
    label: '视觉多模态讲义（下）',
    short: '讲义（下）',
    eyebrow: '讲义 · 第二部分',
    desc: '生成式视觉建模主线：GAN、自编码器与流模型的数学基础，扩散模型的推导与训练目标，以及统一理解与生成模型（UMM）。',
    pills: ['GAN', 'Autoencoder', 'Flow', 'Diffusion', 'UMM'],
    meta: ['3 章', '15 节', '303 张配图']
  },
  {
    key: 'interview',
    page: 'interview.html',
    md: '视觉多模态面试题.md',
    label: '视觉多模态面试题',
    short: '面试题',
    eyebrow: '面试 · 专项',
    desc: '按模型原理、对比分析、架构细节、训练微调、生成推理与未来方向组织的面试题集，末章附面试真题与算法刷题。',
    pills: ['模型原理', '架构对比', '训练微调', '生成推理', '真题'],
    meta: ['7 章', '22 节', '74 张配图']
  },
  {
    key: 'projects',
    page: 'projects.html',
    md: '视觉多模态项目.md',
    label: '视觉多模态项目',
    short: '项目实战',
    eyebrow: '项目 · 实战',
    desc: '从 LLaMA-Factory 环境搭建与多模态数据格式出发，覆盖 VLM 后训练、应用落地、Diffusion 与统一理解生成实战。',
    pills: ['LLaMA-Factory', '后训练', '应用落地', 'Diffusion', '统一生成'],
    meta: ['5 章', '11 节', '36 张配图']
  }
];

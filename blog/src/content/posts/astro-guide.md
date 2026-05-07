---
title: "Astro 入门指南"
description: "快速上手 Astro 框架，从零搭建一个静态网站。"
pubDate: 2026-05-06
tags: ["Astro", "前端", "教程"]
---

## 什么是 Astro？

Astro 是一个现代化的静态站点生成器，核心特点是 **岛屿架构** —— 页面默认输出纯 HTML，只在需要交互的地方加载 JavaScript。

## 快速开始

```bash
# 创建项目
npm create astro@latest

# 启动开发服务器
npm run dev
```

## 项目结构

```
src/
├── components/    # 组件
├── layouts/       # 布局
├── pages/         # 页面路由
└── content/       # 内容集合
```

### 页面路由

Astro 使用文件系统路由。`src/pages/index.astro` 对应 `/`，`src/pages/blog/[slug].astro` 对应 `/blog/:slug`。

### 内容集合

在 `src/content/` 下的 Markdown 文件可以通过 `getCollection()` 获取：

```astro
---
import { getCollection } from "astro:content";
const posts = await getCollection("posts");
---
```

## 总结

Astro 的学习曲线非常平缓，如果你熟悉 HTML 和 Markdown，几乎可以立刻上手。

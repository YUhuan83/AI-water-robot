#!/usr/bin/env bash
# usage: bash new-post.sh "文章标题" "文章描述" "标签1,标签2"

set -e

TITLE="${1:?请提供文章标题}"
DESC="${2:-}"
TAGS="${3:-}"

SLUG=$(echo "$TITLE" | sed 's/[^a-zA-Z0-9一-龥]/-/g' | sed 's/--*/-/g' | sed 's/^-//;s/-$//' | tr '[:upper:]' '[:lower:]')
DATE=$(date +%Y-%m-%d)

cat > "src/content/posts/${SLUG}.md" << EOF
---
title: "${TITLE}"
description: "${DESC}"
pubDate: ${DATE}
tags: [${TAGS}]
---

## 开始写作

在这里写你的文章内容。

### 小标题

支持所有 Markdown 语法：

- **加粗** 和 *斜体*
- 代码 \`console.log("hello")\`
- 代码块：

\`\`\`python
print("Hello, World!")
\`\`\`

> 引用块也很方便
EOF

echo "✓ 文章已创建: src/content/posts/${SLUG}.md"

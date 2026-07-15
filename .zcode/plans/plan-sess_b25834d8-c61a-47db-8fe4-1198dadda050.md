# 文件上传编译文档 — 实现计划

## 目标
在文档库的"编译文档"功能中，支持：
1. **单/多文件上传** — 选择 .md/.txt 等文件，读取内容填入编译区
2. **文件夹遍历上传** — 选择文件夹，递归遍历其中所有 .md/.txt 文件，批量编译

## 设计思路

### 前端（`static/index.html`）

**改造编译区 HTML** — 在现有 textarea 上方增加上传控件：
- `<input type="file" multiple accept=".md,.txt,.markdown">` — 多文件选择
- `<input type="file" webkitdirectory>` — 文件夹选择（浏览器原生支持 `webkitdirectory` 属性）
- 上传后文件列表展示（文件名 + 大小 + 可删除），内容合并填入 textarea 或直接提交

**新增 JS 函数：**
- `handleFileSelect(event)` — 处理文件选择，用 `FileReader` 读取文本内容
- `handleFolderSelect(event)` — 处理文件夹选择，`webkitRelativePath` 递归遍历
- `renderFileList()` — 渲染已选文件列表（可删除单个文件）
- `compileBatch()` — 批量编译：遍历文件列表逐个调用 `/api/compile`，展示进度

### 后端（`app.py`）

**新增批量编译端点：**
- `POST /api/compile-batch` — 接收 multipart 文件上传（`List[UploadFile]`），逐个读取内容→调用 `llm.compile_document()`→`wr.create_document()`，返回每个文件的编译结果
- 现有 `POST /api/compile` 保持不变（JSON body 手动粘贴场景仍可用）

**新增导入：** `from fastapi import UploadFile, File`

### 文件读取
- 前端用 `FileReader.readAsText()` 读取文件内容，支持 .md/.txt
- 后端用 `await file.read()` 读取上传文件内容
- 文件名作为 `title_hint` 自动传入

### 批量编译流程
1. 用户选择文件/文件夹
2. 前端读取所有文件内容，展示文件列表
3. 用户点击"批量编译"
4. 前端逐个文件调用 `POST /api/compile`（复用现有端点），展示进度条
5. 全部完成后刷新文档列表

## 修改文件
1. `static/index.html` — HTML + CSS + JS
2. `app.py` — 新增 `POST /api/compile-batch` 端点 + `UploadFile` 导入

## 不修改
- `llm.py` — `compile_document()` 已支持 raw_content 参数，无需改动
- `wiki_runtime.py` — `create_document()` 已支持，无需改动
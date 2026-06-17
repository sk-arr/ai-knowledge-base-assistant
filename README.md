# AI企业知识库问答助手

一个面向企业内部资料查询、制度问答和文档检索场景的轻量化 AI 应用 MVP。

项目目标是将“上传资料 → 文本切分 → 相关片段检索 → 回答生成 → 来源展示”的知识库问答流程工具化，展示 AI 应用开发中的 RAG 思路和可落地工具搭建能力。

当前版本不接入真实大模型 API，使用本地检索和 Mock 回答，优先保证项目稳定、可运行、可演示。

## 功能列表

- 支持上传 TXT / MD 文档
- 自动解析并切分文档片段
- 输入问题后检索 Top-3 相关片段
- 生成包含“简要结论、依据摘要、建议下一步”的 Mock 回答
- 展示引用来源片段，便于人工复核
- 支持最近 10 条问答历史
- 支持导出当前问答结果为 Markdown 文件

## 技术栈

- Python
- Streamlit
- 文本切分
- 关键词检索
- RAG 思路
- JSON 本地存储
- Markdown 导出

## 项目结构

```text
ai-knowledge-base-assistant/
├─ app.py
├─ requirements.txt
├─ README.md
├─ services/
│  ├─ document_service.py
│  ├─ retrieval_service.py
│  └─ answer_service.py
├─ utils/
│  └─ storage.py
├─ data/
│  └─ history.json
└─ sample_docs/
   └─ company_ai_policy.md
```

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

启动项目：

```bash
streamlit run app.py
```

## 如何测试

1. 打开 `sample_docs/company_ai_policy.md`
2. 在页面上传该文档
3. 输入问题，例如：`公司使用 AI 工具时有哪些注意事项？`
4. 点击“检索资料并生成回答”
5. 查看回答、依据摘要和引用片段
6. 点击 Markdown 导出按钮

## 项目亮点

- 覆盖 RAG 应用的基础流程：文档输入、切分、检索、回答和引用
- 不依赖真实 API，演示稳定，便于在线部署
- 使用引用片段降低回答无依据的问题
- UI 采用企业知识库控制台风格，与内容生产类工具区分开
- 代码结构清晰，便于后续接入真实大模型 API

## 后续规划

- 接入 Claude / OpenAI / DeepSeek 等大模型 API
- 使用 Embedding 和向量数据库增强检索效果
- 支持 PDF / DOCX 文档解析
- 支持多文档知识库
- 增加 Prompt 模板管理
- 增加用户权限和知识库分组
- 增加回答质量评估和日志记录

## 与 AI 应用开发岗位的匹配点

- 能将企业内部资料检索需求转化为可运行工具
- 理解 RAG 应用的基础链路
- 能完成轻量化 AI 工具搭建和在线演示
- 能沉淀 README、运行说明和项目展示材料
- 具备后续接入真实大模型 API 和向量检索的扩展空间

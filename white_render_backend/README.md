# 数据管线独立后端：白模 → H3 渲染

此交付包含我们数据管线的完整后端，独立于原数据引擎运行。

保留：受控自然语言任务 → 物理白模；上传 2–15 秒白模 → H3；仅生成白模；预览下载；源语义/GT 导出；项目鉴权、报价预算、异步任务、取消/恢复和幂等；以及双块叠放加干扰物实验生成器。

不包含 LeRobot 转换、数据清洗、外部数据集读取/适配、训练集划分、数据集构建发布和评测模块。依赖中没有 pyarrow、SQLAlchemy、Hugging Face 或 LeRobot。

推荐阅读：docs/01_运行位置与安装使用.md → docs/02_前端接入准则.md → docs/03_能力与验证范围.md。OpenAPI 在 docs/openapi.json；Python SDK 在 src/white_render_backend/sdk.py；网页客户端在 web/white-render-client.js。

先解压并进入本目录再安装。代码包包含用户要求的新密钥 private/minimaxapi.docx，仅供后端私下部署；别把它发到浏览器或公开仓库。Docker 平台令牌在 .env；Windows 启动令牌在 work/local-auth.private.json。

生成白模在 worker 所在电脑/服务器运行；H3 在云端运行。前端同学本机启动本包，就使用她本机 CPU 生成白模；连接你的后端则在你的机器生成。本地 GPU 不是必需。

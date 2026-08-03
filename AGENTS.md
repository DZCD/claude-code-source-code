# AGENTS.md

## SDK 变更必须同步文档

当 `packages/agent-sdk` 的语法、公开 API 形状、默认运行时语义、事件名称或模型侧工具契约发生变化时，必须在同一次变更中同步更新文档，不得留作后续补做：

- `packages/agent-sdk/README.md` — 包级文档与示例
- `packages/docs/src/content/docs/` — 文档站对应页面，中英文目录（含 `zh/`）都要更新
- `reference/public-api.mdx` — 公开 API 参考（新增/修改导出时更新 Functions 与 Types 表）
- 涉及行为变化的示例代码也要一并核对

文档是 SDK 接口的一部分。改完代码后运行 `bun run build`（packages/docs）确认文档站构建通过。


## 分支开发工作流

改代码前先建分支，**不在主干上直接改**。一个分支对应一件事：一个新特性，或一个修复。做完并验证通过，再合并回主干。

- 收到「加个功能」「修个 bug」类的请求，**动手写代码之前**先 `git checkout -b <分支名>`，不要先改了再想起来建分支。
- 分支名反映这次改的是什么（如 `feat/case-classification`、`fix/mask-region`）。
- **分支内每完成一个相对完整的版本就提交一次**，不要攒到最后一次性提交。一层能独立跑通、测试齐了，就是一个可提交的版本（例如「OCR 端口 + 适配器 + 缓存 + 测试」是一个版本，之后的「智能体」是下一个）。阶段性提交是默认动作，不用每次问。
- **合并回主干、push 到远端仍然要先问。**
- 提交前跑校验：imdt-claw 是 `pnpm typecheck` + `pnpm test`，涉及外部服务时加 `pnpm smoke`。GitLab CI 在 push 时跑同一组校验（`imdt-claw/.gitlab-ci.yml`，与 imdt-backend 同一套约定：zhdev 基础镜像 + TCR 镜像仓库）；main 分支自动构建两个镜像推 TCR——服务 `zhdev/imdt-claw` 和文档站 `zhdev/imdt-claw-docs`（Mintlify 站点 `docs/`：CI 的 `docs-export` job 用 `mint export` 生成静态站点走 artifact，npm cache 交 GitLab cache 只慢第一次——CI 出口逐 tarball 15s+，不能装进 docker build；`docker/Dockerfile.docs` 只是 COPY site 的运行时镜像，stack 里 `imdt-claw-docs` 服务暴露 3001，网关按独立子域名转发：测试 `imdt-claw-docs-test.zhonghuimedical.com` / 生产 `imdt-claw-docs.zhonghuimedical.com`，export 资源是绝对路径、挂子路径会 404），deploy 手动触发——ansible + docker stack 滚动更新（`ansible.yaml` + `docker/docker-compose.yml.j2`），应用环境变量用 `C_` 前缀的 CI 变量注入。
- 一次性的探针脚本（probe_*）不进仓库，验证价值该沉淀成 smoke 测试。
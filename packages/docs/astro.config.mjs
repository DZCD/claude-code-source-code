import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

export default defineConfig({
  site: "https://docs.claude-code-sdk.com",
  integrations: [
    starlight({
      title: "Claude Agent SDK",
      description:
        "Independent TypeScript SDK for building Claude-style agents without the Claude Code runtime.",
      defaultLocale: "root",
      locales: {
        root: {
          label: "English",
          lang: "en",
        },
        zh: {
          label: "简体中文",
          lang: "zh-CN",
        },
      },
      customCss: ["./src/styles/custom.css"],
      sidebar: [
        {
          label: "Start",
          translations: { "zh-CN": "开始" },
          items: [
            { label: "Overview", translations: { "zh-CN": "概览" }, slug: "" },
            { label: "Quickstart", translations: { "zh-CN": "快速开始" }, slug: "quickstart" },
            { label: "Configuration", translations: { "zh-CN": "配置" }, slug: "configuration" },
          ],
        },
        {
          label: "Core Concepts",
          translations: { "zh-CN": "核心概念" },
          items: [
            { label: "Agent Loop", translations: { "zh-CN": "Agent 循环" }, slug: "concepts/agent-loop" },
            { label: "Streaming Events", translations: { "zh-CN": "流式事件" }, slug: "concepts/streaming-events" },
            { label: "Tools", translations: { "zh-CN": "工具" }, slug: "concepts/tools" },
            { label: "MCP", slug: "concepts/mcp" },
            { label: "Skills", translations: { "zh-CN": "技能" }, slug: "concepts/skills" },
            { label: "Team", translations: { "zh-CN": "Team 团队" }, slug: "concepts/multi-agent" },
            { label: "Built-in Tools", translations: { "zh-CN": "内置工具" }, slug: "concepts/built-in-tools" },
            { label: "Permissions", translations: { "zh-CN": "权限" }, slug: "concepts/permissions" },
          ],
        },
        {
          label: "Reference",
          translations: { "zh-CN": "参考" },
          items: [{ label: "Public API", translations: { "zh-CN": "公开 API" }, slug: "reference/public-api" }],
        },
      ],
    }),
  ],
});

/**
 * Site navigation, shared by the Starlight config and the llms.txt endpoints so
 * the agent-facing index always lists the same pages in the same order as the
 * sidebar a human sees.
 */
export type SidebarItem = {
  label: string;
  translations?: Record<string, string>;
  slug: string;
};

export type SidebarGroup = {
  label: string;
  translations?: Record<string, string>;
  items: SidebarItem[];
};

export const sidebar: SidebarGroup[] = [
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
      { label: "Context Tracing", translations: { "zh-CN": "上下文追踪" }, slug: "concepts/context-tracing" },
      { label: "Tools", translations: { "zh-CN": "工具" }, slug: "concepts/tools" },
      { label: "MCP", slug: "concepts/mcp" },
      { label: "Skills", translations: { "zh-CN": "技能" }, slug: "concepts/skills" },
      { label: "Supervisor Delegation", translations: { "zh-CN": "Supervisor 子智能体委派" }, slug: "concepts/supervisor-delegation" },
      { label: "Mailbox Team", translations: { "zh-CN": "Mailbox Team 持久团队" }, slug: "concepts/mailbox-team" },
      { label: "Built-in Tools", translations: { "zh-CN": "内置工具" }, slug: "concepts/built-in-tools" },
      { label: "Permissions", translations: { "zh-CN": "权限" }, slug: "concepts/permissions" },
    ],
  },
  {
    label: "Reference",
    translations: { "zh-CN": "参考" },
    items: [{ label: "Public API", translations: { "zh-CN": "公开 API" }, slug: "reference/public-api" }],
  },
];

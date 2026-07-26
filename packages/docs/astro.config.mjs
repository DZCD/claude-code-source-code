import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import { sidebar } from "./src/sidebar.ts";

export default defineConfig({
  site: "https://docs.claude-code-sdk.com",
  integrations: [
    starlight({
      title: "AgentLattice",
      description:
        "TypeScript framework for building coordinated agent systems with tools, skills, tracing, and teams.",
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
      sidebar,
    }),
  ],
});

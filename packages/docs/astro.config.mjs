import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import { sidebar } from "./src/sidebar.ts";

const GOOGLE_ANALYTICS_ID = "G-WB0FG6FHTV";

/**
 * Only built into production output, so `astro dev` and `astro preview` do not
 * report local page views as real traffic. The measurement ID is public by
 * design — it ships in the page source of every analytics-enabled site.
 */
const analytics =
  process.env.NODE_ENV === "production"
    ? [
        {
          tag: "script",
          attrs: {
            async: true,
            src: `https://www.googletagmanager.com/gtag/js?id=${GOOGLE_ANALYTICS_ID}`,
          },
        },
        {
          tag: "script",
          content: [
            "window.dataLayer = window.dataLayer || [];",
            "function gtag(){dataLayer.push(arguments);}",
            "gtag('js', new Date());",
            `gtag('config', '${GOOGLE_ANALYTICS_ID}');`,
          ].join("\n"),
        },
      ]
    : [];

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
      head: analytics,
      sidebar,
    }),
  ],
});

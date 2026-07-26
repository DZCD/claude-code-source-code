import type { APIRoute } from "astro";
import { getCollection } from "astro:content";
import { sidebar } from "../sidebar";
import { isEnglish, isSplash, markdownUrl, type DocEntry } from "../lib/page-markdown";

const INTRO = `# AgentLattice

> TypeScript framework for building coordinated agent systems with tools, skills, tracing, and teams.

AgentLattice runs an agent loop against any Anthropic-compatible model, executes
tools you define, and coordinates several agents through supervisor delegation or
durable mailbox teams. Install it with \`npm install agent-lattice zod\`.

Every link below serves clean Markdown. Fetch the page that matches the task at
hand, or read llms-full.txt for the whole documentation in one request.`;

export const GET: APIRoute = async ({ site }) => {
  const entries = await getCollection("docs");
  const bySlug = new Map<string, DocEntry>(entries.map(entry => [entry.id, entry]));

  const sections = sidebar
    .map(group => {
      const lines = group.items
        .map(item => {
          const entry = bySlug.get(item.slug);
          // A sidebar entry with no page, or a landing page, has nothing an
          // agent can read.
          if (!entry || !isEnglish(entry) || isSplash(entry)) return undefined;
          const hint = entry.data.description ? `: ${entry.data.description}` : "";
          return `- [${item.label}](${markdownUrl(site, item.slug)})${hint}`;
        })
        .filter((line): line is string => line !== undefined);
      return lines.length > 0 ? `## ${group.label}\n\n${lines.join("\n")}` : undefined;
    })
    .filter((section): section is string => section !== undefined);

  const optional = `## Optional

- [Full documentation in one file](${new URL("/llms-full.txt", site ?? "https://docs.claude-code-sdk.com").href}): every page above concatenated, for agents that prefer one large read over several small ones.`;

  return new Response([INTRO, ...sections, optional].join("\n\n") + "\n", {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
};

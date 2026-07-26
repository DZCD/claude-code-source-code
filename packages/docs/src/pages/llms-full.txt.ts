import type { APIRoute } from "astro";
import { getCollection } from "astro:content";
import { sidebar } from "../sidebar";
import { isEnglish, isSplash, pageMarkdown, type DocEntry } from "../lib/page-markdown";

const INTRO = `# AgentLattice — full documentation

> TypeScript framework for building coordinated agent systems with tools, skills, tracing, and teams.

Every English documentation page, in sidebar order. Install with
\`npm install agent-lattice zod\`. Individual pages are also available as
Markdown at their own URLs; see llms.txt for the index.`;

export const GET: APIRoute = async () => {
  const entries = await getCollection("docs");
  const bySlug = new Map<string, DocEntry>(entries.map(entry => [entry.id, entry]));

  const knownSlugs = new Set(entries.filter(entry => !isSplash(entry)).map(entry => entry.id));
  const pages = sidebar
    .flatMap(group => group.items)
    .map(item => bySlug.get(item.slug))
    .filter((entry): entry is DocEntry => entry !== undefined && isEnglish(entry) && !isSplash(entry))
    .map(entry => pageMarkdown(entry, knownSlugs));

  return new Response([INTRO, ...pages].join("\n\n---\n\n"), {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
};

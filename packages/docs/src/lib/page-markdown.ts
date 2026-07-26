import type { CollectionEntry } from "astro:content";

export type DocEntry = CollectionEntry<"docs">;

/**
 * Landing pages use Starlight components and carry no prose worth reading, so
 * they are left out of every agent-facing output. Skipping them is also why no
 * JSX stripping is needed: they are the only pages that use components.
 */
export function isSplash(entry: DocEntry): boolean {
  return entry.data.template === "splash";
}

export function isEnglish(entry: DocEntry): boolean {
  return !entry.id.startsWith("zh/");
}

/** Page URL as an agent should fetch it: absolute, and Markdown rather than HTML. */
export function markdownUrl(site: URL | undefined, slug: string): string {
  return new URL(`/${slug}.md`, site ?? "https://docs.claude-code-sdk.com").href;
}

/**
 * The page as plain Markdown: the title and description promoted out of
 * frontmatter, then the body verbatim. Astro's own metadata is dropped since it
 * means nothing outside the site build.
 */
export function pageMarkdown(entry: DocEntry): string {
  const body = (entry.body ?? "").trim();
  if (body.includes("@astrojs/starlight/components")) {
    // Not fatal, but the mirror will contain raw JSX until the page is reworked
    // or marked as a splash page.
    console.warn(
      `[llms.txt] ${entry.id} imports Starlight components; its Markdown mirror will contain JSX.`,
    );
  }
  const description = entry.data.description ? `> ${entry.data.description}\n\n` : "";
  return `# ${entry.data.title}\n\n${description}${body}\n`;
}

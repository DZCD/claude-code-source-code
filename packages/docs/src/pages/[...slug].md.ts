import type { APIRoute, GetStaticPaths } from "astro";
import { getCollection } from "astro:content";
import { isSplash, pageMarkdown, type DocEntry } from "../lib/page-markdown";

/**
 * A Markdown mirror of every documentation page, at the HTML page's URL with a
 * `.md` suffix. Generated for both languages: `llms.txt` only indexes the
 * English pages, but a Chinese page is still worth fetching directly.
 */
export const getStaticPaths: GetStaticPaths = async () => {
  const entries = await getCollection("docs");
  const pages = entries.filter(entry => !isSplash(entry));
  const knownSlugs = new Set(pages.map(entry => entry.id));
  return pages.map(entry => ({ params: { slug: entry.id }, props: { entry, knownSlugs } }));
};

export const GET: APIRoute = ({ props }) =>
  new Response(pageMarkdown(props.entry as DocEntry, props.knownSlugs as Set<string>), {
    headers: { "Content-Type": "text/markdown; charset=utf-8" },
  });

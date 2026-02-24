/**
 * Scans text for concept terms (from concept_linked messages) and returns
 * annotated tokens. Used in parallel mode to highlight medical concepts
 * in the answer text and link them to segments on the image.
 */

import type { ConceptLink } from "../types/pipeline";

export interface TextToken {
  type: "text";
  text: string;
}

export interface ConceptToken {
  type: "concept";
  text: string;
  segmentId: string;
  color: string;
}

export type AnnotatedToken = TextToken | ConceptToken;

/**
 * Parse text and replace concept terms/aliases with ConceptToken spans.
 * - Collects all terms + aliases from conceptLinks
 * - Sorts by length descending (match "pleural effusion" before "pleural")
 * - Builds combined regex with word boundaries, case-insensitive
 * - Non-overlapping (first match wins due to length-sorted precedence)
 */
export function parseConceptAnnotations(
  text: string,
  conceptLinks: ConceptLink[],
): AnnotatedToken[] {
  if (!text || conceptLinks.length === 0) {
    return text ? [{ type: "text", text }] : [];
  }

  // Build term→{segmentId, color} lookup (longest first)
  const entries: { term: string; segmentId: string; color: string }[] = [];
  for (const cl of conceptLinks) {
    entries.push({ term: cl.concept, segmentId: cl.segmentId, color: cl.color });
    for (const alias of cl.aliases) {
      if (alias) {
        entries.push({ term: alias, segmentId: cl.segmentId, color: cl.color });
      }
    }
  }

  // Deduplicate and sort by length descending
  const seen = new Set<string>();
  const uniqueEntries: typeof entries = [];
  for (const e of entries) {
    const key = e.term.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      uniqueEntries.push(e);
    }
  }
  uniqueEntries.sort((a, b) => b.term.length - a.term.length);

  if (uniqueEntries.length === 0) {
    return [{ type: "text", text }];
  }

  // Build combined regex: escape special chars, word boundaries
  const patterns = uniqueEntries.map(
    (e) => `\\b${escapeRegex(e.term)}\\b`
  );
  const combined = new RegExp(`(${patterns.join("|")})`, "gi");

  // Build lowercase term→entry map for quick lookup
  const termMap = new Map<string, { segmentId: string; color: string }>();
  for (const e of uniqueEntries) {
    termMap.set(e.term.toLowerCase(), { segmentId: e.segmentId, color: e.color });
  }

  const tokens: AnnotatedToken[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = combined.exec(text)) !== null) {
    const matchedText = match[0];
    const entry = termMap.get(matchedText.toLowerCase());
    if (!entry) continue;

    if (match.index > lastIndex) {
      tokens.push({ type: "text", text: text.slice(lastIndex, match.index) });
    }

    tokens.push({
      type: "concept",
      text: matchedText,
      segmentId: entry.segmentId,
      color: entry.color,
    });
    lastIndex = match.index + matchedText.length;
  }

  if (lastIndex < text.length) {
    tokens.push({ type: "text", text: text.slice(lastIndex) });
  }

  return tokens.length > 0 ? tokens : [{ type: "text", text }];
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

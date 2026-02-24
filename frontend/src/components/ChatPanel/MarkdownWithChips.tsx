/**
 * Renders markdown content with inline segment chip support.
 * Markdown is parsed first (preserving bold, lists, etc.), then
 * chip patterns ([SEG:A], Segment A, etc.) in text nodes are
 * replaced with interactive SegmentChip components.
 *
 * In parallel mode, concept terms from concept_linked messages
 * are highlighted with ConceptHighlight components instead.
 */

import { Children, type ReactNode, useMemo } from "react";
import Markdown from "react-markdown";
import { useJobStore } from "../../stores/jobStore";
import { parseConceptAnnotations } from "../../utils/conceptAnnotator";
import { parseSegmentChips } from "../../utils/segmentChipParser";
import { ConceptHighlight } from "./ConceptHighlight";
import { SegmentChip } from "./SegmentChip";
import type { ConceptLink } from "../../types/pipeline";

interface Props {
  content: string;
}

export function MarkdownWithChips({ content }: Props) {
  const segments = useJobStore((s) => s.segments);
  const conceptLinks = useJobStore((s) => s.conceptLinks);
  const segmentIds = useMemo(() => Array.from(segments.keys()), [segments]);
  const components = useMemo(
    () => chipComponents(segmentIds, conceptLinks),
    [segmentIds, conceptLinks],
  );

  return <Markdown components={components}>{content}</Markdown>;
}

/** Replace string children containing chip/concept patterns with interactive components */
function injectAnnotations(
  children: ReactNode,
  segmentIds: string[],
  conceptLinks: ConceptLink[],
): ReactNode {
  return Children.map(children, (child) => {
    if (typeof child !== "string") return child;

    // Parallel mode: use concept annotations
    if (conceptLinks.length > 0) {
      const tokens = parseConceptAnnotations(child, conceptLinks);
      if (tokens.length === 1 && tokens[0].type === "text") return child;
      return (
        <>
          {tokens.map((tok, i) =>
            tok.type === "concept" ? (
              <ConceptHighlight
                key={i}
                text={tok.text}
                segmentId={tok.segmentId}
                color={tok.color}
              />
            ) : (
              <span key={i}>{tok.text}</span>
            ),
          )}
        </>
      );
    }

    // Sequential mode: use segment chips
    if (segmentIds.length === 0) return child;
    const tokens = parseSegmentChips(child, segmentIds);
    if (tokens.length === 1 && tokens[0].type === "text") return child;
    return (
      <>
        {tokens.map((tok, i) =>
          tok.type === "chip" ? (
            <SegmentChip key={i} segmentId={tok.segmentId} />
          ) : (
            <span key={i}>{tok.text}</span>
          ),
        )}
      </>
    );
  });
}

/** Build react-markdown custom components that process text for chips/concepts.
 *  Code blocks are left untouched so patterns in code stay as text. */
function chipComponents(segmentIds: string[], conceptLinks: ConceptLink[]) {
  const inject = (c: ReactNode) => injectAnnotations(c, segmentIds, conceptLinks);

  return {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    p: ({ children }: any) => <p>{inject(children)}</p>,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    li: ({ children }: any) => <li>{inject(children)}</li>,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    strong: ({ children }: any) => <strong>{inject(children)}</strong>,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    em: ({ children }: any) => <em>{inject(children)}</em>,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    h1: ({ children }: any) => <h1>{inject(children)}</h1>,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    h2: ({ children }: any) => <h2>{inject(children)}</h2>,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    h3: ({ children }: any) => <h3>{inject(children)}</h3>,
  };
}

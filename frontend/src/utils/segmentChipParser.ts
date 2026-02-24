/**
 * Parse text containing segment references into an array of text and chip tokens.
 * Mirrors the regex patterns from utils/segment_chip_processor.py.
 */

export interface TextToken {
  type: "text";
  text: string;
}

export interface ChipToken {
  type: "chip";
  segmentId: string;
}

export type Token = TextToken | ChipToken;

// Match patterns: [[SEG:A]], [SEG:A], Segment A, SEG:A
const CHIP_REGEX =
  /\[\[SEG:([A-Z])\]\]|\[SEG:([A-Z])\]|(?:Segment\s+)([A-Z])(?=[\s.,;:!?\)]|$)|<span[^>]*data-seg-id="([A-Z])"[^>]*>\[SEG:\4\]<\/span>/gi;

export function parseSegmentChips(
  text: string,
  availableSegments: string[]
): Token[] {
  if (!text || availableSegments.length === 0) {
    return text ? [{ type: "text", text }] : [];
  }

  const tokens: Token[] = [];
  let lastIndex = 0;

  // Reset regex state
  CHIP_REGEX.lastIndex = 0;

  let match: RegExpExecArray | null;
  while ((match = CHIP_REGEX.exec(text)) !== null) {
    // Get the segment ID from whichever capture group matched
    const segId = (
      match[1] || match[2] || match[3] || match[4] || ""
    ).toUpperCase();

    if (!availableSegments.includes(segId)) continue;

    // Add preceding text
    if (match.index > lastIndex) {
      tokens.push({ type: "text", text: text.slice(lastIndex, match.index) });
    }

    tokens.push({ type: "chip", segmentId: segId });
    lastIndex = match.index + match[0].length;
  }

  // Add trailing text
  if (lastIndex < text.length) {
    tokens.push({ type: "text", text: text.slice(lastIndex) });
  }

  return tokens.length > 0 ? tokens : [{ type: "text", text }];
}

//! Recursive token-based document chunking.
//!
//! Direct port of `src/ragrep/ingest/chunk.py`. The token estimator is a
//! heuristic — `max(word_count, len_bytes/4)` — chosen over a real
//! tokenizer (tiktoken etc.) because it handles tab-delimited content like
//! CSV exports and code where real tokenizers underestimate.

use crate::models::{Chunk, Document, MetaValue};

/// Sources that are typically small and never get split.
const NO_CHUNK_SOURCES: &[&str] = &["bookmark", "pin"];

/// Separators in priority order for recursive splitting.
const SEPARATORS: &[&str] = &["\n\n", "\n", "\t", ". ", " "];

/// Hard byte limit per chunk (safety valve; ~1K tokens at ~4 bytes/token).
const MAX_BYTES: usize = 4000;

/// Heuristic token count: max(word_count, byte_len / 4). The byte/4 fallback
/// catches tab-delimited content and dense code where word_count
/// underestimates token cost.
fn approx_tokens(text: &str) -> usize {
    let words = text.split_whitespace().count();
    let bytes_4 = text.len() / 4;
    words.max(bytes_4)
}

/// Recursively split `text` into chunks under `max_tokens`. Walks
/// SEPARATORS in priority order, falling back to a hard word-count split
/// if no separator yields multiple parts.
fn recursive_split(text: &str, max_tokens: usize, overlap_tokens: usize) -> Vec<String> {
    if approx_tokens(text) <= max_tokens {
        return vec![text.to_string()];
    }

    for sep in SEPARATORS {
        let parts: Vec<&str> = text.split(sep).collect();
        if parts.len() <= 1 {
            continue;
        }

        let mut chunks: Vec<String> = Vec::new();
        let mut current = parts[0].to_string();

        for part in &parts[1..] {
            let candidate = format!("{current}{sep}{part}");
            if approx_tokens(&candidate) <= max_tokens {
                current = candidate;
            } else {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    chunks.push(trimmed.to_string());
                }
                if overlap_tokens > 0 {
                    let words: Vec<&str> = current.split_whitespace().collect();
                    let take = overlap_tokens.min(words.len());
                    let tail = words[words.len() - take..].join(" ");
                    if tail.is_empty() {
                        current = (*part).to_string();
                    } else {
                        current = format!("{tail}{sep}{part}").trim().to_string();
                    }
                } else {
                    current = (*part).to_string();
                }
            }
        }

        let trimmed = current.trim();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
        }

        if chunks.len() > 1 {
            return chunks;
        }
    }

    // Final fallback: hard split by words at max_tokens.
    let words: Vec<&str> = text.split_whitespace().collect();
    let stride = max_tokens.saturating_sub(overlap_tokens).max(1);
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let end = (i + max_tokens).min(words.len());
        let piece = words[i..end].join(" ");
        let trimmed = piece.trim();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
        }
        if end == words.len() {
            break;
        }
        i += stride;
    }
    chunks
}

fn make_id(doc_id: &str, idx: usize) -> String {
    format!("{doc_id}:{idx}")
}

fn metadata_with_chunk_idx(
    base: &std::collections::BTreeMap<String, MetaValue>,
    idx: usize,
) -> std::collections::BTreeMap<String, MetaValue> {
    let mut m = base.clone();
    m.insert("chunk_idx".to_string(), MetaValue::Int(idx as i64));
    m
}

pub fn chunk_document(doc: &Document, max_tokens: usize, overlap_tokens: usize) -> Vec<Chunk> {
    if NO_CHUNK_SOURCES.contains(&doc.source.as_str()) || approx_tokens(&doc.content) <= max_tokens
    {
        return vec![Chunk {
            id: make_id(&doc.id, 0),
            doc_id: doc.id.clone(),
            content: doc.content.clone(),
            title: doc.title.clone(),
            source: doc.source.clone(),
            metadata: metadata_with_chunk_idx(&doc.metadata, 0),
        }];
    }

    let parts = recursive_split(&doc.content, max_tokens, overlap_tokens);

    // Safety valve: split by byte length when any single part exceeds MAX_BYTES.
    let mut safe: Vec<String> = Vec::new();
    for part in parts {
        if part.len() <= MAX_BYTES {
            safe.push(part);
        } else {
            // Walk char boundaries to avoid splitting a multibyte sequence.
            let bytes = part.as_bytes();
            let mut start = 0;
            while start < bytes.len() {
                let mut end = (start + MAX_BYTES).min(bytes.len());
                while end < bytes.len() && (bytes[end] & 0b1100_0000) == 0b1000_0000 {
                    end -= 1;
                }
                let segment = part[start..end].trim();
                if !segment.is_empty() {
                    safe.push(segment.to_string());
                }
                start = end;
            }
        }
    }

    safe.into_iter()
        .enumerate()
        .map(|(i, content)| Chunk {
            id: make_id(&doc.id, i),
            doc_id: doc.id.clone(),
            content,
            title: doc.title.clone(),
            source: doc.source.clone(),
            metadata: metadata_with_chunk_idx(&doc.metadata, i),
        })
        .collect()
}

pub fn all(docs: &[Document], max_tokens: usize, overlap_tokens: usize) -> Vec<Chunk> {
    let mut out: Vec<Chunk> = Vec::new();
    for doc in docs {
        out.extend(chunk_document(doc, max_tokens, overlap_tokens));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn doc(id: &str, source: &str, content: &str) -> Document {
        Document {
            id: id.into(),
            source: source.into(),
            content: content.into(),
            title: format!("title for {id}"),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn small_doc_yields_single_chunk() {
        let d = doc("d1", "slack", "short message");
        let chunks = chunk_document(&d, 512, 64);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].id, "d1:0");
        assert_eq!(chunks[0].content, "short message");
        assert!(matches!(
            chunks[0].metadata.get("chunk_idx"),
            Some(MetaValue::Int(0))
        ));
    }

    #[test]
    fn no_chunk_sources_never_split() {
        let big = "x ".repeat(5000); // 10000 chars, far over max_tokens
        let d = doc("d1", "bookmark", &big);
        let chunks = chunk_document(&d, 512, 64);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].source, "bookmark");
    }

    #[test]
    fn long_doc_splits_at_paragraph() {
        let para = "word ".repeat(300); // ~300 tokens
        let content = format!("{para}\n\n{para}\n\n{para}");
        let d = doc("d1", "slack", &content);
        let chunks = chunk_document(&d, 512, 0);
        assert!(chunks.len() >= 2, "got {} chunks", chunks.len());
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.id, format!("d1:{i}"));
            assert!(
                matches!(c.metadata.get("chunk_idx"), Some(MetaValue::Int(n)) if *n == i as i64)
            );
        }
    }

    #[test]
    fn approx_tokens_uses_byte_max_for_dense_content() {
        // 10 words but 1000 bytes (CSV-style): byte/4 = 250 dominates.
        let dense = "a,b,c,d,e,f,g,h,i,j,".repeat(50);
        assert!(approx_tokens(&dense) > 50);
    }

    #[test]
    fn overlap_carries_tail_words() {
        let para = "alpha beta gamma delta epsilon zeta eta theta ".repeat(50);
        let content = format!("{para}\n{para}");
        let d = doc("d1", "slack", &content);
        let chunks = chunk_document(&d, 64, 8);
        assert!(chunks.len() >= 2);
        // The second chunk should start with words drawn from the tail of the first.
        if chunks.len() >= 2 {
            let first_tail: Vec<&str> =
                chunks[0].content.split_whitespace().rev().take(8).collect();
            let second_head: Vec<&str> = chunks[1].content.split_whitespace().take(8).collect();
            // At least one of the trailing words from chunk 0 should appear in
            // the first 16 words of chunk 1 (overlap may straddle separator).
            let any_overlap = first_tail.iter().any(|w| second_head.contains(w));
            assert!(
                any_overlap,
                "no overlap; first tail={first_tail:?} second head={second_head:?}"
            );
        }
    }
}

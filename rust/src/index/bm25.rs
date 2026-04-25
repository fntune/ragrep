//! Okapi BM25 over whitespace+lowercase tokens.
//!
//! Mirrors `rank_bm25.BM25Okapi` (which `src/ragrep/ingest/store.py` uses):
//! same k1=1.5, b=0.75, IDF formula `ln((N - df + 0.5) / (df + 0.5) + 1)`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

const K1: f32 = 1.5;
const B: f32 = 0.75;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25 {
    /// Doc length (token count) per document.
    doc_lens: Vec<u32>,
    /// Average doc length.
    avg_doc_len: f32,
    /// term -> Vec<(doc_idx, term_freq_in_doc)>
    postings: HashMap<String, Vec<(u32, u32)>>,
    /// term -> idf
    idf: HashMap<String, f32>,
    /// Total number of docs.
    n_docs: u32,
}

impl Bm25 {
    pub fn build<I, S>(docs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut doc_lens: Vec<u32> = Vec::new();
        let mut postings: HashMap<String, Vec<(u32, u32)>> = HashMap::new();

        for (i, doc) in docs.into_iter().enumerate() {
            let tokens = tokenize(doc.as_ref());
            doc_lens.push(tokens.len() as u32);

            let mut tf: HashMap<String, u32> = HashMap::new();
            for tok in tokens {
                *tf.entry(tok).or_insert(0) += 1;
            }
            for (term, count) in tf {
                postings.entry(term).or_default().push((i as u32, count));
            }
        }

        let n_docs = doc_lens.len() as u32;
        let avg_doc_len = if n_docs == 0 {
            0.0
        } else {
            doc_lens.iter().map(|&l| l as f32).sum::<f32>() / n_docs as f32
        };

        let idf: HashMap<String, f32> = postings
            .iter()
            .map(|(term, plist)| {
                let df = plist.len() as f32;
                let v = ((n_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
                (term.clone(), v)
            })
            .collect();

        Self {
            doc_lens,
            avg_doc_len,
            postings,
            idf,
            n_docs,
        }
    }

    pub fn n_docs(&self) -> usize {
        self.n_docs as usize
    }

    /// Score every document against the query. Returns a Vec of length `n_docs`
    /// where index i is the score for doc i. Mirrors `BM25Okapi.get_scores`.
    pub fn scores(&self, query: &str) -> Vec<f32> {
        let mut scores = vec![0.0f32; self.n_docs as usize];
        if self.n_docs == 0 {
            return scores;
        }
        for term in tokenize(query) {
            let Some(idf) = self.idf.get(&term) else {
                continue;
            };
            let Some(plist) = self.postings.get(&term) else {
                continue;
            };
            for &(doc_idx, tf) in plist {
                let tf = tf as f32;
                let dl = self.doc_lens[doc_idx as usize] as f32;
                let denom = tf + K1 * (1.0 - B + B * dl / self.avg_doc_len);
                scores[doc_idx as usize] += idf * (tf * (K1 + 1.0)) / denom;
            }
        }
        scores
    }
}

pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_match_python_reference() {
        // 3 docs, the same as a rank_bm25 example
        let docs = ["the cat sat", "the dog ran", "cats and dogs"];
        let bm = Bm25::build(docs);
        let s = bm.scores("cat");
        // doc 0 contains "cat", others don't → only s[0] > 0
        assert!(s[0] > 0.0);
        assert_eq!(s[1], 0.0);
        assert_eq!(s[2], 0.0);
    }

    #[test]
    fn empty_query_zero_scores() {
        let bm = Bm25::build(["a b c", "d e f"]);
        let s = bm.scores("");
        assert_eq!(s, vec![0.0, 0.0]);
    }

    #[test]
    fn unknown_term_zero_scores() {
        let bm = Bm25::build(["a b c", "d e f"]);
        let s = bm.scores("zzz");
        assert_eq!(s, vec![0.0, 0.0]);
    }
}

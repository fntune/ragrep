use std::cmp::Ordering;
use std::fs::File;
use std::io;
use std::path::Path;

use memmap2::Mmap;
use rayon::prelude::*;
use wide::f32x8;

pub struct Flat {
    mmap: Mmap,
    pub dim: usize,
    pub n: usize,
}

impl Flat {
    pub fn open(path: &Path, dim: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let row_bytes = dim * 4;
        if mmap.len() % row_bytes != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("file size {} not a multiple of dim*4 ({})", mmap.len(), row_bytes),
            ));
        }
        let n = mmap.len() / row_bytes;
        Ok(Self { mmap, dim, n })
    }

    fn embeddings(&self) -> &[f32] {
        bytemuck::cast_slice(&self.mmap[..])
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim, "query dim mismatch");
        let embs = self.embeddings();
        let mut scored: Vec<(f32, u32)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let row = &embs[i * self.dim..(i + 1) * self.dim];
                (inner_product(query, row), i as u32)
            })
            .collect();

        let k = top_k.min(scored.len());
        if k == 0 {
            return Vec::new();
        }
        scored.select_nth_unstable_by(k - 1, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal)
        });
        scored.truncate(k);
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        scored.into_iter().map(|(s, i)| (i, s)).collect()
    }
}

fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = f32x8::ZERO;
    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);
    for (av, bv) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        let av: [f32; 8] = av.try_into().unwrap();
        let bv: [f32; 8] = bv.try_into().unwrap();
        acc += f32x8::new(av) * f32x8::new(bv);
    }
    let mut sum = acc.reduce_add();
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder().iter()) {
        sum += x * y;
    }
    sum
}

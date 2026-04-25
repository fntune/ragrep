//! Git commits → Document. (TODO Phase 2 step 2.5h)

use std::path::Path;

use anyhow::Result;

use crate::models::Document;

pub fn normalize(_raw_dir: &Path) -> Result<Vec<Document>> {
    Ok(Vec::new())
}

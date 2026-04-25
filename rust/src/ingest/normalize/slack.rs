//! Slack messages → Document. (TODO Phase 2 step 2.5i)

use std::path::Path;

use anyhow::Result;

use super::{ChannelMap, UserMap};
use crate::models::Document;

pub fn normalize(
    _raw_dir: &Path,
    _users: &UserMap,
    _channels: &ChannelMap,
) -> Result<Vec<Document>> {
    Ok(Vec::new())
}

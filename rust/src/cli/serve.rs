use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::Args;

use crate::config;
use crate::index::store;
use crate::serve;

const STORAGE_SCOPE: &str = "https://www.googleapis.com/auth/devstorage.read_only";
const DEFAULT_GCS_INDEX_DIR: &str = "/tmp/ragrep-index";

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Port to bind.
    #[arg(long, default_value_t = 8321)]
    pub port: u16,

    /// Host to bind.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Path to config.toml.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

pub fn run(args: ServeArgs) -> Result<()> {
    init_tracing();

    // Pre-warm rayon (the search-path's parallel inner-product loop needs it).
    std::thread::spawn(|| {
        use rayon::prelude::*;
        [0u8; 1].par_iter().for_each(|_| ());
    });

    let mut cfg = config::load(args.config.as_deref())?;
    let index_dir = resolve_index_dir(&cfg)?;
    cfg.data.index_dir = index_dir.display().to_string();
    tracing::info!(
        target: "ragrep::serve",
        "loading index from {}",
        cfg.index_dir().display()
    );
    let auth_policy = serve::auth::Policy::from_env()?;
    let state = Arc::new(serve::AppState::load(cfg)?);
    tracing::info!(
        target: "ragrep::serve",
        "ready: {} chunks, embedder={}/{} dim={}, auth={}",
        state.chunks.len(),
        state.embedder.provider(),
        state.embedder.model(),
        state.embedder.dim(),
        auth_policy.label()
    );

    let app = serve::router(state, auth_policy);
    let bind = format!("{}:{}", args.host, args.port);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&bind).await?;
        tracing::info!(target: "ragrep::serve", "listening on http://{bind}");
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })?;

    Ok(())
}

fn resolve_index_dir(cfg: &config::Config) -> Result<PathBuf> {
    if let Ok(raw) = std::env::var("RAGREP_INDEX_DIR") {
        let dir = PathBuf::from(raw);
        if store::index_exists(&dir) {
            tracing::info!(target: "ragrep::serve", "using RAGREP_INDEX_DIR={}", dir.display());
            return Ok(dir);
        }
        if let Some(bucket) = std::env::var("RAGREP_GCS_BUCKET")
            .ok()
            .filter(|value| !value.trim().is_empty())
        {
            download_index(&bucket, &dir)?;
            return Ok(dir);
        }
        bail!(
            "RAGREP_INDEX_DIR={} does not contain {}, {}, and {}; set RAGREP_GCS_BUCKET to download them",
            dir.display(),
            store::FILE_CHUNKS,
            store::FILE_EMBEDDINGS,
            store::FILE_BM25
        );
    }

    let configured = cfg.index_dir();
    if store::index_exists(&configured) {
        return Ok(configured);
    }

    let Some(bucket) = std::env::var("RAGREP_GCS_BUCKET")
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        bail!(
            "no Rust index found in {}; set RAGREP_INDEX_DIR or RAGREP_GCS_BUCKET",
            configured.display()
        );
    };

    let cache_dir = std::env::var("RAGREP_INDEX_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_GCS_INDEX_DIR));
    if !store::index_exists(&cache_dir) {
        download_index(&bucket, &cache_dir)?;
    }
    Ok(cache_dir)
}

fn download_index(bucket_spec: &str, dest: &PathBuf) -> Result<()> {
    let (bucket, prefix) = parse_bucket_spec(bucket_spec)?;
    std::fs::create_dir_all(dest).with_context(|| format!("creating {}", dest.display()))?;
    tracing::info!(
        target: "ragrep::serve",
        "downloading Rust index from gs://{}/{} to {}",
        bucket,
        prefix,
        dest.display()
    );

    let token = google_access_token(STORAGE_SCOPE).context(
        "getting Google Storage credentials; Cloud Run should use its service account, \
         or set GOOGLE_APPLICATION_CREDENTIALS locally",
    )?;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .context("building Google Storage HTTP client")?;

    for file in [store::FILE_CHUNKS, store::FILE_EMBEDDINGS, store::FILE_BM25] {
        let object = object_name(&prefix, file);
        let target = dest.join(file);
        download_object(&client, &token, &bucket, &object, &target)?;
    }
    Ok(())
}

fn google_access_token(scope: &str) -> Result<String> {
    if let Ok(token) = std::env::var("GOOGLE_ACCESS_TOKEN") {
        let token = token.trim();
        if !token.is_empty() {
            return Ok(token.to_string());
        }
    }

    let runtime = tokio::runtime::Runtime::new().context("creating Tokio runtime")?;
    runtime.block_on(async {
        use yup_oauth2::authenticator::ApplicationDefaultCredentialsTypes;

        let opts = yup_oauth2::ApplicationDefaultCredentialsFlowOpts::default();
        let auth = match yup_oauth2::ApplicationDefaultCredentialsAuthenticator::builder(opts).await
        {
            ApplicationDefaultCredentialsTypes::ServiceAccount(builder) => builder
                .build()
                .await
                .context("building service-account authenticator")?,
            ApplicationDefaultCredentialsTypes::InstanceMetadata(builder) => builder
                .build()
                .await
                .context("building instance-metadata authenticator")?,
        };
        let token = auth
            .token(&[scope])
            .await
            .context("requesting Google access token")?;
        token
            .token()
            .map(ToString::to_string)
            .context("Google auth response did not include an access token")
    })
}

fn download_object(
    client: &reqwest::blocking::Client,
    token: &str,
    bucket: &str,
    object: &str,
    target: &std::path::Path,
) -> Result<()> {
    let url = format!(
        "https://storage.googleapis.com/storage/v1/b/{}/o/{}",
        encode_path_segment(bucket),
        encode_path_segment(object)
    );
    tracing::info!(target: "ragrep::serve", "downloading gs://{bucket}/{object}");
    let resp = client
        .get(url)
        .bearer_auth(token)
        .query(&[("alt", "media")])
        .send()
        .with_context(|| format!("requesting gs://{bucket}/{object}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        bail!("download failed for gs://{bucket}/{object}: HTTP {status}: {body}");
    }

    let tmp = target.with_extension("tmp");
    let mut file =
        std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let bytes = resp
        .bytes()
        .with_context(|| format!("reading gs://{bucket}/{object}"))?;
    std::io::copy(&mut bytes.as_ref(), &mut file)
        .with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, target)
        .with_context(|| format!("renaming {} to {}", tmp.display(), target.display()))?;
    Ok(())
}

fn parse_bucket_spec(raw: &str) -> Result<(String, String)> {
    let trimmed = raw
        .trim()
        .strip_prefix("gs://")
        .unwrap_or(raw.trim())
        .trim_matches('/');
    let Some((bucket, prefix)) = trimmed.split_once('/') else {
        if trimmed.is_empty() {
            bail!("RAGREP_GCS_BUCKET is empty");
        }
        return Ok((trimmed.to_string(), String::new()));
    };
    if bucket.is_empty() {
        bail!("RAGREP_GCS_BUCKET is missing a bucket name");
    }
    Ok((bucket.to_string(), prefix.trim_matches('/').to_string()))
}

fn object_name(prefix: &str, file: &str) -> String {
    if prefix.is_empty() {
        file.to_string()
    } else {
        format!("{prefix}/{file}")
    }
}

fn encode_path_segment(value: &str) -> String {
    let mut out = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let _ = fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bucket_with_prefix() {
        assert_eq!(
            parse_bucket_spec("gs://bucket/path/to/index").unwrap(),
            ("bucket".to_string(), "path/to/index".to_string())
        );
    }

    #[test]
    fn joins_object_prefix() {
        assert_eq!(
            object_name("index/v1", store::FILE_CHUNKS),
            "index/v1/chunks.msgpack"
        );
        assert_eq!(object_name("", store::FILE_BM25), "bm25.msgpack");
    }

    #[test]
    fn percent_encodes_storage_object_names() {
        assert_eq!(encode_path_segment("a/b c"), "a%2Fb%20c");
    }
}

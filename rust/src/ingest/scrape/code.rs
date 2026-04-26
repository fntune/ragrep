//! Scrape source code files from git repositories.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use git2::Repository;
use regex::Regex;
use serde::Serialize;

use super::{expand_repos, repo_name, string_list, write_jsonl};

const MIN_FILE_BYTES: u64 = 100;
const MAX_FILE_BYTES: u64 = 100_000;

#[derive(Debug, Clone, Serialize)]
struct Record {
    repo: String,
    path: String,
    language: String,
    size_bytes: u64,
    content: String,
}

pub fn scrape(raw_dir: &Path, config: &BTreeMap<String, toml::Value>) -> Result<usize> {
    let repos = string_list(config, "repos");
    if repos.is_empty() {
        eprintln!("No code repos configured, skipping");
        return Ok(0);
    }

    let skip_repos: BTreeSet<String> = string_list(config, "skip_repos").into_iter().collect();
    let expanded = expand_repos(&repos);
    let mut records = Vec::new();

    for repo_path in expanded {
        let repo_dir_name = repo_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if skip_repos.contains(repo_dir_name) {
            eprintln!("Skipping {repo_dir_name} (in skip_repos)");
            continue;
        }
        match scrape_repo(&repo_path) {
            Ok(mut repo_records) => records.append(&mut repo_records),
            Err(err) => eprintln!(
                "warning: code scrape failed for {}: {err:#}",
                repo_path.display()
            ),
        }
    }

    let output = raw_dir.join("code.jsonl");
    write_jsonl(&output, &records)?;
    eprintln!(
        "Code scrape complete: {} files written to {}",
        records.len(),
        output.display()
    );
    Ok(records.len())
}

fn scrape_repo(repo_path: &Path) -> Result<Vec<Record>> {
    let repo =
        Repository::open(repo_path).with_context(|| format!("opening {}", repo_path.display()))?;
    let name = repo_name(&repo, repo_path);
    let workdir = repo
        .workdir()
        .with_context(|| format!("{} is bare; cannot scrape files", repo_path.display()))?;
    let index = repo.index().context("reading git index")?;

    let mut records = Vec::new();
    let mut skipped = 0usize;

    for entry in index.iter() {
        let path = String::from_utf8_lossy(&entry.path).to_string();
        if should_exclude(&path) {
            skipped += 1;
            continue;
        }

        let full_path = workdir.join(&path);
        let Ok(meta) = std::fs::metadata(&full_path) else {
            skipped += 1;
            continue;
        };
        let size = meta.len();
        if !(MIN_FILE_BYTES..=MAX_FILE_BYTES).contains(&size) {
            skipped += 1;
            continue;
        }

        let Ok(bytes) = std::fs::read(&full_path) else {
            skipped += 1;
            continue;
        };
        let content = String::from_utf8_lossy(&bytes).into_owned();
        if is_likely_binary(&content) {
            skipped += 1;
            continue;
        }

        let ext = Path::new(&path)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| format!(".{}", ext.to_lowercase()))
            .unwrap_or_default();

        records.push(Record {
            repo: name.clone(),
            path,
            language: ext_to_language(&ext).to_string(),
            size_bytes: size,
            content,
        });
    }

    eprintln!(
        "Scraped {} code files from {} (skipped {})",
        records.len(),
        name,
        skipped
    );
    Ok(records)
}

fn should_exclude(filepath: &str) -> bool {
    let path = Path::new(filepath);
    let basename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| format!(".{}", ext.to_lowercase()))
        .unwrap_or_default();

    if !is_code_extension(&ext) {
        return true;
    }
    if is_excluded_extension(&ext) || is_excluded_filename(basename) || basename == "__init__.py" {
        return true;
    }
    if excluded_path_patterns()
        .iter()
        .any(|pattern| pattern.is_match(filepath))
    {
        return true;
    }
    autogen_dag_re().is_match(filepath)
}

fn is_code_extension(ext: &str) -> bool {
    matches!(
        ext,
        ".py" | ".ts" | ".tsx" | ".js" | ".jsx" | ".go" | ".rs" | ".java" | ".sql" | ".sh"
    )
}

fn is_excluded_extension(ext: &str) -> bool {
    matches!(
        ext,
        ".ipynb"
            | ".csv"
            | ".tsv"
            | ".parquet"
            | ".pkl"
            | ".pickle"
            | ".h5"
            | ".hdf5"
            | ".npy"
            | ".npz"
            | ".feather"
            | ".arrow"
            | ".avro"
            | ".orc"
            | ".xls"
            | ".xlsx"
            | ".ods"
            | ".sqlite"
            | ".db"
            | ".bin"
            | ".model"
            | ".safetensors"
            | ".joblib"
            | ".sav"
            | ".pb"
            | ".onnx"
            | ".pt"
            | ".pth"
            | ".png"
            | ".jpg"
            | ".jpeg"
            | ".gif"
            | ".svg"
            | ".ico"
            | ".bmp"
            | ".webp"
            | ".mp4"
            | ".mp3"
            | ".wav"
            | ".ttf"
            | ".woff"
            | ".woff2"
            | ".eot"
            | ".zip"
            | ".tar"
            | ".gz"
            | ".bz2"
            | ".tgz"
            | ".pdf"
            | ".doc"
            | ".docx"
            | ".pem"
            | ".crt"
            | ".key"
            | ".lock"
            | ".min.js"
            | ".min.css"
            | ".map"
            | ".bat"
            | ".dat"
    )
}

fn is_excluded_filename(name: &str) -> bool {
    matches!(
        name,
        "package-lock.json"
            | "yarn.lock"
            | "pnpm-lock.yaml"
            | "uv.lock"
            | "poetry.lock"
            | "Pipfile.lock"
            | "composer.lock"
            | "Gemfile.lock"
            | "go.sum"
            | "cargo.lock"
            | ".gitignore"
            | ".gitattributes"
            | ".gitmodules"
            | ".editorconfig"
            | ".prettierrc"
            | ".eslintrc"
            | "LICENSE"
            | "LICENSE.md"
            | "LICENSE.txt"
            | "NOTICE"
            | "CHANGELOG.md"
            | "CHANGES.md"
            | "RECORD"
            | "WHEEL"
            | "METADATA"
            | "tokenizer"
            | "moves"
            | "key2row"
            | "vectors"
            | "cfg"
    )
}

fn excluded_path_patterns() -> &'static [Regex] {
    static R: OnceLock<Vec<Regex>> = OnceLock::new();
    R.get_or_init(|| {
        [
            r"(^|/)vendor/",
            r"(^|/)third_party/",
            r"(^|/)node_modules/",
            r"(^|/)\.venv/",
            r"(^|/)venv/",
            r"(^|/)site-packages/",
            r"(^|/)openapi/",
            r"(^|/)openapi_client/",
            r"(^|/)generated/",
            r"_pb2\.py$",
            r"_pb2_grpc\.py$",
            r"\.pb\.go$",
            r"(^|/)alembic/versions/",
            r"(^|/)dist/",
            r"(^|/)build/",
            r"(^|/)\.next/",
            r"(^|/)__pycache__/",
            r"(^|/)\.vscode/",
            r"(^|/)\.idea/",
            r"(^|/)swagger-ui/",
            r"(^|/)static/swagger",
            r"(^|/)htmlcov/",
            r"(^|/)\.coverage",
            r"(^|/)fixtures/.*\.json$",
            r"(^|/)snapshots/",
            r"(^|/)__snapshots__/",
            r"(^|/)data/corpora/",
            r"(^|/)data/tokenizers/",
            r"(^|/)data/raw/",
            r"(^|/)output/model-",
            r"(^|/)load_test/",
            r"(^|/)stopwords/",
        ]
        .into_iter()
        .map(|pattern| Regex::new(pattern).unwrap())
        .collect()
    })
    .as_slice()
}

fn autogen_dag_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"^dags/(shopify/shopify-|hospitality/[^/]+/|ecommerce/[^/]+/|marine/[^/]+/)")
            .unwrap()
    })
}

fn is_likely_binary(content: &str) -> bool {
    let sample = content.chars().take(1024).collect::<Vec<_>>();
    if sample.is_empty() {
        return true;
    }
    let control = sample
        .iter()
        .filter(|c| (**c as u32) < 32 && !matches!(**c, '\n' | '\r' | '\t'))
        .count();
    control as f64 / sample.len() as f64 > 0.1
}

fn ext_to_language(ext: &str) -> &'static str {
    match ext {
        ".py" => "python",
        ".ts" | ".tsx" => "typescript",
        ".js" | ".jsx" => "javascript",
        ".go" => "go",
        ".rs" => "rust",
        ".java" => "java",
        ".sql" => "sql",
        ".sh" => "shell",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn excludes_non_code_and_noise_paths() {
        assert!(should_exclude("node_modules/pkg/index.ts"));
        assert!(should_exclude("src/generated/client.ts"));
        assert!(should_exclude("README.md"));
        assert!(!should_exclude("src/main.rs"));
    }

    #[test]
    fn maps_extensions_to_languages() {
        assert_eq!(ext_to_language(".tsx"), "typescript");
        assert_eq!(ext_to_language(".rs"), "rust");
    }
}

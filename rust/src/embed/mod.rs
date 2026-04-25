pub mod cache;
pub mod gemini;
pub mod openai;
pub mod throttle;
pub mod voyage;

/// One-shot query embedding (sync, used by the CLI). Routes by provider.
pub fn embed_query(provider: &str, model: &str, text: &str) -> anyhow::Result<Vec<f32>> {
    match provider {
        "voyage" => voyage::embed_query(model, text),
        other => anyhow::bail!("embedding provider '{other}' not yet supported in the Rust port"),
    }
}

use anyhow::{bail, Result};
use axum::{
    body::Body,
    extract::State,
    http::{header, HeaderMap, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    Json, Router,
};
use serde_json::json;

const SERVERLESS_AUTHORIZATION: &str = "x-serverless-authorization";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Policy {
    Open,
    Bearer { token: String },
}

impl Policy {
    pub fn from_env() -> Result<Self> {
        if let Some(token) = env_non_empty("RAGREP_AUTH_TOKEN") {
            return Ok(Self::Bearer { token });
        }

        match env_non_empty("RAGREP_AUTH_MODE").as_deref() {
            None | Some("off" | "none" | "open") => Ok(Self::Open),
            Some("bearer" | "token") => {
                bail!(
                    "RAGREP_AUTH_MODE={mode} requires RAGREP_AUTH_TOKEN",
                    mode = env_non_empty("RAGREP_AUTH_MODE").unwrap_or_default()
                )
            }
            Some("cloud-run") => {
                if std::env::var("K_SERVICE").is_err() {
                    bail!("RAGREP_AUTH_MODE=cloud-run is only valid inside Cloud Run; locally, set RAGREP_AUTH_TOKEN or leave auth mode unset")
                }
                Ok(Self::Open)
            }
            Some(mode) => bail!(
                "unsupported RAGREP_AUTH_MODE={mode}; expected off, bearer, token, or cloud-run"
            ),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Bearer { .. } => "bearer",
        }
    }
}

pub fn apply<S>(router: Router<S>, policy: Policy) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    match policy {
        Policy::Open => router,
        Policy::Bearer { .. } => router.layer(middleware::from_fn_with_state(policy, require_auth)),
    }
}

async fn require_auth(State(policy): State<Policy>, req: Request<Body>, next: Next) -> Response {
    match policy {
        Policy::Open => next.run(req).await,
        Policy::Bearer { token } => {
            if bearer_matches(req.headers(), &token) {
                next.run(req).await
            } else {
                unauthorized()
            }
        }
    }
}

fn bearer_matches(headers: &HeaderMap, expected: &str) -> bool {
    bearer_token(headers, header::AUTHORIZATION.as_str())
        .or_else(|| bearer_token(headers, SERVERLESS_AUTHORIZATION))
        .is_some_and(|actual| actual == expected)
}

fn bearer_token<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    let raw = headers.get(name)?.to_str().ok()?.trim();
    raw.strip_prefix("Bearer ")
        .or_else(|| raw.strip_prefix("bearer "))
        .map(str::trim)
        .filter(|token| !token.is_empty())
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "error": "unauthorized" })),
    )
        .into_response()
}

fn env_non_empty(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_authorization_bearer() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            "Bearer secret".parse().expect("valid header"),
        );

        assert!(bearer_matches(&headers, "secret"));
        assert!(!bearer_matches(&headers, "other"));
    }

    #[test]
    fn accepts_serverless_authorization_bearer() {
        let mut headers = HeaderMap::new();
        headers.insert(
            SERVERLESS_AUTHORIZATION,
            "Bearer secret".parse().expect("valid header"),
        );

        assert!(bearer_matches(&headers, "secret"));
    }

    #[test]
    fn rejects_missing_bearer_prefix() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            "secret".parse().expect("valid header"),
        );

        assert!(!bearer_matches(&headers, "secret"));
    }
}

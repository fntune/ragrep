//! Metadata filters + relative-date parsing.
//!
//! Mirrors `parse_filters`, `parse_date`, `matches_filters` in
//! `src/ragrep/search.py`. Date format is YYYY-MM-DD; relative shorthand
//! `Nd`/`Nw`/`Nm`/`Ny` resolves against today (matching Python's
//! `date.today() - timedelta(...)`).

use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use time::macros::format_description;
use time::{Date, Duration, OffsetDateTime};

use crate::models::MetaValue;

const DAYS_IN_MONTH: i64 = 30;
const DAYS_IN_YEAR: i64 = 365;

pub fn parse_filters(raw: &[String]) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for item in raw {
        let (k, v) = item
            .split_once('=')
            .ok_or_else(|| anyhow!("invalid filter {item:?} (expected key=value)"))?;
        out.insert(k.trim().to_string(), v.trim().to_string());
    }
    Ok(out)
}

pub fn parse_date(s: &str) -> Result<String> {
    let s = s.trim().to_lowercase();
    if let Some(rel) = parse_relative(&s) {
        return Ok(rel);
    }
    let fmt = format_description!("[year]-[month]-[day]");
    Date::parse(&s, &fmt).map_err(|e| anyhow!("invalid date {s:?}: {e}"))?;
    Ok(s)
}

fn parse_relative(s: &str) -> Option<String> {
    let mut iter = s.chars();
    let unit = iter.next_back()?;
    let n_str: String = iter.collect();
    let n: i64 = n_str.parse().ok()?;
    let days = match unit {
        'd' => n,
        'w' => n * 7,
        'm' => n * DAYS_IN_MONTH,
        'y' => n * DAYS_IN_YEAR,
        _ => return None,
    };
    let today = OffsetDateTime::now_local()
        .unwrap_or_else(|_| OffsetDateTime::now_utc())
        .date();
    let target = today - Duration::days(days);
    let fmt = format_description!("[year]-[month]-[day]");
    target.format(&fmt).ok()
}

pub fn matches(
    metadata: &BTreeMap<String, MetaValue>,
    filters: &BTreeMap<String, String>,
    after: Option<&str>,
    before: Option<&str>,
) -> bool {
    for (k, needle) in filters {
        let Some(chunk_val) = metadata.get(k) else {
            return false;
        };
        let haystack = match chunk_val {
            MetaValue::Str(s) => s.to_lowercase(),
            MetaValue::Int(i) => i.to_string(),
            MetaValue::Float(f) => f.to_string(),
        };
        if !haystack.contains(&needle.to_lowercase()) {
            return false;
        }
    }

    if after.is_some() || before.is_some() {
        let date_str = match metadata.get("date") {
            Some(MetaValue::Str(s)) => s.clone(),
            Some(MetaValue::Int(i)) => i.to_string(),
            _ => return false,
        };
        if date_str.len() < 10 {
            return false;
        }
        let prefix = &date_str[..10];
        if let Some(a) = after {
            if prefix < a {
                return false;
            }
        }
        if let Some(b) = before {
            if prefix >= b {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_filters_basic() {
        let r = parse_filters(&["channel=eng".into(), "user=Naoki".into()]).unwrap();
        assert_eq!(r.get("channel").map(String::as_str), Some("eng"));
        assert_eq!(r.get("user").map(String::as_str), Some("Naoki"));
    }

    #[test]
    fn parse_filters_invalid() {
        assert!(parse_filters(&["no-equals".into()]).is_err());
    }

    #[test]
    fn parse_date_iso() {
        assert_eq!(parse_date("2025-01-15").unwrap(), "2025-01-15");
    }

    #[test]
    fn parse_date_relative_units() {
        // We can't assert the exact value (depends on today), but format must be YYYY-MM-DD.
        for s in ["7d", "2w", "3m", "1y"] {
            let out = parse_date(s).unwrap();
            assert_eq!(out.len(), 10, "got: {out}");
            assert_eq!(out.chars().nth(4), Some('-'));
        }
    }

    #[test]
    fn parse_date_invalid() {
        assert!(parse_date("garbage").is_err());
        assert!(parse_date("2025-13-99").is_err());
    }

    #[test]
    fn matches_substring_case_insensitive() {
        let mut md = BTreeMap::new();
        md.insert(
            "channel".to_string(),
            MetaValue::Str("Eng-Platform".to_string()),
        );
        let f: BTreeMap<String, String> = [("channel".to_string(), "platform".to_string())]
            .into_iter()
            .collect();
        assert!(matches(&md, &f, None, None));
    }

    #[test]
    fn matches_date_window() {
        let mut md = BTreeMap::new();
        md.insert("date".to_string(), MetaValue::Str("2025-06-01".to_string()));
        let empty = BTreeMap::new();
        assert!(matches(&md, &empty, Some("2025-01-01"), Some("2025-12-31")));
        assert!(!matches(&md, &empty, Some("2025-12-31"), None));
        assert!(!matches(&md, &empty, None, Some("2025-01-01")));
    }
}

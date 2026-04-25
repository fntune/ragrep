use std::io::{self, IsTerminal, Write};

const LOGO: &[&str] = &[
    "██████╗  █████╗  ██████╗ ██████╗ ███████╗██████╗ ",
    "██╔══██╗██╔══██╗██╔════╝ ██╔══██╗██╔════╝██╔══██╗",
    "██████╔╝███████║██║  ███╗██████╔╝█████╗  ██████╔╝",
    "██╔══██╗██╔══██║██║   ██║██╔══██╗██╔══╝  ██╔═══╝ ",
    "██║  ██║██║  ██║╚██████╔╝██║  ██║███████╗██║     ",
    "╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝     ",
];

pub fn print() {
    let stdout = io::stdout();
    let use_color = stdout.is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let mut out = stdout.lock();

    let _ = writeln!(out);
    for line in LOGO {
        let _ = writeln!(out, "  {}", color(line, "38;5;81", use_color));
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {}",
        color("ripgrep for your team's knowledge base.", "1", use_color)
    );
    let _ = writeln!(
        out,
        "  {}",
        color(
            "hybrid retrieval · self-hosted · single command.",
            "2",
            use_color
        )
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "  {}", color("Get started", "1;4", use_color));
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {}",
        color("Point at a running server:", "2", use_color)
    );
    let _ = writeln!(out, "    export RAGREP_SERVER=http://your-server:8321");
    let _ = writeln!(out, "    ragrep \"your question\"");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {}",
        color("Or build a local index:", "2", use_color)
    );
    let _ = writeln!(
        out,
        "    git clone https://github.com/fntune/ragrep && cd ragrep"
    );
    let _ = writeln!(
        out,
        "    cp .env.example .env{}",
        color("   # add VOYAGE_API_KEY, SLACK_TOKEN, etc.", "2", use_color)
    );
    let _ = writeln!(out, "    ragrep ingest");
    let _ = writeln!(out, "    ragrep \"your question\"");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {}https://ragrep.cc",
        color("Docs   ", "2", use_color)
    );
    let _ = writeln!(
        out,
        "  {}https://github.com/fntune/ragrep/issues",
        color("Issues ", "2", use_color)
    );
    let _ = writeln!(out);
}

fn color(text: &str, code: &str, on: bool) -> String {
    if on {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

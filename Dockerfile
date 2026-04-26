FROM rust:1.89-bookworm AS build

RUN apt-get update && \
    apt-get install -y --no-install-recommends cmake pkg-config zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY README.md ./
COPY rust/Cargo.toml rust/Cargo.lock ./rust/
COPY rust/src ./rust/src
COPY rust/examples ./rust/examples

RUN cargo build --manifest-path rust/Cargo.toml --release --bin ragrep

FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /src/rust/target/release/ragrep /usr/local/bin/ragrep
COPY config.toml ./config.toml

ENV RAGREP_CONFIG=/app/config.toml
EXPOSE 8080

CMD ["ragrep", "serve", "--host", "0.0.0.0", "--port", "8080"]

# regit-covariance — Task runner
# Run `just` to see available recipes.

# Quality gate — run all checks across the workspace
check: fmt-check lint test doc
    @echo "All checks passed."

# Format check
fmt-check:
    cargo fmt --all --check

# Format (fix)
fmt:
    cargo fmt --all

# Lint (workspace, all targets, warnings as errors)
lint:
    cargo clippy --workspace --all-targets -- -D warnings

# Test (workspace)
test:
    cargo test --workspace

# Build docs (workspace)
doc:
    cargo doc --workspace --no-deps

# Build release (workspace)
build:
    cargo build --workspace --release

# Run the demo HTTP server
serve *ARGS:
    cargo run -p regit-covariance-server --release -- {{ARGS}}

# Run the library quickstart example
example:
    cargo run -p regit-covariance --example quickstart

# Benchmark the math core
bench:
    cargo bench -p regit-covariance

# Dependency audit (requires cargo-deny)
deny:
    cargo deny check

# WASM build smoke test for the math core
wasm:
    cargo build -p regit-covariance --target wasm32-unknown-unknown --release

# regit-covariance — Task runner
# Run `just` to see available recipes.

# Quality gate — run all checks
check: fmt-check lint test doc
    @echo "All checks passed."

# Format check
fmt-check:
    cargo fmt --check

# Format (fix)
fmt:
    cargo fmt

# Lint
lint:
    cargo clippy -- -D warnings

# Test
test:
    cargo test

# Build docs
doc:
    cargo doc --no-deps

# Build release
build:
    cargo build --release

# Run
run *ARGS:
    cargo run -- {{ARGS}}

# Benchmark
bench:
    cargo bench

# Dependency audit (requires cargo-deny)
deny:
    cargo deny check

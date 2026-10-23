---
name: rust-builder
description: Rust/Cargo build automation for compiling, testing, and packaging Rust projects
author: jarvis
version: 1.0.0
tags: [rust, cargo, build, compile, test]
metadata:
  jarvis:
    requires:
      bins: [cargo, rustc]
      env: []
      config: []
    always: false
    user-invocable: true
    emoji: "🦀"
---

# Rust Builder Skill

Comprehensive Rust/Cargo build automation for compiling, testing, and packaging Rust projects.

## Prerequisites

Requires Rust toolchain installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Project Detection

When working in a Rust project, detect:
- `Cargo.toml` location
- Project type (binary, library, workspace)
- Edition (2021, 2018, etc.)
- Dependencies status

## Build Commands

### Compile
```bash
cargo build                # Debug build
cargo build --release      # Optimized release build
cargo build --all          # Build all crates in workspace
cargo build --target x86_64-unknown-linux-musl  # Cross-compile
```

### Check (Fast)
```bash
cargo check                # Type check without codegen
cargo check --all-targets  # Check all targets including tests
cargo clippy               # Lint with clippy
```

### Test
```bash
cargo test                 # Run all tests
cargo test --release       # Test optimized code
cargo test --doc           # Run doc tests
cargo test <test_name>     # Run specific test
cargo test -- --nocapture  # Show test output
cargo nextest run          # Faster test runner (if installed)
```

### Documentation
```bash
cargo doc                  # Generate documentation
cargo doc --open           # Generate and open docs
cargo doc --no-deps        # Only doc this crate
```

### Run
```bash
cargo run                  # Build and run
cargo run --release        # Run optimized
cargo run --bin <name>     # Run specific binary
cargo run --example <name> # Run example
```

## Project Management

### Dependencies
```bash
cargo add <crate>          # Add dependency
cargo add <crate> --features "feat1,feat2"
cargo remove <crate>       # Remove dependency
cargo update               # Update dependencies
cargo tree                 # Show dependency tree
cargo outdated             # Check for updates (needs cargo-outdated)
```

### Formatting & Linting
```bash
cargo fmt                  # Format code
cargo fmt -- --check       # Check formatting
cargo clippy               # Run linter
cargo clippy -- -W clippy::all  # All warnings
cargo clippy -- -D warnings     # Fail on warnings
```

### Security Audit
```bash
cargo audit                # Check for vulnerabilities
cargo deny check           # Policy checking (needs cargo-deny)
```

## Release Workflow

### Version Bumping
```bash
cargo set-version 1.2.0    # Set version (needs cargo-set-version)
# Or manually edit Cargo.toml
```

### Publishing
```bash
cargo publish              # Publish to crates.io
cargo publish --dry-run    # Test publish without uploading
```

## Workspace Commands

```bash
cargo build --workspace    # Build all members
cargo test --workspace     # Test all members
cargo run -p <crate>       # Run specific workspace member
```

## Common Issues

### Dependencies Won't Compile
```bash
cargo clean
rm Cargo.lock
cargo build
```

### Feature Flags
```bash
cargo build --features "feature1,feature2"
cargo build --all-features
cargo build --no-default-features
```

### Build Scripts
If build fails due to missing native libs, check `build.rs` for system dependencies.

## Best Practices

1. Always `cargo check` before `cargo build` for faster feedback
2. Use `cargo clippy` for linting before commits
3. Run `cargo fmt` to maintain consistent style
4. Use `cargo audit` regularly for security
5. For large projects, consider `cargo nextest` for faster tests
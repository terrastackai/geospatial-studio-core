# Contributing to GEOStudio Core

We welcome contributions from the community and strive to make the process as smooth as possible.

---

## Getting Started

If you are new to the project, the best way to begin is by exploring open issues:

👉 **Issues:** https://github.com/terrastackai/geospatial-studio-core/issues

### Contribution Workflow

1. **Fork and Clone** the repository

    ```sh
    git clone https://github.com/terrastackai/geospatial-studio-core
    cd geospatial-studio-core
    ```

2. **Create a branch:** Use a descriptive branch naming convention:

    - `feature/<short-description>`
    - `bugfix/<ticket-number>-<short-description>` or
    - `docs/<short-description>`
    - `chore/<short-description>`

3. **Project Setup:** To set up the project for local development, follow the official **[Run Locally](README.md#local-setup-with-venv)** section in the README.

4. **Make changes**

5. **Write Tests**

    - All new features must include tests
    - The project uses **pytest**
    - To run the tests: `pytest -vv`

6. **Code Quality & Pre-Commit Hooks**
    This project uses **pre-commit** to enforce consistent formatting, linting, and security checks before code is committed.

    Install pre-commit (if not already installed)

    ```bash
    pip install -r requirements-dev.txt
    pre-commit install
    ```

7. **Commit Messages**

    Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard:

    - `feat:` new feature  
    - `fix:` bug fix  
    - `docs:` documentation only  
    - `refactor:` structural change  
    - `test:` test additions  
    - `chore:` tooling updates  

    Example: `feat: add redis connection pooling to reduce latency`

---

## CI/CD Requirements

All PRs must pass automated checks:

- Linting
- Type checking
- Tests
- Security scanning (e.g., detect-secrets)

---

## Code of Conduct

This project adheres to a [Code of Conduct](./CODE_OF_CONDUCT.md).
By participating, you agree to uphold these standards.

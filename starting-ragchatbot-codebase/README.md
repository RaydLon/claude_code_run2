# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.


## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- An Anthropic API key (for Claude AI)
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Development

This project uses modern code quality tools to maintain consistent formatting and catch potential issues.

### Setup Development Environment

Install development dependencies:
```bash
uv sync --extra dev
```

This installs Black (formatter), Ruff (linter), mypy (type checker), pytest (testing), and pre-commit (optional hooks).

### Code Quality Scripts

Run quality checks using the provided scripts:

```bash
# Run all quality checks (recommended before committing)
./scripts/quality.sh

# Individual tools
./scripts/format.sh      # Format code with Black
./scripts/lint.sh        # Lint with Ruff (auto-fix)
./scripts/typecheck.sh   # Type check with mypy
./scripts/test.sh        # Run tests with coverage
```

### Testing

Tests are located in `backend/tests/`. Run tests with coverage:
```bash
./scripts/test.sh
```

View coverage report:
```bash
open htmlcov/index.html
```

### Optional Pre-commit Hooks

Automatically run quality checks before each commit:
```bash
uv run pre-commit install
```


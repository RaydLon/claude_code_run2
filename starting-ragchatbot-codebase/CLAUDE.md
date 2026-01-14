# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) chatbot** that uses Anthropic's Claude with tool calling to answer questions about course materials. The system uses ChromaDB for vector storage and sentence-transformers for embeddings.

## Development Commands

### Setup
```bash
# Install dependencies using uv package manager
uv sync

# Create environment file
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

**IMPORTANT: Always use `uv` for running commands. Do NOT use pip, python, or python3 directly.**

### Running the Application
```bash
# Quick start (from project root)
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000

# Running any Python script
uv run python script_name.py
```

**Access Points:**
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Development Notes
- **Always use `uv run` to execute Python commands** - this ensures correct dependencies and environment
- No test suite exists in this codebase
- ChromaDB data persists in `./chroma_db/` directory
- Sessions are in-memory only (lost on server restart)

## Architecture

### Core Data Flow

```
User Query (Frontend)
    ↓
FastAPI /api/query
    ↓
RAGSystem.query()
    ↓
AIGenerator.generate_response() with tools
    ↓
Claude decides to use CourseSearchTool
    ↓
VectorStore.search() (ChromaDB semantic search)
    ↓
Claude synthesizes answer from search results
    ↓
Response + Sources returned to Frontend
```

### Component Responsibilities

**RAGSystem** (`rag_system.py`) - Main orchestrator
- Coordinates document processing, vector storage, AI generation, and session management
- Entry point for document ingestion (`add_course_document`, `add_course_folder`)
- Entry point for queries (`query`)

**VectorStore** (`vector_store.py`) - ChromaDB wrapper
- Manages **two collections**:
  - `course_catalog`: Course metadata (titles, instructors, links) for fuzzy course name matching
  - `course_content`: Actual lesson chunks with embeddings for semantic search
- Search resolution: Converts course names to exact titles via semantic search, then filters content

**DocumentProcessor** (`document_processor.py`) - Text parsing
- Extracts structured data from course documents (title, instructor, lessons)
- Chunks text using sentence-based splitting with overlap (800 chars, 100 char overlap)
- Adds contextual prefixes to chunks: `"Course {title} Lesson {number} content: {text}"`

**AIGenerator** (`ai_generator.py`) - Claude API interface
- Uses tool calling pattern (not traditional RAG)
- Static system prompt emphasizes: one search per query, no meta-commentary, brief responses
- Temperature: 0 (deterministic), Max tokens: 800

**SessionManager** (`session_manager.py`) - Conversation memory
- In-memory storage (sessions don't persist across restarts)
- Stores up to `MAX_HISTORY * 2` messages (default: 4 messages = 2 Q&A pairs)
- Session IDs: `session_1`, `session_2`, etc.

**ToolManager & CourseSearchTool** (`search_tools.py`) - Tool definitions
- Implements Anthropic's tool calling interface
- `CourseSearchTool`: Allows Claude to search by query, optional course name, optional lesson number
- Tracks sources for frontend display (stored in `last_sources`)

### Key Architectural Decisions

**Tool-Based Search (Not Traditional RAG)**
- Claude decides whether to search based on query context
- Claude chooses search parameters (course name, lesson number)
- Enables natural follow-up questions without explicit tool calls

**Two-Collection Strategy**
- `course_catalog` enables fuzzy course name resolution via semantic search
- `course_content` enables filtered semantic search by exact course title and/or lesson number
- Prevents wrong course matches while allowing natural language course queries

**Chunk Context Enrichment**
- Each chunk includes course and lesson metadata in the text itself
- Helps Claude understand chunk origin without additional lookups
- Format: `"Course {title} Lesson {number} content: {chunk_text}"`

## Important Implementation Details

### Document Format Expected

Course documents must follow this structure:
```
Line 1: Course Title: [title]
Line 2: Course Link: [optional URL]
Line 3: Course Instructor: [optional name]
[Empty line]
Lesson 0: Introduction
Lesson Link: [optional URL]
[lesson content]
Lesson 1: Main Topic
[lesson content]
```

- Title defaults to filename if "Course Title:" prefix missing
- Lesson numbers extracted via regex: `Lesson \d+: [title]`
- Lesson links are parsed but **not currently returned to frontend**

### Chunking Algorithm

Sentence-based chunking that:
1. Splits on sentence boundaries using regex (handles abbreviations)
2. Builds chunks up to `CHUNK_SIZE` (800 chars) without breaking sentences
3. Overlaps chunks by `CHUNK_OVERLAP` (100 chars) calculated backwards from chunk end
4. Guarantees progress by at least 1 sentence per chunk

### Course Name Resolution

When searching with `course_name` parameter:
1. VectorStore queries `course_catalog` collection semantically
2. Returns best matching course title (used as exact filter)
3. If no match found, returns error (no fallback)
4. Semantic matching means "MCP" can match "MCP: Build Rich-Context AI Apps"

### Tool Execution Flow

1. **Message 1**: User query sent to Claude with tool definitions
2. **Claude Response**: Decides to use `search_course_content` tool (stop_reason: "tool_use")
3. **Tool Execution**: App executes search, collects results
4. **Message 2**: Search results sent back to Claude
5. **Final Response**: Claude synthesizes answer from search results

### Session Management

- Sessions created on first query if no session_id provided
- History trimmed when `> max_history * 2` messages (keeps most recent)
- History formatted as: `"User: {msg}\nAssistant: {msg}"`
- Passed to Claude in system prompt under "Previous conversation:"

## Configuration

All settings in `config.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `ANTHROPIC_API_KEY` | from .env | Claude API access |
| `ANTHROPIC_MODEL` | claude-sonnet-4-20250514 | Model version |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | SentenceTransformers model |
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `MAX_RESULTS` | 5 | Search results per query |
| `MAX_HISTORY` | 2 | Conversation pairs to remember |
| `CHROMA_PATH` | ./chroma_db | Persistent storage location |

## Common Gotchas

### Sessions Lost on Restart
Sessions are in-memory only. Server restart clears all conversation history, but course data in ChromaDB persists.

### Course Deduplication
On startup, existing courses are not re-indexed. To force rebuild:
- Delete `./chroma_db/` directory, OR
- Change course document metadata (title/instructor), OR
- Call `add_course_folder(path, clear_existing=True)` programmatically

### Sources Tracking
Sources come from tool execution, not metadata lookups. If Claude doesn't use the search tool (e.g., for general knowledge questions), sources list will be empty.

### CORS Configuration
`allow_origins=["*"]` means any origin can call the API. Fine for local development, not for production.

### UTF-8 Decoding Errors
Document processor ignores UTF-8 errors (`errors='ignore'`), which silently loses non-UTF-8 data.

### Course Catalog Query Limitations
Semantic search can match wrong course if titles are very similar. No exact-match fallback exists.

## Adding New Course Materials

1. Create a formatted `.txt`, `.pdf`, or `.docx` file following the document format
2. Place it in the `/docs` directory
3. Restart the server (or call the API endpoint if implemented)
4. Server automatically processes and indexes on startup

## Modifying Search Behavior

**To change search logic:**
- Edit `VectorStore.search()` in `vector_store.py`
- Edit `CourseSearchTool.execute()` in `search_tools.py`

**To add new tool:**
1. Create class implementing `Tool` abstract base class
2. Implement `get_tool_definition()` and `execute()` methods
3. Register with `ToolManager` in `RAGSystem.__init__()`

**To change chunk size/overlap:**
- Edit `CHUNK_SIZE` and `CHUNK_OVERLAP` in `config.py`
- Delete `./chroma_db/` to rebuild with new settings

## File Purposes

| File | Key Responsibility |
|------|-------------------|
| `app.py` | FastAPI routes, startup document loading, static file serving |
| `rag_system.py` | Orchestrates all components, main query/ingestion entry point |
| `vector_store.py` | ChromaDB interface, two-collection management, search resolution |
| `document_processor.py` | Course document parsing, text chunking, context enrichment |
| `ai_generator.py` | Claude API calls, tool execution handling, response generation |
| `session_manager.py` | In-memory conversation history tracking |
| `search_tools.py` | Tool definitions, search tool implementation, source tracking |
| `models.py` | Pydantic data models (Course, Lesson, CourseChunk) |
| `config.py` | Centralized configuration with .env loading |

## Python Requirements

- Python >= 3.13 (specified in pyproject.toml)
- uv package manager
- Dependencies locked in `uv.lock` for reproducibility

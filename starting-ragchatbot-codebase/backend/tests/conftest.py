"""
Shared pytest fixtures for RAG chatbot test suite.

This module provides:
- Two Config instances (buggy with MAX_RESULTS=0, working with MAX_RESULTS=5)
- VectorStore instances using both configs with real ChromaDB
- Sample test queries and expected course data
- Mock objects for Anthropic API responses
- FastAPI test client and app fixtures for API endpoint testing
"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from config import Config
from vector_store import VectorStore
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def buggy_config():
    """
    Config with MAX_RESULTS=0 (the bug).

    This fixture represents the current broken state where MAX_RESULTS=0
    causes all searches to return empty results.

    Returns:
        Config: Configuration with MAX_RESULTS=0
    """
    return Config(
        ANTHROPIC_API_KEY="test-api-key",
        ANTHROPIC_MODEL="claude-sonnet-4-20250514",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        MAX_RESULTS=0,  # THE BUG
        MAX_HISTORY=2,
        CHROMA_PATH="./chroma_db"
    )


@pytest.fixture
def working_config():
    """
    Config with MAX_RESULTS=5 (the fix).

    This fixture represents the corrected state where MAX_RESULTS=5
    allows searches to return up to 5 results.

    Returns:
        Config: Configuration with MAX_RESULTS=5
    """
    return Config(
        ANTHROPIC_API_KEY="test-api-key",
        ANTHROPIC_MODEL="claude-sonnet-4-20250514",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        MAX_RESULTS=5,  # THE FIX
        MAX_HISTORY=2,
        CHROMA_PATH="./chroma_db"
    )


# ============================================================================
# VectorStore Fixtures (Using Real ChromaDB)
# ============================================================================

@pytest.fixture
def buggy_vector_store(buggy_config):
    """
    VectorStore instance with MAX_RESULTS=0.

    Uses the real ChromaDB at ./chroma_db with existing course data.
    This will demonstrate the bug where searches return no results.

    Returns:
        VectorStore: Instance configured with MAX_RESULTS=0
    """
    return VectorStore(
        chroma_path=buggy_config.CHROMA_PATH,
        embedding_model=buggy_config.EMBEDDING_MODEL,
        max_results=buggy_config.MAX_RESULTS
    )


@pytest.fixture
def working_vector_store(working_config):
    """
    VectorStore instance with MAX_RESULTS=5.

    Uses the real ChromaDB at ./chroma_db with existing course data.
    This will demonstrate correct behavior where searches return results.

    Returns:
        VectorStore: Instance configured with MAX_RESULTS=5
    """
    return VectorStore(
        chroma_path=working_config.CHROMA_PATH,
        embedding_model=working_config.EMBEDDING_MODEL,
        max_results=working_config.MAX_RESULTS
    )


# ============================================================================
# Tool Fixtures
# ============================================================================

@pytest.fixture
def buggy_course_search_tool(buggy_vector_store):
    """
    CourseSearchTool with buggy VectorStore (MAX_RESULTS=0).

    Returns:
        CourseSearchTool: Tool instance that will return empty results
    """
    return CourseSearchTool(buggy_vector_store)


@pytest.fixture
def working_course_search_tool(working_vector_store):
    """
    CourseSearchTool with working VectorStore (MAX_RESULTS=5).

    Returns:
        CourseSearchTool: Tool instance that returns proper results
    """
    return CourseSearchTool(working_vector_store)


@pytest.fixture
def tool_manager_with_buggy_tool(buggy_course_search_tool):
    """
    ToolManager with buggy CourseSearchTool registered.

    Returns:
        ToolManager: Manager with tool that returns empty results
    """
    manager = ToolManager()
    manager.register_tool(buggy_course_search_tool)
    return manager


@pytest.fixture
def tool_manager_with_working_tool(working_course_search_tool):
    """
    ToolManager with working CourseSearchTool registered.

    Returns:
        ToolManager: Manager with tool that returns proper results
    """
    manager = ToolManager()
    manager.register_tool(working_course_search_tool)
    return manager


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_queries():
    """
    Sample test queries that should return results from ChromaDB.

    These queries are designed to match content in the existing courses.

    Returns:
        dict: Dictionary of query types and their test queries
    """
    return {
        "general_content": "computer use",
        "specific_topic": "prompt caching",
        "technical_term": "embeddings",
        "course_specific": "Anthropic models",
        "lesson_specific": "introduction to course"
    }


@pytest.fixture
def expected_courses():
    """
    Expected course data from ChromaDB.

    This data matches what's currently in the chroma_db/ directory.

    Returns:
        dict: Course information for validation
    """
    return {
        "building_computer_use": {
            "title": "Building Towards Computer Use with Anthropic",
            "instructor": "Colt Steele",
            "lesson_count": 7,
            "has_lesson_0": True,
            "has_lesson_1": True
        }
    }


@pytest.fixture
def course_name_variations():
    """
    Different ways users might reference course names.

    Tests fuzzy course name matching via semantic search in course_catalog.

    Returns:
        dict: Variations mapped to expected course titles
    """
    return {
        "computer use": "Building Towards Computer Use with Anthropic",
        "Building Toward": "Building Towards Computer Use with Anthropic"
    }


# ============================================================================
# Mock Anthropic API Fixtures
# ============================================================================

@pytest.fixture
def mock_anthropic_client():
    """
    Mock Anthropic client for testing AIGenerator without API calls.

    Returns:
        Mock: Mock client with messages.create() method
    """
    mock_client = Mock()
    mock_client.messages = Mock()
    return mock_client


@pytest.fixture
def mock_tool_use_response():
    """
    Mock Anthropic API response indicating tool use.

    Simulates Claude deciding to use the search_course_content tool.
    This is the typical response format when Claude calls a tool.

    Returns:
        Mock: Response object with stop_reason="tool_use"
    """
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"

    # Create mock tool use block
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.id = "toolu_01A09q90qw90lq917835lq9"
    tool_block.name = "search_course_content"
    tool_block.input = {
        "query": "computer use"
    }

    mock_response.content = [tool_block]
    return mock_response


@pytest.fixture
def mock_text_response():
    """
    Mock Anthropic API response with text only (no tool use).

    Simulates Claude responding directly without calling tools.
    This happens for general knowledge questions.

    Returns:
        Mock: Response object with stop_reason="end_turn"
    """
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"

    # Create mock text block
    text_block = Mock()
    text_block.type = "text"
    text_block.text = "Computer use refers to Claude's ability to control computers."

    mock_response.content = [text_block]
    return mock_response


@pytest.fixture
def mock_final_response_after_tool():
    """
    Mock Anthropic API response after tool execution.

    This is the second response Claude makes after receiving tool results.
    It synthesizes the search results into a final answer.

    Returns:
        Mock: Response with synthesized answer
    """
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"

    text_block = Mock()
    text_block.type = "text"
    text_block.text = (
        "Computer use is a feature where Claude can control a computer "
        "by taking screenshots and generating mouse clicks and keystrokes."
    )

    mock_response.content = [text_block]
    return mock_response


# ============================================================================
# Multi-Round Tool Calling Fixtures
# ============================================================================

@pytest.fixture
def mock_second_tool_use_response():
    """
    Mock Anthropic API response for second tool call in a sequence.

    Simulates Claude making a second tool call after reviewing first results.

    Returns:
        Mock: Response object with stop_reason="tool_use" for second tool
    """
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"

    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.id = "toolu_02SECOND"
    tool_block.name = "search_course_content"
    tool_block.input = {
        "query": "lesson 2 content",
        "lesson_number": 2
    }

    mock_response.content = [tool_block]
    return mock_response


@pytest.fixture
def mock_comparison_final_response():
    """
    Mock final response after two sequential tool calls.

    Simulates Claude synthesizing comparison after getting two pieces of info.

    Returns:
        Mock: Response with synthesized comparison
    """
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"

    text_block = Mock()
    text_block.type = "text"
    text_block.text = (
        "Based on the course materials:\n"
        "Lesson 1 focuses on X, while Lesson 2 covers Y. "
        "The main difference is Z."
    )

    mock_response.content = [text_block]
    return mock_response


# ============================================================================
# FastAPI Test Client Fixtures
# ============================================================================

@pytest.fixture
def test_app(working_config):
    """
    Create a test FastAPI app without static file mounting.

    This avoids issues with missing frontend directory during tests.
    The app includes all API endpoints but no static file serving.

    Returns:
        FastAPI: Test application instance
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Define Pydantic models inline
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        text: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # Create test app
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")

    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Initialize RAG system
    rag_system = RAGSystem(working_config)

    # Define API endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            # Create session if not provided
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()

            # Process query using RAG system
            answer, sources = rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Store rag_system as app state for test access
    app.state.rag_system = rag_system

    return app


@pytest.fixture
def test_client(test_app):
    """
    Create a test client for the FastAPI app.

    Uses FastAPI's TestClient which is based on requests library.
    Allows making HTTP requests to the app without running a server.

    Returns:
        TestClient: Client for making test requests
    """
    return TestClient(test_app)


@pytest.fixture
def mock_rag_query_response():
    """
    Mock response from RAGSystem.query() for API testing.

    Simulates a successful query with answer and sources.

    Returns:
        tuple: (answer, sources) matching RAGSystem.query() return format
    """
    answer = "This is a test answer from the RAG system."
    sources = [
        {
            "text": "Source 1: Course content excerpt",
            "url": "http://example.com/course1"
        },
        {
            "text": "Source 2: Another course excerpt",
            "url": None
        }
    ]
    return (answer, sources)


@pytest.fixture
def mock_course_analytics():
    """
    Mock response from RAGSystem.get_course_analytics().

    Returns:
        dict: Course analytics with total_courses and course_titles
    """
    return {
        "total_courses": 2,
        "course_titles": [
            "Building Towards Computer Use with Anthropic",
            "Prompt Engineering Fundamentals"
        ]
    }

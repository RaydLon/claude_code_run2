"""
Test suite for RAG system integration.

Tests end-to-end functionality including:
- Complete query workflow from RAGSystem.query() to response
- Tool registration and management
- Source tracking and retrieval
- Session management

Uses real ChromaDB but mocks Anthropic API.

Test Strategy:
- Integration tests that verify complete system behavior
- Tests demonstrate how MAX_RESULTS=0 bug propagates through entire system
- Tests verify source tracking works correctly after fix
"""

import pytest
from unittest.mock import Mock, patch
from rag_system import RAGSystem


class TestRAGSystemInitialization:
    """Test RAG system initialization and component setup."""

    def test_buggy_rag_system_initializes_with_zero_max_results(self, buggy_config):
        """
        Test that RAG system initializes with buggy config.

        Expected: PASSES with both configs (initialization succeeds)
        """
        rag_system = RAGSystem(buggy_config)

        # Verify components initialized
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.tool_manager is not None
        assert rag_system.session_manager is not None

        # Verify vector store has MAX_RESULTS=0
        assert rag_system.vector_store.max_results == 0


    def test_working_rag_system_initializes_with_correct_max_results(
        self,
        working_config
    ):
        """
        Test that RAG system initializes with working config.

        Expected: PASSES with both configs
        """
        rag_system = RAGSystem(working_config)

        # Verify vector store has MAX_RESULTS=5
        assert rag_system.vector_store.max_results == 5


    def test_rag_system_registers_tools(self, working_config):
        """
        Test that RAG system registers CourseSearchTool on initialization.

        Expected: PASSES with both configs
        """
        rag_system = RAGSystem(working_config)

        # Verify tool registered
        tool_defs = rag_system.tool_manager.get_tool_definitions()
        assert len(tool_defs) >= 1

        # Check for search tool
        tool_names = [tool["name"] for tool in tool_defs]
        assert "search_course_content" in tool_names


class TestRAGSystemQueryWithBuggyConfig:
    """Test RAG system query behavior with MAX_RESULTS=0 bug."""

    @patch('ai_generator.anthropic.Anthropic')
    def test_buggy_query_returns_response_based_on_empty_results(
        self,
        mock_anthropic_class,
        buggy_config,
        mock_tool_use_response,
        mock_final_response_after_tool
    ):
        """
        Bug demonstration: Query with MAX_RESULTS=0 gives Claude empty results.

        Expected: PASSES with current config (Claude gets "No relevant content found")
        """
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]

        # Create RAG system with buggy config
        rag_system = RAGSystem(buggy_config)
        rag_system.ai_generator.client = mock_client

        # Execute query
        response, sources = rag_system.query("What is computer use?")

        # Should get a response (Claude tries to answer anyway)
        assert isinstance(response, str)
        assert len(response) > 0

        # Bug: Sources should be empty because tool found nothing
        assert len(sources) == 0, "Buggy config should result in no sources"

        # Verify tool was called but returned empty
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        tool_result = next(
            c for c in messages[2]["content"] if c["type"] == "tool_result"
        )

        # Bug: Tool result contains "No relevant content found"
        assert "No relevant content found" in tool_result["content"]


class TestRAGSystemQueryWithWorkingConfig:
    """Test RAG system query behavior with correct MAX_RESULTS=5."""

    @patch('ai_generator.anthropic.Anthropic')
    def test_working_query_returns_response_with_sources(
        self,
        mock_anthropic_class,
        working_config,
        mock_tool_use_response,
        mock_final_response_after_tool
    ):
        """
        Fix demonstration: Query with MAX_RESULTS=5 gives Claude actual results.

        Expected: FAILS with buggy config (no sources)
        Expected: PASSES with working config (sources populated)
        """
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]

        # Create RAG system with working config
        rag_system = RAGSystem(working_config)
        rag_system.ai_generator.client = mock_client

        # Execute query
        response, sources = rag_system.query("What is computer use?")

        # Should get response
        assert isinstance(response, str)
        assert len(response) > 0

        # Fix: Sources should be populated
        assert len(sources) > 0, "Working config should return sources"

        # Validate source structure
        for source in sources:
            assert "text" in source
            assert "url" in source

        # Verify tool returned actual content
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        tool_result = next(
            c for c in messages[2]["content"] if c["type"] == "tool_result"
        )

        # Fix: Tool result contains actual content
        assert "No relevant content found" not in tool_result["content"]
        assert len(tool_result["content"]) > 0


class TestRAGSystemSourceTracking:
    """Test that RAG system correctly tracks and returns sources."""

    @patch('ai_generator.anthropic.Anthropic')
    def test_working_system_tracks_sources_from_tool(
        self,
        mock_anthropic_class,
        working_config,
        mock_tool_use_response,
        mock_final_response_after_tool
    ):
        """
        Test that sources from tool execution are properly tracked.

        Expected: FAILS with buggy config (no sources)
        Expected: PASSES with working config
        """
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]

        rag_system = RAGSystem(working_config)
        rag_system.ai_generator.client = mock_client

        # Execute query
        response, sources = rag_system.query("What is computer use?")

        # Sources should match what tool tracked
        assert len(sources) > 0

        # Each source should have text and URL
        for source in sources:
            assert isinstance(source["text"], str)
            assert len(source["text"]) > 0


    @patch('ai_generator.anthropic.Anthropic')
    def test_sources_reset_after_retrieval(
        self,
        mock_anthropic_class,
        working_config,
        mock_tool_use_response,
        mock_final_response_after_tool
    ):
        """
        Test that sources are reset after each query.

        Expected: PASSES with both configs
        """
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        rag_system = RAGSystem(working_config)
        rag_system.ai_generator.client = mock_client

        # First query
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]
        response1, sources1 = rag_system.query("First query")

        # Sources should be cleared in tool manager
        current_sources = rag_system.tool_manager.get_last_sources()
        assert len(current_sources) == 0, "Sources should be reset after query"


class TestRAGSystemSessionManagement:
    """Test conversation session management."""

    @patch('ai_generator.anthropic.Anthropic')
    def test_working_system_creates_session_history(
        self,
        mock_anthropic_class,
        working_config,
        mock_tool_use_response,
        mock_final_response_after_tool
    ):
        """
        Test that session history is created and maintained.

        Expected: PASSES with both configs
        """
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]

        rag_system = RAGSystem(working_config)
        rag_system.ai_generator.client = mock_client

        # First query with session
        session_id = "test_session_1"
        response1, sources1 = rag_system.query(
            "What is computer use?",
            session_id=session_id
        )

        # Verify session created
        history = rag_system.session_manager.get_conversation_history(session_id)
        assert history is not None
        assert "What is computer use?" in history
        assert response1 in history


    @patch('ai_generator.anthropic.Anthropic')
    def test_working_system_maintains_conversation_context(
        self,
        mock_anthropic_class,
        working_config,
        mock_tool_use_response,
        mock_final_response_after_tool
    ):
        """
        Test that conversation context is passed to subsequent queries.

        Expected: PASSES with both configs
        """
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        rag_system = RAGSystem(working_config)
        rag_system.ai_generator.client = mock_client

        session_id = "test_session_2"

        # First query
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]
        response1, _ = rag_system.query("First question", session_id=session_id)

        # Second query - reset mock
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]
        response2, _ = rag_system.query("Second question", session_id=session_id)

        # Check that second query included history in system prompt
        call_args = mock_client.messages.create.call_args_list[2]
        system_content = call_args[1]["system"]

        assert "Previous conversation:" in system_content
        assert "First question" in system_content


class TestRAGSystemComparisonBuggyVsWorking:
    """Side-by-side comparison of buggy vs working configurations."""

    @patch('ai_generator.anthropic.Anthropic')
    def test_same_query_different_configs_comparison(
        self,
        mock_anthropic_class,
        buggy_config,
        working_config,
        mock_tool_use_response,
        mock_final_response_after_tool
    ):
        """
        Direct comparison: same query with buggy vs working config.

        This test clearly demonstrates the bug impact.

        Expected: Shows difference between configs
        """
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        query = "What is computer use?"

        # Test with buggy config
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]
        buggy_system = RAGSystem(buggy_config)
        buggy_system.ai_generator.client = mock_client
        buggy_response, buggy_sources = buggy_system.query(query)

        # Test with working config
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]
        working_system = RAGSystem(working_config)
        working_system.ai_generator.client = mock_client
        working_response, working_sources = working_system.query(query)

        # Compare results
        # Bug: Buggy config has no sources
        assert len(buggy_sources) == 0, "Buggy config should have no sources"

        # Fix: Working config has sources
        assert len(working_sources) > 0, "Working config should have sources"

        # Both should return responses (Claude tries regardless)
        assert len(buggy_response) > 0
        assert len(working_response) > 0

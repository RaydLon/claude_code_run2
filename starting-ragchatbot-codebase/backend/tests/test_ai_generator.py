"""
Test suite for AIGenerator tool calling functionality.

Tests the AIGenerator's integration with tools, focusing on:
- Tool use detection
- Tool execution orchestration
- Response generation after tool execution

Uses mocked Anthropic API to avoid real API calls.

Test Strategy:
- Mock API responses to simulate tool use workflow
- Test both buggy and working tool execution paths
- Verify correct message passing between Claude and tools
"""

import pytest
from unittest.mock import Mock, patch
from ai_generator import AIGenerator


class TestAIGeneratorBasicToolCalling:
    """Test basic tool calling workflow."""

    @patch('ai_generator.anthropic.Anthropic')
    def test_ai_generator_detects_tool_use(
        self,
        mock_anthropic_class,
        working_config,
        mock_tool_use_response,
        mock_final_response_after_tool,
        tool_manager_with_working_tool
    ):
        """
        Test that AIGenerator correctly detects when Claude wants to use a tool.

        Workflow:
        1. User query sent to Claude with tool definitions
        2. Claude responds with stop_reason="tool_use"
        3. AIGenerator detects this and calls _handle_tool_execution

        Expected: PASSES with both configs (detection logic doesn't depend on MAX_RESULTS)
        """
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First call: Claude decides to use tool
        # Second call: Claude synthesizes final response
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]

        # Create AI generator
        ai_gen = AIGenerator(
            api_key=working_config.ANTHROPIC_API_KEY,
            model=working_config.ANTHROPIC_MODEL
        )
        ai_gen.client = mock_client  # Use mock

        # Generate response with tools
        response = ai_gen.generate_response(
            query="What is computer use?",
            tools=tool_manager_with_working_tool.get_tool_definitions(),
            tool_manager=tool_manager_with_working_tool
        )

        # Should have made two API calls
        assert mock_client.messages.create.call_count == 2

        # Should return final synthesized response
        assert "computer" in response.lower()


    @patch('ai_generator.anthropic.Anthropic')
    def test_buggy_tool_execution_returns_empty_to_claude(
        self,
        mock_anthropic_class,
        working_config,
        mock_tool_use_response,
        mock_final_response_after_tool,
        tool_manager_with_buggy_tool
    ):
        """
        Test that buggy tool (MAX_RESULTS=0) passes "No relevant content found" to Claude.

        This demonstrates how the bug propagates: tool returns empty message,
        Claude receives it and must synthesize response from nothing.

        Expected: PASSES with buggy config (Claude gets "No relevant content found")
        """
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]

        # Create AI generator
        ai_gen = AIGenerator(
            api_key=working_config.ANTHROPIC_API_KEY,
            model=working_config.ANTHROPIC_MODEL
        )
        ai_gen.client = mock_client

        # Generate response with buggy tool
        response = ai_gen.generate_response(
            query="What is computer use?",
            tools=tool_manager_with_buggy_tool.get_tool_definitions(),
            tool_manager=tool_manager_with_buggy_tool
        )

        # Check that second API call received "No relevant content found"
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]

        # Find tool_result message
        tool_result_msg = next(
            msg for msg in messages if msg["role"] == "user" and
            any(c["type"] == "tool_result" for c in msg["content"])
        )

        # Extract tool result content
        tool_result = next(
            c for c in tool_result_msg["content"] if c["type"] == "tool_result"
        )

        # Bug: Tool result contains "No relevant content found"
        assert "No relevant content found" in tool_result["content"]


class TestAIGeneratorToolExecutionWorkflow:
    """Test the complete tool execution workflow."""

    @patch('ai_generator.anthropic.Anthropic')
    def test_working_tool_execution_passes_results_to_claude(
        self,
        mock_anthropic_class,
        working_config,
        mock_tool_use_response,
        mock_final_response_after_tool,
        tool_manager_with_working_tool
    ):
        """
        Test that working tool results are correctly passed to Claude.

        Expected: FAILS with buggy config (passes "No relevant content found")
        Expected: PASSES with working config (passes actual content)
        """
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]

        # Create AI generator
        ai_gen = AIGenerator(
            api_key=working_config.ANTHROPIC_API_KEY,
            model=working_config.ANTHROPIC_MODEL
        )
        ai_gen.client = mock_client

        # Generate response
        response = ai_gen.generate_response(
            query="What is computer use?",
            tools=tool_manager_with_working_tool.get_tool_definitions(),
            tool_manager=tool_manager_with_working_tool
        )

        # Verify second API call structure
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]

        # Should have 3 messages:
        # 1. User query
        # 2. Assistant with tool_use
        # 3. User with tool_result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Tool result should contain actual content (not error)
        tool_result = next(
            c for c in messages[2]["content"] if c["type"] == "tool_result"
        )
        assert "No relevant content found" not in tool_result["content"]
        assert len(tool_result["content"]) > 0


    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_use_id_correctly_passed_back(
        self,
        mock_anthropic_class,
        working_config,
        mock_tool_use_response,
        mock_final_response_after_tool,
        tool_manager_with_working_tool
    ):
        """
        Test that tool_use_id is correctly matched in tool_result.

        Anthropic requires tool_result to reference the tool_use_id.

        Expected: PASSES with both configs
        """
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]

        ai_gen = AIGenerator(
            api_key=working_config.ANTHROPIC_API_KEY,
            model=working_config.ANTHROPIC_MODEL
        )
        ai_gen.client = mock_client

        # Generate response
        response = ai_gen.generate_response(
            query="What is computer use?",
            tools=tool_manager_with_working_tool.get_tool_definitions(),
            tool_manager=tool_manager_with_working_tool
        )

        # Extract tool_use_id from first response
        tool_use_id = mock_tool_use_response.content[0].id

        # Extract tool_result from second call
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        tool_result = next(
            c for c in messages[2]["content"] if c["type"] == "tool_result"
        )

        # IDs should match
        assert tool_result["tool_use_id"] == tool_use_id


class TestAIGeneratorWithoutTools:
    """Test AI generator behavior when no tools are provided."""

    @patch('ai_generator.anthropic.Anthropic')
    def test_direct_response_without_tools(
        self,
        mock_anthropic_class,
        working_config,
        mock_text_response
    ):
        """
        Test that AI generator works correctly without tools.

        For general knowledge questions, Claude shouldn't use tools.

        Expected: PASSES with both configs
        """
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = mock_text_response

        ai_gen = AIGenerator(
            api_key=working_config.ANTHROPIC_API_KEY,
            model=working_config.ANTHROPIC_MODEL
        )
        ai_gen.client = mock_client

        # Generate response without tools
        response = ai_gen.generate_response(
            query="What is 2+2?",
            tools=None,
            tool_manager=None
        )

        # Should make only one API call
        assert mock_client.messages.create.call_count == 1

        # Should return text directly
        assert response == mock_text_response.content[0].text


class TestAIGeneratorConversationHistory:
    """Test that conversation history is correctly passed to Claude."""

    @patch('ai_generator.anthropic.Anthropic')
    def test_conversation_history_included_in_system_prompt(
        self,
        mock_anthropic_class,
        working_config,
        mock_text_response
    ):
        """
        Test that conversation history is added to system prompt.

        Expected: PASSES with both configs
        """
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = mock_text_response

        ai_gen = AIGenerator(
            api_key=working_config.ANTHROPIC_API_KEY,
            model=working_config.ANTHROPIC_MODEL
        )
        ai_gen.client = mock_client

        # Generate response with history
        history = "User: Previous question\nAssistant: Previous answer"
        response = ai_gen.generate_response(
            query="New question",
            conversation_history=history,
            tools=None,
            tool_manager=None
        )

        # Check that history was included in system prompt
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]

        assert "Previous conversation:" in system_content
        assert "Previous question" in system_content
        assert "Previous answer" in system_content

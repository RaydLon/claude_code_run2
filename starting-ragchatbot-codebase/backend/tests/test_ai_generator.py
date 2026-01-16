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

        # Find tool_result message (content must be a list)
        tool_result_msg = next(
            msg for msg in messages if msg["role"] == "user" and
            isinstance(msg["content"], list) and
            any(c["type"] == "tool_result" for c in msg["content"])
        )

        # Extract tool result content
        tool_result = next(
            c for c in tool_result_msg["content"] if c["type"] == "tool_result"
        )

        # Bug: Tool result contains error message about MAX_RESULTS=0
        assert ("No relevant content found" in tool_result["content"] or
                "Search error" in tool_result["content"] or
                "cannot be negative, or zero" in tool_result["content"])


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


class TestAIGeneratorSequentialToolCalling:
    """Test sequential tool calling with up to 2 rounds."""

    @patch('ai_generator.anthropic.Anthropic')
    def test_single_tool_call_completes_naturally(
        self,
        mock_anthropic_class,
        working_config,
        mock_tool_use_response,
        mock_final_response_after_tool,
        tool_manager_with_working_tool
    ):
        """
        Test backward compatibility: single tool call still works.

        Flow:
        1. User query → API call 1: Claude requests tool
        2. Tool executes → API call 2: Claude provides final answer

        Verify: 2 API calls, 1 tool execution, final response returned
        """
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # API responses: tool use, then final text
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]

        ai_gen = AIGenerator(
            api_key=working_config.ANTHROPIC_API_KEY,
            model=working_config.ANTHROPIC_MODEL
        )
        ai_gen.client = mock_client

        response = ai_gen.generate_response(
            query="What is computer use?",
            tools=tool_manager_with_working_tool.get_tool_definitions(),
            tool_manager=tool_manager_with_working_tool
        )

        # Verify behavior
        assert mock_client.messages.create.call_count == 2
        assert "computer" in response.lower()

    @patch('ai_generator.anthropic.Anthropic')
    def test_two_sequential_tool_calls(
        self,
        mock_anthropic_class,
        working_config,
        tool_manager_with_working_tool
    ):
        """
        Test two sequential tool calls in separate rounds.

        Flow:
        1. API call 1: Claude requests first tool (get_course_outline)
        2. First tool executes
        3. API call 2: Claude reviews results, requests second tool (search_course_content)
        4. Second tool executes
        5. API call 3: Claude synthesizes final answer

        Verify: 3 API calls, 2 tool executions, correct message history
        """
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create mock responses for 2-round flow
        first_tool_response = Mock()
        first_tool_response.stop_reason = "tool_use"
        first_tool_block = Mock()
        first_tool_block.type = "tool_use"
        first_tool_block.id = "toolu_01FIRST"
        first_tool_block.name = "search_course_content"
        first_tool_block.input = {"query": "lesson 1"}
        first_tool_response.content = [first_tool_block]

        second_tool_response = Mock()
        second_tool_response.stop_reason = "tool_use"
        second_tool_block = Mock()
        second_tool_block.type = "tool_use"
        second_tool_block.id = "toolu_02SECOND"
        second_tool_block.name = "search_course_content"
        second_tool_block.input = {"query": "lesson 2"}
        second_tool_response.content = [second_tool_block]

        final_text_response = Mock()
        final_text_response.stop_reason = "end_turn"
        final_text_block = Mock()
        final_text_block.type = "text"
        final_text_block.text = "Comparison: Lesson 1 covers X, Lesson 2 covers Y"
        final_text_response.content = [final_text_block]

        # API call sequence: tool1 → tool2 → final
        mock_client.messages.create.side_effect = [
            first_tool_response,
            second_tool_response,
            final_text_response
        ]

        ai_gen = AIGenerator(
            api_key=working_config.ANTHROPIC_API_KEY,
            model=working_config.ANTHROPIC_MODEL
        )
        ai_gen.client = mock_client

        response = ai_gen.generate_response(
            query="Compare lesson 1 and lesson 2",
            tools=tool_manager_with_working_tool.get_tool_definitions(),
            tool_manager=tool_manager_with_working_tool
        )

        # Verify 3 API calls made (2 tool rounds + 1 final)
        assert mock_client.messages.create.call_count == 3

        # Verify final response
        assert "Comparison" in response

        # Verify that tools were available for first 2 calls
        first_call_kwargs = mock_client.messages.create.call_args_list[0][1]
        assert "tools" in first_call_kwargs

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs

        # Third call should NOT have tools (forced final response)
        third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in third_call_kwargs

    @patch('ai_generator.anthropic.Anthropic')
    def test_max_rounds_enforced_after_two_tool_calls(
        self,
        mock_anthropic_class,
        working_config,
        tool_manager_with_working_tool
    ):
        """
        Test that after 2 rounds, tools are removed to force final response.

        Simulates Claude wanting to use tools indefinitely.
        After 2 rounds, should make final call WITHOUT tools.

        Verify: Tools parameter is absent in 3rd API call
        """
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create mock that always wants to use tools
        infinite_tool_response = Mock()
        infinite_tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_INFINITE"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "something"}
        infinite_tool_response.content = [tool_block]

        final_response = Mock()
        final_response.stop_reason = "end_turn"
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Final answer"
        final_response.content = [text_block]

        # Always return tool_use for first 2 calls, then final
        mock_client.messages.create.side_effect = [
            infinite_tool_response,  # Round 1
            infinite_tool_response,  # Round 2
            final_response           # Forced final (no tools)
        ]

        ai_gen = AIGenerator(
            api_key=working_config.ANTHROPIC_API_KEY,
            model=working_config.ANTHROPIC_MODEL
        )
        ai_gen.client = mock_client

        response = ai_gen.generate_response(
            query="Test infinite loop prevention",
            tools=tool_manager_with_working_tool.get_tool_definitions(),
            tool_manager=tool_manager_with_working_tool
        )

        # Should make exactly 3 API calls
        assert mock_client.messages.create.call_count == 3

        # First 2 calls should have tools
        first_call_kwargs = mock_client.messages.create.call_args_list[0][1]
        assert "tools" in first_call_kwargs

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs

        # Third call should NOT have tools (forced final)
        third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in third_call_kwargs

        assert response == "Final answer"

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_error_handled_gracefully(
        self,
        mock_anthropic_class,
        working_config
    ):
        """
        Test that tool execution errors are passed to Claude as error results.

        Flow:
        1. Claude requests tool
        2. Tool execution raises exception
        3. Error formatted as tool_result with is_error=True
        4. Claude receives error and provides response

        Verify: Error caught, passed to Claude, response generated
        """
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_use_response = Mock()
        tool_use_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_ERROR"
        tool_block.name = "failing_tool"
        tool_block.input = {}
        tool_use_response.content = [tool_block]

        final_response = Mock()
        final_response.stop_reason = "end_turn"
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "I encountered an error retrieving that information"
        final_response.content = [text_block]

        mock_client.messages.create.side_effect = [
            tool_use_response,
            final_response
        ]

        # Create tool manager that raises exception
        failing_tool_manager = Mock()
        failing_tool_manager.get_tool_definitions.return_value = []
        failing_tool_manager.execute_tool.side_effect = Exception("Database connection failed")

        ai_gen = AIGenerator(
            api_key=working_config.ANTHROPIC_API_KEY,
            model=working_config.ANTHROPIC_MODEL
        )
        ai_gen.client = mock_client

        response = ai_gen.generate_response(
            query="Test error handling",
            tools=[],
            tool_manager=failing_tool_manager
        )

        # Should still get a response
        assert "error" in response.lower()

        # Verify second API call received error in tool_result
        second_call_messages = mock_client.messages.create.call_args_list[1][1]["messages"]
        tool_result_message = second_call_messages[-1]
        tool_result_content = tool_result_message["content"][0]

        assert tool_result_content["type"] == "tool_result"
        assert "Error executing tool" in tool_result_content["content"]
        assert tool_result_content.get("is_error") == True

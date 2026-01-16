import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for searching course information and retrieving course outlines.

Tool Usage:
- **search_course_content**: Use for questions about specific course content or detailed educational materials
- **get_course_outline**: Use for questions about course structure, outlines, or lesson listings
- **Sequential tool calling**: You can make up to 2 tool calls total to gather information
  - Use the first tool call to get initial information
  - After reviewing results, decide if you need one more tool call
  - Use the second tool call if you need additional or related information
  - After gathering information, provide a complete synthesized answer
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Multi-step query examples:
- To compare content from two lessons: first get one lesson, then get the other
- To find a course discussing the same topic as a specific lesson: first get that lesson's content/outline, then search for courses with similar topics
- After each tool result, evaluate: do I have enough information to answer completely?

When answering outline-related queries:
- Always include the course title
- Always include the course link
- List all lessons with their lesson number and lesson title
- Format clearly and concisely

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use search_course_content tool first, then answer
- **Course outline questions**: Use get_course_outline tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool usage explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the tool results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: str | None = None,
        tools: list | None = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional multi-round tool usage.

        Supports up to 2 sequential tool calling rounds where Claude can:
        1. Make a tool call and review results
        2. Optionally make a second tool call based on first results
        3. Synthesize final answer from gathered information

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        MAX_ROUNDS = 2

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize messages for the conversation
        messages = [{"role": "user", "content": query}]

        # Agentic loop: allow up to MAX_ROUNDS of tool calling
        for round_num in range(MAX_ROUNDS):
            # Prepare API call with tools available
            api_params = {**self.base_params, "messages": messages, "system": system_content}

            # Add tools if provided
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # Check if Claude wants to use tools
            if response.stop_reason != "tool_use":
                # Claude provided final text response - we're done
                return self._extract_text_content(response)

            # Claude wants to use tools - check if tool_manager available
            if not tool_manager:
                # No tool manager available, return what we can
                return self._extract_text_content(response)

            # Execute tools and collect results
            tool_results = self._execute_tools_for_round(response, tool_manager)

            # Append assistant's tool use to messages
            messages.append({"role": "assistant", "content": response.content})

            # Append tool results to messages
            messages.append({"role": "user", "content": tool_results})

            # Continue to next round

        # Reached MAX_ROUNDS - force final response without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
            # No tools - Claude must synthesize from available info
        }

        final_response = self.client.messages.create(**final_params)
        return self._extract_text_content(final_response)

    def _execute_tools_for_round(self, response, tool_manager) -> list[dict]:
        """
        Execute all tool calls from a response and return formatted results.

        Args:
            response: Claude API response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            List of tool_result dictionaries ready for API
        """
        tool_results = []

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    # Execute tool
                    result = tool_manager.execute_tool(content_block.name, **content_block.input)

                    # Format result for API
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": content_block.id, "content": result}
                    )

                except Exception as e:
                    # Handle tool execution errors gracefully
                    # Pass error to Claude so it can work with partial info
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Error executing tool: {str(e)}",
                            "is_error": True,
                        }
                    )

        return tool_results

    def _extract_text_content(self, response) -> str:
        """
        Extract text content from Claude API response.

        Args:
            response: Claude API response

        Returns:
            Text content from response, or empty string if none found
        """
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

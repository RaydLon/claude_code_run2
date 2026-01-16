"""
Test suite for CourseSearchTool functionality.

Tests the tool interface that Claude uses for searching course content.
Validates search execution, result formatting, and source tracking.

Test Strategy:
- Tests demonstrate buggy tool returns "No relevant content found"
- Tests verify working tool returns formatted results with sources
"""


class TestCourseSearchToolBasicExecution:
    """Test basic tool execution with different configs."""

    def test_buggy_tool_returns_no_content_message(self, buggy_course_search_tool, sample_queries):
        """
        Bug: Tool returns "No relevant content found" due to MAX_RESULTS=0.

        Expected: PASSES with current config (returns "No relevant content found")
        """
        query = sample_queries["general_content"]
        result = buggy_course_search_tool.execute(query=query)

        # Bug causes tool to return empty message
        assert (
            "No relevant content found" in result
        ), "Buggy tool should return 'No relevant content found'"

        # No sources tracked
        assert len(buggy_course_search_tool.last_sources) == 0

    def test_working_tool_returns_formatted_results(
        self, working_course_search_tool, sample_queries
    ):
        """
        Fix: Tool returns properly formatted results with course context.

        Expected: FAILS with buggy config (gets "No relevant content found")
        Expected: PASSES with working config
        """
        query = sample_queries["general_content"]
        result = working_course_search_tool.execute(query=query)

        # Should return actual content, not error message
        assert "No relevant content found" not in result
        assert len(result) > 0, "Should return non-empty result"

        # Should contain course context headers
        assert "[" in result, "Should have course context headers like [Course Title - Lesson N]"
        assert "]" in result

        # Sources should be tracked
        assert len(working_course_search_tool.last_sources) > 0

        # Validate source structure
        for source in working_course_search_tool.last_sources:
            assert "text" in source
            assert "url" in source
            assert isinstance(source["text"], str)


class TestCourseSearchToolWithFilters:
    """Test tool execution with course and lesson filters."""

    def test_working_tool_with_course_name_filter(
        self, working_course_search_tool, expected_courses
    ):
        """
        Test tool filtering by course name.

        Expected: FAILS with buggy config
        Expected: PASSES with working config
        """
        course_title = expected_courses["building_computer_use"]["title"]
        result = working_course_search_tool.execute(query="computer use", course_name=course_title)

        assert "No relevant content found" not in result
        assert course_title in result, "Result should mention the course title"

        # All sources should be from this course
        for source in working_course_search_tool.last_sources:
            assert course_title in source["text"]

    def test_working_tool_with_lesson_number_filter(
        self, working_course_search_tool, expected_courses
    ):
        """
        Test tool filtering by lesson number.

        Expected: FAILS with buggy config
        Expected: PASSES with working config
        """
        course_title = expected_courses["building_computer_use"]["title"]
        result = working_course_search_tool.execute(
            query="introduction", course_name=course_title, lesson_number=0
        )

        assert "No relevant content found" not in result
        assert "Lesson 0" in result, "Result should mention Lesson 0"

        # All sources should be from lesson 0
        for source in working_course_search_tool.last_sources:
            assert "Lesson 0" in source["text"]

    def test_working_tool_with_fuzzy_course_name(
        self, working_course_search_tool, course_name_variations
    ):
        """
        Test tool handles fuzzy course name matching.

        Expected: FAILS with buggy config
        Expected: PASSES with working config
        """
        partial_name = "computer use"
        expected_title = course_name_variations[partial_name]

        result = working_course_search_tool.execute(query="introduction", course_name=partial_name)

        assert "No relevant content found" not in result
        # Should resolve to full course title
        assert expected_title in result

    def test_tool_with_nonexistent_course_returns_error(self, working_course_search_tool):
        """
        Test tool returns appropriate error for non-existent course.

        Expected: PASSES with both configs
        """
        result = working_course_search_tool.execute(
            query="test", course_name="Nonexistent Course XYZ123"
        )

        assert "No course found" in result, "Should return course not found error"
        assert len(working_course_search_tool.last_sources) == 0


class TestCourseSearchToolResultFormatting:
    """Test the formatting of tool results for Claude."""

    def test_working_tool_formats_with_course_context(self, working_course_search_tool):
        """
        Test that results include proper course and lesson context headers.

        Format should be:
        [Course Title - Lesson N]
        content text

        Expected: FAILS with buggy config
        Expected: PASSES with working config
        """
        result = working_course_search_tool.execute(query="computer use")

        # Check for header format
        assert "[" in result and "]" in result

        # Headers should be on separate lines from content
        lines = result.split("\n")
        header_lines = [line for line in lines if line.startswith("[")]
        assert len(header_lines) > 0, "Should have at least one context header"

        # Each header should have course title
        for header in header_lines:
            assert header.startswith("[") and "]" in header

    def test_working_tool_separates_multiple_results(self, working_course_search_tool):
        """
        Test that multiple search results are properly separated.

        Results should be separated by double newlines.

        Expected: FAILS with buggy config (no results to separate)
        Expected: PASSES with working config
        """
        result = working_course_search_tool.execute(query="computer use")

        # Should have multiple results separated by \n\n
        result_blocks = result.split("\n\n")
        assert len(result_blocks) > 1, "Should have multiple result blocks"


class TestCourseSearchToolSourceTracking:
    """Test that tool properly tracks sources for UI display."""

    def test_working_tool_tracks_sources(self, working_course_search_tool):
        """
        Test that last_sources attribute is populated correctly.

        Expected: FAILS with buggy config (no sources)
        Expected: PASSES with working config
        """
        result = working_course_search_tool.execute(query="computer use")

        sources = working_course_search_tool.last_sources
        assert len(sources) > 0, "Should track sources"

        # Validate source structure
        for source in sources:
            assert isinstance(source, dict)
            assert "text" in source
            assert "url" in source

            # Text should mention course and optionally lesson
            assert len(source["text"]) > 0

            # URL should be None or valid string
            assert source["url"] is None or isinstance(source["url"], str)

    def test_working_tool_includes_lesson_links_in_sources(
        self, working_course_search_tool, expected_courses
    ):
        """
        Test that sources include lesson links when available.

        Expected: FAILS with buggy config
        Expected: PASSES with working config
        """
        course_title = expected_courses["building_computer_use"]["title"]
        result = working_course_search_tool.execute(
            query="introduction", course_name=course_title, lesson_number=0
        )

        sources = working_course_search_tool.last_sources
        assert len(sources) > 0

        # At least one source should have a URL (lesson link)
        has_url = any(source["url"] is not None for source in sources)
        assert has_url, "Should have lesson links in sources"


class TestCourseSearchToolDefinition:
    """Test tool definition for Anthropic tool calling."""

    def test_tool_definition_structure(self, working_course_search_tool):
        """
        Test that tool provides correct definition for Claude.

        Expected: PASSES with both configs
        """
        definition = working_course_search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

        # Validate input schema
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema

        # Required parameters
        assert "query" in schema["properties"]
        assert "query" in schema["required"]

        # Optional parameters
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        assert "course_name" not in schema["required"]
        assert "lesson_number" not in schema["required"]


class TestCourseSearchToolErrorHandling:
    """Test tool error handling and edge cases."""

    def test_working_tool_handles_empty_query(self, working_course_search_tool):
        """
        Test tool behavior with empty query string.

        Expected: PASSES with both configs (should handle gracefully)
        """
        result = working_course_search_tool.execute(query="")

        # Should return something (either results or no content message)
        assert isinstance(result, str)
        assert len(result) > 0

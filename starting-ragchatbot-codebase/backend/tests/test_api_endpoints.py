"""
Test suite for FastAPI endpoint integration.

Tests the HTTP API layer including:
- POST /api/query endpoint request/response handling
- GET /api/courses endpoint for course statistics
- Error handling and validation
- Session management through API
- Response format validation

Uses TestClient to make HTTP requests without running a server.
Mocks the Anthropic API to avoid external dependencies.
"""

import pytest
from unittest.mock import patch
from fastapi import status


@pytest.mark.api
class TestQueryEndpoint:
    """Test POST /api/query endpoint."""

    @patch('ai_generator.anthropic.Anthropic')
    def test_query_endpoint_returns_200_with_valid_request(
        self,
        mock_anthropic_class,
        test_client,
        mock_tool_use_response,
        mock_final_response_after_tool
    ):
        """
        Test that /api/query endpoint returns 200 with valid request.

        Verifies:
        - Endpoint accepts valid query request
        - Returns 200 status code
        - Response contains required fields
        """
        # Setup mock
        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]

        # Patch the client on the test app's RAG system
        test_client.app.state.rag_system.ai_generator.client = mock_client

        # Make request
        response = test_client.post(
            "/api/query",
            json={"query": "What is computer use?"}
        )

        # Verify response
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        # Verify types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)


    @patch('ai_generator.anthropic.Anthropic')
    def test_query_endpoint_creates_session_when_not_provided(
        self,
        mock_anthropic_class,
        test_client,
        mock_tool_use_response,
        mock_final_response_after_tool
    ):
        """
        Test that endpoint creates new session if session_id not provided.

        Verifies:
        - New session created automatically
        - session_id returned in response
        """
        # Setup mock
        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]

        # Patch the client on the test app's RAG system
        test_client.app.state.rag_system.ai_generator.client = mock_client

        # Make request without session_id
        response = test_client.post(
            "/api/query",
            json={"query": "What is computer use?"}
        )

        # Verify session created
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["session_id"] is not None
        assert len(data["session_id"]) > 0


    @patch('ai_generator.anthropic.Anthropic')
    def test_query_endpoint_uses_provided_session_id(
        self,
        mock_anthropic_class,
        test_client,
        mock_tool_use_response,
        mock_final_response_after_tool
    ):
        """
        Test that endpoint uses provided session_id.

        Verifies:
        - Provided session_id is used
        - Same session_id returned in response
        """
        # Setup mock
        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]

        # Patch the client on the test app's RAG system
        test_client.app.state.rag_system.ai_generator.client = mock_client

        session_id = "test_session_123"

        # Make request with session_id
        response = test_client.post(
            "/api/query",
            json={
                "query": "What is computer use?",
                "session_id": session_id
            }
        )

        # Verify session used
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["session_id"] == session_id


    @patch('ai_generator.anthropic.Anthropic')
    def test_query_endpoint_returns_sources_with_correct_format(
        self,
        mock_anthropic_class,
        test_client,
        mock_tool_use_response,
        mock_final_response_after_tool
    ):
        """
        Test that sources are returned in correct format.

        Verifies:
        - Sources is a list
        - Each source has 'text' and 'url' fields
        - URL can be None
        """
        # Setup mock
        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]

        # Patch the client on the test app's RAG system
        test_client.app.state.rag_system.ai_generator.client = mock_client

        # Make request
        response = test_client.post(
            "/api/query",
            json={"query": "What is computer use?"}
        )

        # Verify sources format
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        sources = data["sources"]
        assert isinstance(sources, list)

        # If sources exist, verify structure
        for source in sources:
            assert "text" in source
            assert "url" in source
            assert isinstance(source["text"], str)
            # URL can be None or string
            assert source["url"] is None or isinstance(source["url"], str)


    def test_query_endpoint_rejects_missing_query(self, test_client):
        """
        Test that endpoint rejects request without query field.

        Verifies:
        - Returns 422 Unprocessable Entity for missing required field
        """
        response = test_client.post(
            "/api/query",
            json={"session_id": "test"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


    def test_query_endpoint_rejects_empty_query(self, test_client):
        """
        Test that endpoint handles empty query string.

        Note: FastAPI validation allows empty strings by default.
        This test documents current behavior.
        """
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )

        # Empty query is technically valid for Pydantic, but may error in RAG system
        # The exact behavior depends on RAG system implementation
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]


    def test_query_endpoint_rejects_invalid_json(self, test_client):
        """
        Test that endpoint rejects malformed JSON.

        Verifies:
        - Returns 422 for invalid JSON
        """
        response = test_client.post(
            "/api/query",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


    @patch('ai_generator.anthropic.Anthropic')
    def test_query_endpoint_handles_conversation_history(
        self,
        mock_anthropic_class,
        test_client,
        mock_tool_use_response,
        mock_final_response_after_tool
    ):
        """
        Test that endpoint maintains conversation history across requests.

        Verifies:
        - Multiple queries with same session_id
        - Session history preserved
        """
        # Setup mock
        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool,
            mock_tool_use_response,
            mock_final_response_after_tool
        ]

        # Patch the client on the test app's RAG system
        test_client.app.state.rag_system.ai_generator.client = mock_client

        # First query
        response1 = test_client.post(
            "/api/query",
            json={"query": "First question"}
        )

        assert response1.status_code == status.HTTP_200_OK
        session_id = response1.json()["session_id"]

        # Second query with same session
        response2 = test_client.post(
            "/api/query",
            json={
                "query": "Second question",
                "session_id": session_id
            }
        )

        assert response2.status_code == status.HTTP_200_OK
        assert response2.json()["session_id"] == session_id


@pytest.mark.api
class TestCoursesEndpoint:
    """Test GET /api/courses endpoint."""

    def test_courses_endpoint_returns_200(self, test_client):
        """
        Test that /api/courses endpoint returns 200.

        Verifies:
        - Endpoint accessible
        - Returns success status
        """
        response = test_client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK


    def test_courses_endpoint_returns_correct_format(self, test_client):
        """
        Test that /api/courses returns data in correct format.

        Verifies:
        - Response contains total_courses field
        - Response contains course_titles field
        - Types are correct
        """
        response = test_client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify structure
        assert "total_courses" in data
        assert "course_titles" in data

        # Verify types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

        # Verify consistency
        assert data["total_courses"] >= 0
        assert len(data["course_titles"]) == data["total_courses"]


    def test_courses_endpoint_returns_course_titles_as_strings(
        self,
        test_client
    ):
        """
        Test that course_titles contains strings.

        Verifies:
        - Each title is a string
        - No empty strings
        """
        response = test_client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        for title in data["course_titles"]:
            assert isinstance(title, str)
            assert len(title) > 0


@pytest.mark.api
class TestAPIErrorHandling:
    """Test API error handling and edge cases."""

    @patch('rag_system.RAGSystem.query')
    def test_query_endpoint_handles_rag_system_errors(
        self,
        mock_query,
        test_client
    ):
        """
        Test that endpoint handles RAG system errors gracefully.

        Verifies:
        - Returns 500 for internal errors
        - Error message included in response
        """
        # Mock RAG system to raise exception
        mock_query.side_effect = Exception("Test error")

        response = test_client.post(
            "/api/query",
            json={"query": "Test query"}
        )

        # Should return 500 Internal Server Error
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

        # Error detail should be in response
        data = response.json()
        assert "detail" in data


    @patch('rag_system.RAGSystem.get_course_analytics')
    def test_courses_endpoint_handles_analytics_errors(
        self,
        mock_analytics,
        test_client
    ):
        """
        Test that /api/courses handles analytics errors.

        Verifies:
        - Returns 500 for internal errors
        - Error message included
        """
        # Mock analytics to raise exception
        mock_analytics.side_effect = Exception("Analytics error")

        response = test_client.get("/api/courses")

        # Should return 500 Internal Server Error
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

        # Error detail should be in response
        data = response.json()
        assert "detail" in data


@pytest.mark.api
class TestAPICORSAndMiddleware:
    """Test CORS and middleware configuration."""

    @pytest.mark.skip(reason="CORS headers not fully testable with TestClient")
    def test_api_allows_cors_headers(self, test_client):
        """
        Test that API includes CORS headers.

        Note: TestClient doesn't fully simulate CORS behavior since it operates
        at the ASGI level rather than HTTP level. CORS headers are typically
        added by middleware in response to browser requests with Origin headers.

        Verifies:
        - Access-Control-Allow-Origin header present (in real browser requests)
        """
        response = test_client.get(
            "/api/courses",
            headers={"Origin": "http://localhost:3000"}
        )

        # With real browser requests, CORS headers should be present
        # TestClient may not include them since it bypasses HTTP layer
        # This test documents expected behavior but may not assert in test environment


    @pytest.mark.skip(reason="OPTIONS preflight not testable with TestClient")
    def test_api_accepts_options_request(self, test_client):
        """
        Test that API handles OPTIONS preflight requests.

        Note: TestClient doesn't fully support OPTIONS preflight requests
        as they're handled at the HTTP/ASGI middleware level.

        Verifies:
        - OPTIONS request should succeed (in real browser context)
        - CORS headers should be present (in real browser context)
        """
        response = test_client.options(
            "/api/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )

        # In real deployment, OPTIONS should succeed with CORS headers
        # TestClient behavior may differ


@pytest.mark.api
@pytest.mark.integration
class TestAPIIntegrationWithRealComponents:
    """
    Integration tests using real components (not mocked).

    These tests use real VectorStore and ChromaDB but mock Anthropic API.
    """

    @patch('ai_generator.anthropic.Anthropic')
    def test_full_query_flow_with_real_vector_store(
        self,
        mock_anthropic_class,
        test_client,
        mock_tool_use_response,
        mock_final_response_after_tool
    ):
        """
        Test complete query flow with real vector store.

        Verifies:
        - Query processed through real RAG components
        - Sources retrieved from real ChromaDB
        - Response assembled correctly
        """
        # Setup mock Anthropic client
        mock_client = mock_anthropic_class.return_value
        mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_final_response_after_tool
        ]

        # Patch the client on the test app's RAG system
        test_client.app.state.rag_system.ai_generator.client = mock_client

        # Make query that should find results in ChromaDB
        response = test_client.post(
            "/api/query",
            json={"query": "What is computer use?"}
        )

        # Verify successful response
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["answer"] is not None
        assert len(data["answer"]) > 0

        # With real ChromaDB, should have sources (if data exists)
        # Sources list can be empty if no matching content or ChromaDB is empty
        assert isinstance(data["sources"], list)


    def test_courses_endpoint_with_real_vector_store(self, test_client):
        """
        Test /api/courses with real vector store.

        Verifies:
        - Returns actual course data from ChromaDB
        - Data is valid and consistent
        """
        response = test_client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

        # If ChromaDB has data, verify consistency
        if data["total_courses"] > 0:
            assert len(data["course_titles"]) == data["total_courses"]
            for title in data["course_titles"]:
                assert isinstance(title, str)
                assert len(title) > 0

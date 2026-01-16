"""
Test suite for VectorStore.search() functionality.

Tests the core search behavior with real ChromaDB data to demonstrate
the MAX_RESULTS=0 bug and verify the fix.

Test Strategy:
- Tests FAIL with buggy_vector_store (MAX_RESULTS=0)
- Same tests PASS with working_vector_store (MAX_RESULTS=5)
"""

import pytest
from vector_store import VectorStore, SearchResults


class TestVectorStoreBasicSearch:
    """Test basic search functionality with different MAX_RESULTS values."""

    def test_buggy_search_returns_empty_results(self, buggy_vector_store, sample_queries):
        """
        DEMONSTRATES THE BUG: Search with MAX_RESULTS=0 returns no results.

        This test proves that when MAX_RESULTS=0, even valid queries that
        should match course content return empty results.

        Expected: PASSES with current config (MAX_RESULTS=0)
        """
        query = sample_queries["general_content"]
        results = buggy_vector_store.search(query)

        # THE BUG: No results returned even though content exists
        assert results.is_empty(), (
            "Expected empty results with MAX_RESULTS=0, "
            f"but got {len(results.documents)} results"
        )
        assert len(results.documents) == 0
        assert len(results.metadata) == 0


    def test_working_search_returns_results(self, working_vector_store, sample_queries):
        """
        DEMONSTRATES THE FIX: Search with MAX_RESULTS=5 returns results.

        This test proves that with MAX_RESULTS=5, the same query returns
        relevant course content.

        Expected: FAILS with current config (test will fail because we get 0 results)
        Expected: PASSES after fix (MAX_RESULTS=5 returns results)
        """
        query = sample_queries["general_content"]
        results = working_vector_store.search(query)

        # After fix: Should return results
        assert not results.is_empty(), (
            "Expected results with MAX_RESULTS=5, but got empty results"
        )
        assert len(results.documents) > 0, "Should return at least one result"
        assert len(results.documents) <= 5, "Should not exceed MAX_RESULTS"
        assert len(results.documents) == len(results.metadata)
        assert len(results.documents) == len(results.distances)


    @pytest.mark.parametrize("query_key", [
        "general_content",
        "specific_topic",
        "technical_term",
        "course_specific",
        "lesson_specific"
    ])
    def test_multiple_queries_with_working_config(
        self,
        working_vector_store,
        sample_queries,
        query_key
    ):
        """
        Test that various query types return results with working config.

        Parametrized test ensuring different query patterns all work
        correctly once MAX_RESULTS is fixed.

        Expected: FAILS with buggy config (all return 0 results)
        Expected: PASSES with working config (all return >0 results)
        """
        query = sample_queries[query_key]
        results = working_vector_store.search(query)

        assert not results.is_empty(), (
            f"Query '{query}' should return results with MAX_RESULTS=5"
        )
        assert len(results.documents) > 0

        # Validate result structure
        for doc, meta in zip(results.documents, results.metadata):
            assert isinstance(doc, str)
            assert len(doc) > 0
            assert "course_title" in meta
            assert "lesson_number" in meta
            assert "chunk_index" in meta


class TestVectorStoreWithFilters:
    """Test search with course name and lesson number filters."""

    def test_buggy_search_with_course_filter_returns_empty(
        self,
        buggy_vector_store,
        expected_courses
    ):
        """
        Bug demonstration: Course-filtered search returns empty with MAX_RESULTS=0.

        Expected: PASSES with current config (returns 0 results)
        """
        course_title = expected_courses["building_computer_use"]["title"]
        results = buggy_vector_store.search(
            query="computer use",
            course_name=course_title
        )

        assert results.is_empty(), "Expected empty results with MAX_RESULTS=0"


    def test_working_search_with_course_filter_returns_results(
        self,
        working_vector_store,
        expected_courses
    ):
        """
        Fix demonstration: Course-filtered search returns results with MAX_RESULTS=5.

        Expected: FAILS with current config
        Expected: PASSES after fix
        """
        course_title = expected_courses["building_computer_use"]["title"]
        results = working_vector_store.search(
            query="computer use",
            course_name=course_title
        )

        assert not results.is_empty(), "Should return results with MAX_RESULTS=5"

        # Verify all results are from the correct course
        for meta in results.metadata:
            assert meta["course_title"] == course_title


    def test_working_search_with_lesson_filter(
        self,
        working_vector_store,
        expected_courses
    ):
        """
        Test search filtered by lesson number with working config.

        Expected: FAILS with buggy config (returns 0)
        Expected: PASSES with working config
        """
        course_title = expected_courses["building_computer_use"]["title"]
        results = working_vector_store.search(
            query="introduction",
            course_name=course_title,
            lesson_number=0
        )

        assert not results.is_empty(), "Should find lesson 0 content"

        # Verify all results are from lesson 0
        for meta in results.metadata:
            assert meta["course_title"] == course_title
            assert meta["lesson_number"] == 0


    def test_working_search_with_nonexistent_course_returns_error(
        self,
        working_vector_store
    ):
        """
        Test that searching for non-existent course returns appropriate error.

        This should work regardless of MAX_RESULTS value.

        Expected: PASSES with both configs
        """
        results = working_vector_store.search(
            query="test query",
            course_name="Nonexistent Course Title That Does Not Exist"
        )

        assert results.error is not None, "Should return error for missing course"
        assert "No course found" in results.error


class TestVectorStoreSearchLimit:
    """Test that search respects limit parameter and MAX_RESULTS setting."""

    def test_buggy_config_ignores_explicit_limit(self, buggy_vector_store):
        """
        Bug: Even with explicit limit parameter, MAX_RESULTS=0 returns nothing.

        Expected: PASSES with current config
        """
        # Try to override with explicit limit
        results = buggy_vector_store.search(
            query="computer use",
            limit=3
        )

        # Bug: Still returns 0 because max_results is used as fallback
        assert results.is_empty(), "Bug causes empty results despite explicit limit"


    def test_working_config_respects_explicit_limit(self, working_vector_store):
        """
        Test that explicit limit parameter works correctly.

        Expected: FAILS with buggy config
        Expected: PASSES with working config
        """
        results = working_vector_store.search(
            query="computer use",
            limit=3
        )

        assert not results.is_empty(), "Should return results"
        assert len(results.documents) <= 3, "Should respect explicit limit of 3"
        assert len(results.documents) > 0, "Should return at least one result"


    def test_working_config_uses_max_results_as_default(self, working_vector_store):
        """
        Test that MAX_RESULTS is used when no explicit limit provided.

        Expected: FAILS with buggy config (returns 0)
        Expected: PASSES with working config (returns up to 5)
        """
        results = working_vector_store.search(query="computer use")

        assert not results.is_empty(), "Should return results"
        assert len(results.documents) <= 5, "Should not exceed MAX_RESULTS=5"


class TestVectorStoreCourseNameResolution:
    """Test semantic course name matching functionality."""

    def test_fuzzy_course_name_matching_works_with_working_config(
        self,
        working_vector_store,
        course_name_variations
    ):
        """
        Test that fuzzy course name matching works with various inputs.

        The course_catalog collection uses semantic search to match
        partial or fuzzy course names to exact titles.

        Expected: FAILS with buggy config (returns 0 results after resolution)
        Expected: PASSES with working config
        """
        for partial_name, expected_title in course_name_variations.items():
            results = working_vector_store.search(
                query="introduction",
                course_name=partial_name
            )

            # Should resolve partial name and return results
            assert not results.is_empty(), (
                f"Should resolve '{partial_name}' to '{expected_title}'"
            )

            # Verify results are from expected course
            for meta in results.metadata:
                assert meta["course_title"] == expected_title


class TestVectorStoreResultStructure:
    """Test the structure and content of SearchResults objects."""

    def test_search_results_structure_with_working_config(self, working_vector_store):
        """
        Validate that SearchResults objects have correct structure.

        Expected: FAILS with buggy config (empty results)
        Expected: PASSES with working config
        """
        results = working_vector_store.search("computer use")

        assert isinstance(results, SearchResults)
        assert isinstance(results.documents, list)
        assert isinstance(results.metadata, list)
        assert isinstance(results.distances, list)

        # All lists should have same length
        assert len(results.documents) == len(results.metadata)
        assert len(results.documents) == len(results.distances)

        # Validate metadata structure
        for meta in results.metadata:
            assert "course_title" in meta
            assert "lesson_number" in meta
            assert "chunk_index" in meta
            assert isinstance(meta["course_title"], str)
            assert isinstance(meta["chunk_index"], int)


    def test_empty_results_have_no_error(self, working_vector_store):
        """
        Test that genuinely empty results (no matches) have no error.

        Expected: PASSES with both configs
        """
        # Search for something that definitely won't match
        results = working_vector_store.search(
            "xyzabc123nonexistentquery999"
        )

        # May be empty but shouldn't have error (just no matches)
        if results.is_empty():
            assert results.error is None, "Empty due to no matches, not error"

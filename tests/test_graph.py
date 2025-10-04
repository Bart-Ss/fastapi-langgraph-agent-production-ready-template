import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Mock modules that cause side-effects on import
mock_postgres_saver = MagicMock()
mock_psycopg_pool = MagicMock()
mock_db_service = MagicMock()

# Mock other dependencies to isolate the test
mock_tools = MagicMock()
mock_utils = MagicMock()
mock_prompts = MagicMock()
mock_settings = MagicMock()
mock_settings.LLM_MODEL = "test-model"
mock_settings.DEFAULT_LLM_TEMPERATURE = 0.0
mock_settings.LLM_API_KEY = "test-key"
mock_settings.MAX_TOKENS = 100
mock_settings.ENVIRONMENT.value = "test"
mock_settings.LOG_LEVEL = "INFO"
mock_settings.LANGFUSE_PUBLIC_KEY = "test-public-key"


with patch.dict('sys.modules', {
    'langgraph.checkpoint.postgres.aio': MagicMock(AsyncPostgresSaver=mock_postgres_saver),
    'psycopg_pool': mock_psycopg_pool,
    'app.services.database': MagicMock(database_service=mock_db_service),
    'app.core.langgraph.tools': mock_tools,
    'app.utils': mock_utils,
    'app.core.prompts': mock_prompts,
    'app.core.config': MagicMock(settings=mock_settings),
}):
    from app.core.langgraph.graph import LangGraphAgent
    from app.schemas import Message

from langchain_core.messages import AIMessage

@pytest.fixture
def agent():
    """Provides a LangGraphAgent instance with a mocked graph."""
    agent = LangGraphAgent()
    # Use a standard MagicMock for the graph, and an AsyncMock for the awaited method
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock()
    agent._graph = mock_graph
    return agent

@pytest.mark.asyncio
async def test_get_response_succeeds(agent):
    """
    This test verifies that the fix in get_response is correct by ensuring
    it processes the response correctly without raising a ValueError.
    """
    # Arrange
    # The ainvoke method should return a dictionary containing AIMessage objects.
    agent._graph.ainvoke.return_value = {"messages": [AIMessage(content="Hello")]}
    messages = [Message(role="user", content="Hello")]
    session_id = "test_session"

    # Act
    with patch.object(agent, '_LangGraphAgent__process_messages', return_value=[{'role': 'assistant', 'content': 'Hello', 'tool_calls': None}]) as mock_process:
        try:
            result = await agent.get_response(messages, session_id)
        except ValueError:
            pytest.fail("The fixed get_response code should not raise a ValueError.")

        # Assert
        assert result == [{'role': 'assistant', 'content': 'Hello', 'tool_calls': None}]
        agent._graph.ainvoke.assert_called_once()
        mock_process.assert_called_once_with([AIMessage(content="Hello")])


@pytest.mark.asyncio
async def test_get_stream_response_succeeds(agent):
    """
    This test verifies that the fix in get_stream_response is correct by
    ensuring it processes the stream correctly without raising a ValueError.
    """
    # Arrange
    async def async_iterator():
        yield AIMessage(content="Hello")
        yield AIMessage(content=" World")

    # The astream method should return an async iterator directly.
    agent._graph.astream.return_value = async_iterator()

    messages = [Message(role="user", content="Hello")]
    session_id = "test_session"

    # Act
    result = []
    try:
        async for chunk in agent.get_stream_response(messages, session_id):
            result.append(chunk)
    except ValueError:
        pytest.fail("The fixed get_stream_response code should not raise a ValueError.")

    # Assert
    assert result == ["Hello", " World"]
    agent._graph.astream.assert_called_once()
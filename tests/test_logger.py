from backend.app.core.logger import get_logger


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("tests.logger")

    assert logger.name == "personal_notes_assistant.tests.logger"

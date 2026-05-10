from app.core.bootstrap import (
    BootstrapStatus,
    ensure_models_exist,
    get_bootstrap_status,
)
from app.core.config import settings


def test_bootstrap_status_tracking(mocker):
    """Test that bootstrap failures are correctly tracked in the global state."""
    # Mock settings to enable local models
    mocker.patch.object(settings, "use_local_nlp_models", True)
    mocker.patch.object(settings, "models_dir")

    # Mock is_model_cached to return False to trigger verification failure
    mocker.patch("app.core.bootstrap.is_model_cached", return_value=False)

    ensure_models_exist()
    status = get_bootstrap_status()
    assert status["status"] == BootstrapStatus.DEGRADED
    assert status["error"] is not None
    assert "Missing model" in status["error"]


def test_bootstrap_status_success(mocker):
    """Test that bootstrap success resets the status and clears errors."""
    # Mock settings to enable local models
    mocker.patch.object(settings, "use_local_nlp_models", True)
    mocker.patch.object(settings, "models_dir")

    # Mock success for snapshots and files
    mocker.patch("app.core.bootstrap.is_model_cached", return_value=True)
    mocker.patch("pathlib.Path.exists", return_value=True)

    ensure_models_exist()

    status = get_bootstrap_status()
    assert status["status"] == BootstrapStatus.SUCCESS
    assert status["error"] is None

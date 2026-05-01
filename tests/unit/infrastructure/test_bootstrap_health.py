import pytest

from app.config import settings
from app.infrastructure.bootstrap import (
    BootstrapStatus,
    ensure_models_exist,
    get_bootstrap_status,
)


def test_bootstrap_status_tracking(mocker):
    """Test that bootstrap failures are correctly tracked in the global state."""
    # Mock settings to enable local models
    mocker.patch.object(settings, "use_local_nlp_models", True)
    mocker.patch.object(settings, "models_dir")

    # Mock a failure in snapshot_download
    mocker.patch(
        "app.infrastructure.bootstrap.snapshot_download",
        side_effect=RuntimeError("Download failed"),
    )

    with pytest.raises(RuntimeError, match="Download failed"):
        ensure_models_exist()

    status = get_bootstrap_status()
    assert status["status"] == BootstrapStatus.FAILED
    assert "Download failed" in status["error"]


def test_bootstrap_status_success(mocker):
    """Test that bootstrap success resets the status and clears errors."""
    # Mock settings to enable local models
    mocker.patch.object(settings, "use_local_nlp_models", True)
    mocker.patch.object(settings, "models_dir")

    # Mock success
    mocker.patch("app.infrastructure.bootstrap.snapshot_download")
    mocker.patch("app.infrastructure.bootstrap.hf_hub_download")

    # Mock REQUIRED_MODELS check (it checks if files exist)
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("pathlib.Path.iterdir", return_value=[mocker.MagicMock()])

    ensure_models_exist()

    status = get_bootstrap_status()
    assert status["status"] == BootstrapStatus.SUCCESS
    assert status["error"] is None

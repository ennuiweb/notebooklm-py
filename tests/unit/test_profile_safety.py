import json
import os
from pathlib import Path

import httpx
import pytest

from notebooklm._core import is_auth_error
from notebooklm.auth import (
    AuthTokens,
    CookiePersistenceError,
    ProfileLeaseError,
    canonical_storage_path,
    profile_storage_lease,
    save_cookies_to_storage,
)
from notebooklm.client import NotebookLMClient
from notebooklm.exceptions import NetworkError, RateLimitError, RPCError, ServerError


def _storage(path: Path, value: str = "old") -> None:
    path.write_text(json.dumps({"cookies": [{"name": "SID", "value": value, "domain": ".google.com"}]}))


def test_storage_path_aliases_have_one_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "profile.json"
    assert canonical_storage_path(Path("profile.json")) == canonical_storage_path(path)


def test_profile_lease_contends_and_releases(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    _storage(path)
    with (
        profile_storage_lease(path),
        pytest.raises(ProfileLeaseError),
        profile_storage_lease(path),
    ):
        pass
    with profile_storage_lease(path):
        pass


def test_cookie_save_fsyncs_file_and_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "profile.json"
    _storage(path)
    jar = httpx.Cookies()
    jar.set("SID", "new", domain=".google.com")
    calls: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    save_cookies_to_storage(jar, path)
    assert len(calls) == 2
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["cookies"][0]["value"] == "new"


def test_cookie_save_cleans_temp_after_fsync_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "profile.json"
    _storage(path)
    original = path.read_text()
    jar = httpx.Cookies()
    jar.set("SID", "new", domain=".google.com")
    monkeypatch.setattr(os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("fsync failed")))
    with pytest.raises(CookiePersistenceError):
        save_cookies_to_storage(jar, path)
    assert path.read_text() == original
    assert list(tmp_path.glob(".profile.json.*.tmp")) == []


def test_rpc_16_is_auth_but_other_failure_classes_are_not() -> None:
    assert is_auth_error(RPCError("null result", rpc_code=16))
    assert is_auth_error(RPCError("Unauthenticated"))
    assert not is_auth_error(RateLimitError("429"))
    assert not is_auth_error(ServerError("500"))
    assert not is_auth_error(NetworkError("offline"))


@pytest.mark.asyncio
async def test_client_lifetime_lease_releases_on_close(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    _storage(path)
    jar = httpx.Cookies()
    jar.set("SID", "x", domain=".google.com")
    jar.set("__Secure-1PSIDTS", "y", domain=".google.com")
    auth = AuthTokens(
        {("SID", ".google.com"): "x", ("__Secure-1PSIDTS", ".google.com"): "y"},
        "csrf",
        "sid",
        storage_path=path,
        cookie_jar=jar,
    )
    first = NotebookLMClient(auth)
    second = NotebookLMClient(auth)
    await first.__aenter__()
    try:
        with pytest.raises(ProfileLeaseError):
            await second.__aenter__()
    finally:
        await first.close()
    await second.__aenter__()
    await second.close()

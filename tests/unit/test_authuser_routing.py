"""Focused tests for Google multi-login routing."""

from urllib.parse import parse_qs, urlparse

from notebooklm._core import ClientCore
from notebooklm.auth import AuthTokens
from notebooklm.rpc.types import RPCMethod
import pytest


@pytest.mark.asyncio
async def test_rpc_url_and_headers_route_authuser():
    auth = AuthTokens(cookies={}, csrf_token="csrf", session_id="sid", authuser=3)
    core = ClientCore(auth)

    await core.open()
    try:
        query = parse_qs(urlparse(core._build_url(RPCMethod.LIST_NOTEBOOKS)).query)
        assert query["authuser"] == ["3"]
        assert core.get_http_client().headers["x-goog-authuser"] == "3"
    finally:
        await core.close()

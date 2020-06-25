import contextlib
from unittest import mock

import pytest

from .swift import MockConnection


@pytest.fixture(scope="function")
def mock_swift(tmp_path):
    metadata_file = tmp_path.joinpath("metadata.json")
    metadata_file.touch()
    tmp_dir = tmp_path.as_posix()

    stack = contextlib.ExitStack()
    stack.enter_context(
        mock.patch(
            "swiftclient.client.Connection", return_value=MockConnection(tmpdir=tmp_dir)
        )
    )
    yield MockConnection(tmpdir=tmp_dir)
    stack.close()

import io
import time

import pytest
import swiftclient


@pytest.fixture
def test_data(mock_swift):
    containername = f"functional-tests-container-{int(time.time())}"
    objectname = f"functional-tests-object-{int(time.time())}"
    yield {
        "test_data": b"42" * 10,
        "etag": "2704306ec982238d85d4b235c925d58e",
        "containername": containername,
        "containername_2": f"{containername}_second",
        "containername_3": f"{containername}_third",
        "objectname": objectname,
        "objectname_2": f"{objectname}_second",
        "connection": mock_swift,
    }


class TestRunner:
    def __init__(self, connection, data):
        self.conn = connection
        self.data = data

    def __getattribute__(self, key):
        try:
            return super().__getattribute__("data")[key]
        except (KeyError, AttributeError):
            return super().__getattribute__(key)

    def check_account_headers(self, headers):
        headers_to_check = [
            "content-length",
            "x-account-object-count",
            "x-timestamp",
            "x-trans-id",
            "date",
            "x-account-bytes-used",
            "x-account-container-count",
            "content-type",
            "accept-ranges",
        ]
        for h in headers_to_check:
            assert h in headers
            assert bool(headers[h]) is True

    def check_container_headers(self, headers):
        header_keys = [
            "content-length",
            "x-container-object-count",
            "x-timestamp",
            "x-trans-id",
            "date",
            "x-container-bytes-used",
            "content-type",
            "accept-ranges",
        ]
        for header in header_keys:
            assert header in headers
            assert headers.get(header) is not None


@pytest.fixture
def test_cls(test_data):
    conn = test_data.pop("connection")
    testclass = TestRunner(conn, test_data)
    testclass.conn.put_container(testclass.containername)
    testclass.conn.put_container(testclass.containername_2)
    testclass.conn.put_object(
        testclass.containername, testclass.objectname, testclass.test_data
    )
    testclass.conn.put_object(
        testclass.containername, testclass.objectname_2, testclass.test_data
    )
    testclass.conn._connection._retry.reset_mock()
    yield testclass
    for obj in [testclass.objectname, testclass.objectname_2]:
        try:
            testclass.conn.delete_object(testclass.containername, obj)
        except swiftclient.ClientException:
            pass

    for container in [
        testclass.containername,
        testclass.containername_2,
        testclass.containername_3,
        testclass.containername + "_segments",
    ]:
        try:
            testclass.conn.delete_container(container)
        except swiftclient.ClientException:
            pass

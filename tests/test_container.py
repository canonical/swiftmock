import sys

import pytest
import swiftclient.exceptions


def test_stat_container(test_cls):
    headers = test_cls.conn.head_container(test_cls.containername)
    test_cls.check_container_headers(headers)


def test_list_container(test_cls):
    headers, objects = test_cls.conn.get_container(test_cls.containername)
    test_cls.check_container_headers(headers)
    assert len(objects) > 0
    test_object = next(iter(o for o in objects if o.get("name") == test_cls.objectname))
    assert test_object.get("bytes") == len(test_cls.test_data)
    assert test_object.get("hash") == test_cls.etag
    assert test_object.get("content_type") == "application/octet-stream"

    # Check if list limit is working
    headers, objects = test_cls.conn.get_container(test_cls.containername, limit=1)
    assert len(objects) == 1

    # Check full listing
    headers, objects = test_cls.conn.get_container(
        test_cls.containername, limit=1, full_listing=True
    )
    assert len(objects) == 2

    # Test marker
    headers, objects = test_cls.conn.get_container(
        test_cls.containername, marker=test_cls.objectname
    )
    assert len(objects) == 1
    assert objects[0].get("name") == test_cls.objectname_2


def test_create_container(test_cls):
    test_cls.conn.put_container(test_cls.containername_3)
    assert bool(test_cls.conn.head_container(test_cls.containername_3)) is True


def test_delete_container(test_cls):
    test_cls.conn.delete_object(test_cls.containername, test_cls.objectname)
    test_cls.conn.delete_object(test_cls.containername, test_cls.objectname_2)
    test_cls.conn.delete_container(test_cls.containername)

    # Container HEAD will raise an exception if container doesn't exist
    # which is only possible if previous requests succeeded
    with pytest.raises(swiftclient.exceptions.ClientException):
        test_cls.conn.head_container(test_cls.containername)


def test_post_container(test_cls):
    test_cls.conn.post_container(
        test_cls.containername, {"x-container-meta-color": "Something"}
    )
    headers = test_cls.conn.head_container(test_cls.containername)
    assert headers.get("x-container-meta-color") == "Something"

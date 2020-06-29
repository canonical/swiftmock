import io

import pytest
import swiftclient.exceptions


@pytest.mark.parametrize(
    "content_type, headers, expected_content_type",
    [
        (None, None, "application/octet-stream"),
        ("text/plain", None, "text/plain"),
        (None, {"Content-Type": "text/plain"}, "text/plain"),
        ("image/jpeg", {"Content-Type": "text/plain"}, "image/jpeg"),
    ],
)
def test_upload_object(content_type, headers, expected_content_type, test_cls):
    # Object with content from string
    put_object_kwargs = {
        "contents": test_cls.test_data,
    }
    if headers:
        put_object_kwargs["headers"] = headers
    if content_type:
        put_object_kwargs["content_type"] = content_type
    test_cls.conn.put_object(
        test_cls.containername, test_cls.objectname, **put_object_kwargs
    )
    hdrs = test_cls.conn.head_object(test_cls.containername, test_cls.objectname)
    assert hdrs.get("content-length") == str(len(test_cls.test_data))
    assert hdrs.get("etag") == test_cls.etag
    assert hdrs.get("content-type") == expected_content_type


def test_put_fileobj(test_cls):
    # Content from File-like object
    fileobj = io.BytesIO(test_cls.test_data)
    test_cls.conn.put_object(
        test_cls.containername, test_cls.objectname, contents=fileobj
    )
    hdrs = test_cls.conn.head_object(test_cls.containername, test_cls.objectname)
    assert hdrs.get("content-length") == str(len(test_cls.test_data))
    assert hdrs.get("etag") == test_cls.etag
    assert hdrs.get("content-type") == "application/octet-stream"

    # Content from File-like object, but read in chunks
    # XXX: NOT YET SUPPORTED
    #  fileobj = BytesIO(test_cls.test_data)
    #  test_cls.conn.put_object(
    #  test_cls.containername, test_cls.objectname,
    #  contents=fileobj, content_length=len(test_cls.test_data),
    #  chunk_size=10)
    #  hdrs = test_cls.conn.head_object(test_cls.containername, test_cls.objectname)
    #  assert hdrs.get("content-length") == str(len(test_cls.test_data))
    #  assert hdrs.get("etag") == test_cls.etag
    #  assert hdrs.get("content-type") == "application/octet-stream"


def test_put_invalid_etag(test_cls):
    # Wrong etag arg, should raise an exception
    with pytest.raises(swiftclient.exceptions.ClientException):
        test_cls.conn.put_object(
            test_cls.containername,
            test_cls.objectname,
            contents=test_cls.test_data,
            etag="invalid",
        )


def test_download_object(test_cls):
    hdrs, body = test_cls.conn.get_object(test_cls.containername, test_cls.objectname)
    assert body == test_cls.test_data

    # Download in chunks, should return a generator
    hdrs, body = test_cls.conn.get_object(
        test_cls.containername, test_cls.objectname, resp_chunk_size=10
    )
    downloaded_contents = b""
    while True:
        try:
            chunk = next(body)
        except StopIteration:
            break
        downloaded_contents += chunk
    assert downloaded_contents == test_cls.test_data

    # Download in chunks, should also work with read
    hdrs, body = test_cls.conn.get_object(
        test_cls.containername, test_cls.objectname, resp_chunk_size=10
    )
    num_bytes = 5
    downloaded_contents = body.read(num_bytes)
    assert len(downloaded_contents) == num_bytes
    downloaded_contents += body.read()
    assert downloaded_contents == test_cls.test_data


def test_put_object_using_generator(test_cls):
    # verify that put using a generator yielding empty strings does not
    # cause connection to be closed
    def data():
        yield b"should"
        yield b""
        yield b" tolerate"
        yield b""
        yield b" empty chunks"

    test_cls.conn.put_object(test_cls.containername, test_cls.objectname, data())
    hdrs, body = test_cls.conn.get_object(test_cls.containername, test_cls.objectname)
    assert body == b"should tolerate empty chunks"


def test_download_object_retry_chunked(test_cls):
    resp_chunk_size = 2
    hdrs, body = test_cls.conn.get_object(
        test_cls.containername, test_cls.objectname, resp_chunk_size=resp_chunk_size
    )
    data = next(body)
    assert data == test_cls.test_data[:resp_chunk_size], data
    assert test_cls.conn._connection._retry.call_count == 1
    for chunk in body.resp:
        # Flush remaining data from underlying response
        # (simulate a dropped connection)
        pass
    # Trigger the retry
    for chunk in body:
        data += chunk
    assert data == test_cls.test_data
    assert test_cls.conn._connection._retry.call_count == 2


@pytest.mark.parametrize("chunk_size, expected_attempts", [(None, 1), (0, 1)])
def test_download_object_non_chunked(test_cls, chunk_size, expected_attempts):
    kwargs = {}
    if chunk_size is not None:
        kwargs["resp_chunk_size"] = chunk_size
    hdrs, body = test_cls.conn.get_object(
        test_cls.containername, test_cls.objectname, **kwargs
    )
    data = body
    assert data == test_cls.test_data
    assert test_cls.conn._connection._retry.call_count == expected_attempts


def test_post_object(test_cls):
    test_cls.conn.post_object(
        test_cls.containername,
        test_cls.objectname,
        {
            "x-object-meta-color": "Something",
            "x-object-meta-uni": b"\xd8\xaa".decode("utf8"),
            "x-object-meta-int": 123,
            "x-object-meta-float": 45.67,
            "x-object-meta-bool": False,
        },
    )

    headers = test_cls.conn.head_object(test_cls.containername, test_cls.objectname)
    assert headers.get("x-object-meta-color") == "Something"
    assert headers.get("x-object-meta-uni") == b"\xd8\xaa".decode("utf-8")
    assert headers.get("x-object-meta-int") == "123"
    assert headers.get("x-object-meta-float") == "45.67"
    assert headers.get("x-object-meta-bool") == "False"


def test_post_object_unicode_header_name(test_cls):
    # Note: this header can't be read on python 3 or something so
    # we will just have to make sure we can post it i guess?
    test_cls.conn.post_object(
        test_cls.containername,
        test_cls.objectname,
        {"x-object-meta-\U0001f44d": "\U0001f44d"},
    )


def test_copy_object(test_cls):
    test_cls.conn.put_object(
        test_cls.containername, test_cls.objectname, test_cls.test_data
    )
    test_cls.conn.copy_object(
        test_cls.containername,
        test_cls.objectname,
        headers={"x-object-meta-color": "Something"},
    )

    headers = test_cls.conn.head_object(test_cls.containername, test_cls.objectname)
    assert headers.get("x-object-meta-color") == "Something"

    test_cls.conn.copy_object(
        test_cls.containername,
        test_cls.objectname,
        headers={"x-object-meta-taste": "Second"},
    )

    headers = test_cls.conn.head_object(test_cls.containername, test_cls.objectname)
    assert headers.get("x-object-meta-color") == "Something"
    assert headers.get("x-object-meta-taste") == "Second"

    destination = "/{}/{}".format(test_cls.containername, test_cls.objectname_2)
    test_cls.conn.copy_object(
        test_cls.containername,
        test_cls.objectname,
        destination,
        headers={"x-object-meta-taste": "Second"},
    )
    headers, data = test_cls.conn.get_object(
        test_cls.containername, test_cls.objectname_2
    )
    assert data == test_cls.test_data
    assert headers.get("x-object-meta-color") == "Something"
    assert headers.get("x-object-meta-taste") == "Second"

    destination = "/{}/{}".format(test_cls.containername, test_cls.objectname_2)
    test_cls.conn.copy_object(
        test_cls.containername,
        test_cls.objectname,
        destination,
        headers={"x-object-meta-color": "Else"},
        fresh_metadata=True,
    )

    headers = test_cls.conn.head_object(test_cls.containername, test_cls.objectname_2)
    assert headers.get("x-object-meta-color") == "Else"
    assert headers.get("x-object-meta-taste") is None


def test_symlink(test_cls):
    test_cls.conn.put_object(
        test_cls.containername, test_cls.objectname, test_cls.test_data
    )
    headers = {
        "X-Symlink-Target": f"/{test_cls.containername}/{test_cls.objectname}",
    }
    symlink_object = "fake-symlink"
    test_cls.conn.put_object(
        test_cls.containername,
        symlink_object,
        b"",
        content_length=0,
        content_type="application/symlink",
        headers=headers,
    )
    headers = test_cls.conn.head_object(test_cls.containername, symlink_object)
    assert headers.get("content-type") == "application/symlink"
    assert (
        headers.get("X-Symlink-Target")
        == f"/{test_cls.containername}/{test_cls.objectname}"
    )
    assert headers.get("content-length") == "0"

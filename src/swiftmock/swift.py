import atexit
import datetime
import hashlib
import http
import io
import json
import mimetypes
import os
import pathlib
import shutil
import sys
import tempfile
import uuid
from functools import reduce
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from unittest import mock

import pytest
from requests.exceptions import StreamConsumedError
from requests.utils import stream_decode_response_unicode


def generate_requestid():
    initial_id = "".join(str(uuid.uuid4()).split("-"))
    field1 = initial_id[:20]
    field2 = initial_id[-10:]
    return f"tx{field1}-{field2}"


def generate_id():
    return "".join(str(uuid.uuid4()).split("-"))


def user_id():
    return generate_id()


def domain_id():
    return generate_id()


def gen_account_headers(container_dicts):
    """
    {
    'date': 'Thu, 18 Jun 2020 00:28:54 GMT',
    'server': 'Apache/2.4.29 (Ubuntu)',
    'content-length': '207',
    'x-account-object-count': '1444',
    'x-account-storage-policy-policy-0-bytes-used': '708097799',
    'x-account-storage-policy-policy-0-container-count': '2',
    'x-timestamp': '1576288774.22988',
    'x-account-storage-policy-policy-0-object-count': '1444',
    'x-account-bytes-used': '708097799',
    'x-account-container-count': '2',
    'content-type': 'application/json; charset=utf-8',
    'accept-ranges': 'bytes',
    'x-account-project-domain-id': '938e1b61817846668dfe90546bd18f21',
    'x-trans-id': 'tx4d7dd179ce18439098658-005eeab546',
    'x-openstack-request-id': 'tx4d7dd179ce18...
    }
    """
    request_id = generate_requestid()

    def get_container_dicts(children):
        for child in children:
            yield {"count": 1, "bytes": child["bytes"]}

    request_id = generate_requestid()
    summary = reduce(
        lambda x, y: {
            "count": x["count"] + y["count"],
            "bytes": x["bytes"] + y["bytes"],
        },
        get_container_dicts(container_dicts),
    )
    timestamp = str(
        min(
            [
                datetime.datetime.fromisoformat(c["last_modified"]).timestamp()
                for c in container_dicts
            ]
        )
    )

    return {
        "date": get_swift_object_date(datetime.datetime.utcnow()),
        "server": "Apache/2.4.29 (Ubuntu)",
        "content-length": str(len(str(container_dicts))),
        "content-type": "application/json; charset=utf-8",
        "x-account-object-count": f"{summary['count']}",
        "x-account-storage-policy-policy-0-bytes-used": f"{summary['bytes']}",
        "x-account-storage-policy-policy-0-container-count": str(len(container_dicts)),
        "x-account-bytes-used": f"{summary['bytes']}",
        "x-account-container-count": str(len(container_dicts)),
        "x-account-storage-policy-policy-0-object-count": f"{summary['count']}",
        "x-timestamp": timestamp,
        "accept-ranges": "bytes",
        "x-account-project-domain-id": domain_id(),
        "x-trans-id": request_id,
        "x-openstack-request-id": request_id,
    }


def get_container_headers(path: pathlib.Path, path_dicts: List[Dict[str, str]]):
    """
    {
    'date': 'Thu, 18 Jun 2020 22:35:02 GMT',
    'server': 'Apache/2.4.29 (Ubuntu)',
    'content-length': '751',
    'x-container-object-count': '4',
    'x-timestamp': '1576288774.55503',
    'accept-ranges': 'bytes',
    'x-storage-policy': 'Policy-0',
    'last-modified': 'Sat, 14 Dec 2019 01:59:56 GMT',
    'x-container-bytes-used': '18212',
    'content-type': 'application/json; charset=utf-8',
    'x-trans-id': 'tx5bdcc6206e5240c5b141f-005eebec16',
    'x-openstack-request-id': 'tx5bdcc6206e5240c5b141f-005eebec16'
    }
    """

    def get_path_dicts(children):
        yield {"count": 0, "bytes": 0}
        for child in children:
            if "bytes" not in child:
                yield {"count": 0, "bytes": 0}
            else:
                yield {"count": 1, "bytes": child["bytes"]}

    request_id = generate_requestid()
    summary = reduce(
        lambda x, y: {
            "count": x["count"] + y["count"],
            "bytes": x["bytes"] + y["bytes"],
        },
        get_path_dicts(path_dicts),
    )

    headers = {
        "date": get_swift_object_date(datetime.datetime.utcnow()),
        "server": "Apache/2.4.29 (Ubuntu)",
        "content-length": str(len(str(path_dicts))),
        "x-container-object-count": summary["count"],
        "x-timestamp": datetime.datetime.utcfromtimestamp(
            path.stat().st_ctime
        ).isoformat(),
        "accept-ranges": "bytes",
        "x-storage-policy": "Policy-0",
        "last-modified": get_swift_object_date(
            datetime.datetime.utcfromtimestamp(path.stat().st_mtime)
        ),
        "x-container-bytes-used": summary["bytes"],
        "content-type": "application/json; charset=utf-8",
        "x-trans-id": request_id,
        "x-openstack-request-id": request_id,
    }
    return headers


def get_swift_object_date(date: datetime.date) -> str:
    return (
        date.astimezone(datetime.timezone.utc)
        .strftime("%a, %d %b %Y %H:%M:%S %Z")
        .replace("UTC", "GMT")
    )


def get_swift_date(date: datetime.date) -> str:
    return date.astimezone(datetime.timezone.utc).isoformat()


def recurse(path: pathlib.Path):
    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir():
                yield from recurse(child)
            else:
                yield child
    else:
        yield path


def summarize_path(path: pathlib.Path):
    def get_path_dicts(target):
        yield {"count": 0, "bytes": 0}
        for child in recurse(target):
            yield {"count": 1, "bytes": child.stat().st_size}

    summary = reduce(
        lambda x, y: {
            "count": x["count"] + y["count"],
            "bytes": x["bytes"] + y["bytes"],
        },
        get_path_dicts(path),
    )
    summary.update(
        {
            "last_modified": datetime.datetime.fromtimestamp(
                path.stat().st_mtime
            ).isoformat(),
            "name": path.name,
        }
    )
    return summary


def is_fp_closed(fp):
    try:
        return fp.isclosed()
    except AttributeError:
        pass
    try:
        return fp.closed
    except AttributeError:
        pass
    try:
        return fp.fp is None
    except AttributeError:
        pass
    raise ValueError("could not find fp on object")


class MockUrllib3Response(object):
    def __init__(self, path):
        self._fp = io.open(path, "rb")
        self._fp_bytes_read = 0
        self.auto_close = True
        self.length_remaining = os.stat(path).st_size

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def stream(self, amt=2 ** 16, decode_content=None):
        while not is_fp_closed(self._fp):
            data = self.read(amt=amt, decode_content=decode_content)
            if data:
                yield data

    def close(self):
        if not self.closed:
            self._fp.close()

    @property
    def closed(self):
        if not self.auto_close:
            return io.IOBase.closed.__get__(self)
        elif self._fp is None:
            return True
        elif hasattr(self._fp, "isclosed"):
            return self._fp.isclosed()
        elif hasattr(self._fp, "closed"):
            return self._fp.closed
        else:
            return True

    def fileno(self):
        if self._fp is None:
            raise IOError("HTTPResponse has no file to get a fileno from")
        elif hasattr(self._fp, "fileno"):
            return self._fp.fileno()
        else:
            raise IOError(
                "The file-like object this HTTPResponse is wrapped "
                "around has no file descriptor"
            )

    def readinto(self, b):
        # This method is required for `io` module compatibility.
        temp = self.read(len(b))
        if len(temp) == 0:
            return 0
        else:
            b[: len(temp)] = temp
            return len(temp)

    @property
    def data(self):
        if self._body:
            return self._body
        if self._fp:
            return self.read(cache_content=True)

    def read(self, amt=None, decode_content=None, cache_content=False):
        if self._fp is None:
            return
        fp_closed = getattr(self._fp, "closed", False)
        if amt is None:
            # cStringIO doesn't like amt=None
            data = self._fp.read() if not fp_closed else b""
        else:
            cache_content = False
            data = self._fp.read(amt) if not fp_closed else b""
            if amt != 0 and not data:  # Platform-specific: Buggy versions of Python.
                self._fp.close()

        if data:
            self._fp_bytes_read += len(data)
            if self.length_remaining is not None:
                self.length_remaining -= len(data)

            if cache_content:
                self._body = data

        return data

    def isclosed(self):
        return is_fp_closed(self._fp)

    def tell(self):
        """Obtain the number of bytes pulled over the wire so far.

        May differ from the amount of content returned by
        :meth:``HTTPResponse.read`` if bytes are encoded on the wire
        (e.g, compressed).
        """
        return self._fp_bytes_read

    def __iter__(self):
        buffer = []
        for chunk in self.stream(decode_content=True):
            if b"\n" in chunk:
                chunk = chunk.split(b"\n")
                yield b"".join(buffer) + chunk[0] + b"\n"
                for x in chunk[1:-1]:
                    yield x + b"\n"
                if chunk[-1]:
                    buffer = [chunk[-1]]
                else:
                    buffer = []
            else:
                buffer.append(chunk)
        if buffer:
            yield b"".join(buffer)


def iter_slices(string, slice_length):
    """Iterate over slices of a string."""
    pos = 0
    if slice_length is None or slice_length <= 0:
        slice_length = len(string)
    while pos < len(string):
        yield string[pos : pos + slice_length]
        pos += slice_length


class MockResponse:
    def __init__(self, path):
        self._content = False
        self._content_consumed = False
        self._next = None

        #: Integer Code of responded HTTP Status, e.g. 404 or 200.
        self.status_code = 200
        self.raw = MockUrllib3Response(path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __iter__(self):
        """Allows you to use a response as an iterator."""
        return self.iter_content(128)

    def close(self):
        if not self._content_consumed:
            self.raw.close()

        release_conn = getattr(self.raw, "release_conn", None)
        if release_conn is not None:
            release_conn()

    def iter_content(self, chunk_size=1, decode_unicode=False):
        """Iterates over the response data.

        When stream=True is set on the request, this avoids reading the
        content at once into memory for large responses.  The chunk size
        is the number of bytes it should read into memory.  This is not
        necessarily the length of each item returned as decoding can
        take place. chunk_size must be of type int or None. A value of
        None will function differently depending on the value of
        `stream`. stream=True will read data as it arrives in whatever
        size the chunks are received. If stream=False, data is returned
        as a single chunk. If decode_unicode is True, content will be
        decoded using the best available encoding based on the response.
        """

        def generate():
            # Special case for urllib3.
            if hasattr(self.raw, "stream"):
                for chunk in self.raw.stream(chunk_size, decode_content=True):
                    yield chunk
            else:
                # Standard file-like object.
                while True:
                    chunk = self.raw.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

            self._content_consumed = True

        if self._content_consumed and isinstance(self._content, bool):
            raise StreamConsumedError()
        elif chunk_size is not None and not isinstance(chunk_size, int):
            raise TypeError(
                "chunk_size must be an int, it is instead a %s." % type(chunk_size)
            )
        # simulate reading small chunks of the content
        reused_chunks = iter_slices(self._content, chunk_size)

        stream_chunks = generate()

        chunks = reused_chunks if self._content_consumed else stream_chunks

        if decode_unicode:
            chunks = stream_decode_response_unicode(chunks, self)

        return chunks

    def iter_lines(self, chunk_size=512, decode_unicode=False, delimiter=None):
        """Iterates over the response data, one line at a time.  When
        stream=True is set on the request, this avoids reading the content at
        once into memory for large responses.

        .. note:: This method is not reentrant safe.
        """

        pending = None

        for chunk in self.iter_content(
            chunk_size=chunk_size, decode_unicode=decode_unicode
        ):

            if pending is not None:
                chunk = pending + chunk

            if delimiter:
                lines = chunk.split(delimiter)
            else:
                lines = chunk.splitlines()

            if lines and lines[-1] and chunk and lines[-1][-1] == chunk[-1]:
                pending = lines.pop()
            else:
                pending = None

            for line in lines:
                yield line

        if pending is not None:
            yield pending

    @property
    def content(self):
        """Content of the response, in bytes."""

        if self._content is False:
            # Read the contents.
            if self._content_consumed:
                raise RuntimeError("The content for this response was already consumed")

            if self.status_code == 0 or self.raw is None:
                self._content = None
            else:
                self._content = b"".join(self.iter_content(512)) or b""

        self._content_consumed = True
        # don't need to release the connection; that's been handled by urllib3
        # since we exhausted the data.
        return self._content

    def read(self, *args, **kwargs):
        chunk = self.raw.read(*args, **kwargs)
        if not chunk:
            self.raw.close()
        return chunk


class MockRetryBody:
    def __init__(self, container, obj, path, resp_chunk_size=None):
        self.path = pathlib.Path(path)
        self.resp = MockResponse(path)
        # self.buffer = io.BytesIO(self.path.read_bytes())
        self.chunk_size = resp_chunk_size
        self.expected_length = int(self.path.stat().st_size)
        self.container = container
        self.obj = obj
        self.bytes_read = 0

    def read(self, length=None):
        buf = None
        try:
            buf = self.resp.read(length)
        except OSError:
            raise
        return buf

    def __iter__(self):
        return self

    def next(self):
        buf = self.resp.read(self.chunk_size)
        if not buf:
            raise StopIteration()
        return buf

    def close(self):
        self.resp.close()


class MockConnection:
    """Compatible class to provide local files over the swift interface.

    This is used to mock out the swift interface for testing against the
    storage plugin system.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.tmpdir = kwargs.pop("tmpdir", None)
        if not self.tmpdir:
            self.tmpdir = tempfile.TemporaryDirectory()
            atexit.register(self.tmpdir.cleanup)
            self.base = pathlib.Path(self.tmpdir.name)
        else:
            self.base = pathlib.Path(self.tmpdir)
        self.metadata_file = kwargs.pop(
            "metadata_file", self.base.joinpath("metadata.json")
        )
        self.init_metadata()
        _conn_mock = mock.MagicMock("swiftclient.client.Connection", autospec=True)
        _connection = _conn_mock()
        _connection.get_account.return_value = ("", "")
        self._connection = _connection

    def init_metadata(self):
        content = self.metadata_file.read_text()
        if not content:
            self.metadata_file.write_text("{}")

    def read_metadata(self):
        return json.loads(self.metadata_file.read_text())

    def get_path_metadata(self, container, path):
        metadata = self.read_metadata()
        if container not in metadata:
            metadata[container] = {}
        if path not in metadata[container]:
            metadata[container][path] = {}
        print(f"Reading metadata: {container} {path}", file=sys.stderr)
        print(metadata[container][path], file=sys.stderr)
        return metadata[container][path]

    def write_metadata(self, container, path, data):
        contents = self.read_metadata()
        if container not in contents:
            contents[container] = {}
        if path not in contents[container]:
            contents[container][path] = {}
        contents[container][path].update(data)
        print(f"Writing metadata: {container} {path}", file=sys.stderr)
        self.metadata_file.write_text(json.dumps(contents))

    def __getattr__(self, key: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return self.__getattribute__(key, *args, **kwargs)
        except AttributeError:
            return self.__getattribute__("_connection").getattr(key, *args, **kwargs)

    def get_path(self, container: str, key: Optional[str] = None) -> pathlib.Path:
        if container.startswith(self.base._flavour.sep):
            container = container.lstrip(self.base._flavour.sep)
        path = self.base / container
        if key:
            path = path / key
        return path

    def get_relative_path(
        self, base_container: str, path: pathlib.Path
    ) -> pathlib.Path:
        if base_container.startswith(self.base._flavour.sep):
            base_container = base_container.lstrip(self.base._flavour.sep)
        container_path = self.base / base_container
        return path.relative_to(container_path)

    def get_swift_file_attrs(
        self, path: pathlib.Path, container: str = ""
    ) -> Dict[str, Union[int, str, pathlib.Path, datetime.datetime]]:
        if not path.is_absolute():
            path = self.get_path(container, key=path)
        if not path.exists():
            from swiftclient.exceptions import ClientException

            raise ClientException(f"No such file: {path!s}")
        try:
            last_modified = get_swift_date(
                datetime.datetime.fromtimestamp(path.stat().st_mtime)
            )
        except Exception:
            print(list(path.parent.iterdir()), file=sys.stderr)
        data = path.read_bytes()
        name = str(self.get_relative_path(container, path))
        mimetype, encoding = mimetypes.guess_type(path)
        if mimetype is None:
            mimetype = "application/octet-stream"
        if encoding is not None:
            mimetype = f"{mimetype}; encoding={encoding}"
        extra_headers = self.get_path_metadata(container.strip("/"), name.lstrip("/"))
        result_dict = {
            "bytes": len(data),
            "hash": hashlib.md5(data).hexdigest(),
            "name": type(path)(name.lstrip("/")),
            "content_type": extra_headers.get("content-type", mimetype),
            "last_modified": last_modified,
        }
        return result_dict

    def iter_dir(
        self, path: pathlib.Path, recurse: bool = False, container: str = ""
    ) -> Generator[
        Dict[str, Union[int, str, pathlib.Path, datetime.datetime]], None, None
    ]:
        if path.is_dir():
            for sub_path in path.iterdir():
                if sub_path.is_dir():
                    subdir_path = str(self.get_relative_path(container, sub_path))
                    if recurse:
                        yield from self.iter_dir(
                            sub_path, recurse=recurse, container=container
                        )
                    else:
                        yield {"subdir": subdir_path, "container": container}
                else:
                    yield self.get_swift_file_attrs(sub_path, container=container)
        else:
            yield self.get_swift_file_attrs(path, container=container)

    def put_container(
        self,
        container: str,
        headers: Optional[Dict[str, str]] = None,
        response_dict: Optional[Dict[str, str]] = None,
        query_string: Optional[str] = None,
    ) -> None:
        path = self.get_path(container)
        if path.exists():
            from swiftclient.exceptions import ClientException

            raise ClientException(f"Container {container} already exists!")
        path.mkdir(parents=True)

    def head_container(
        self, container: str, headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        if container.startswith(self.base._flavour.sep):
            container = container.lstrip(self.base._flavour.sep)
        headers, _ = self.get_container(container, headers=headers)
        if "content-length" in headers:
            headers.pop("content-length")
        return headers

    def head_account(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers, _ = self.get_account()
        if "content-length" in headers:
            headers.pop("content-length")
        return headers

    def get_account(
        self,
        marker: Optional[str] = None,
        limit: Optional[str] = None,
        prefix: Optional[str] = None,
        end_marker: Optional[str] = None,
        full_listing: bool = False,
        headers: Optional[Dict[str, str]] = None,
        delimiter: Optional[str] = None,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        container_paths = [pth for pth in self.base.iterdir() if pth.is_dir()]
        containers = []
        for container in container_paths:
            containers.append(summarize_path(container))
        account_headers = gen_account_headers(containers)
        return account_headers, containers

    def get_container(
        self,
        container: str,
        marker: Optional[str] = None,
        limit: Optional[int] = None,
        prefix: Optional[str] = None,
        delimiter: Optional[str] = None,
        end_marker: Optional[str] = None,
        path: Optional[str] = None,
        full_listing: bool = False,
        headers: Optional[Dict[str, str]] = None,
        query_string: Optional[str] = None,
    ) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        target = self.get_path(container, key=prefix)
        if not target.is_dir():
            from swiftclient.exceptions import ClientException

            raise ClientException(f"No such directory: {target!s}")
        if delimiter:
            files = self.iter_dir(target, recurse=False, container=container)
        else:
            files = self.iter_dir(target, recurse=True, container=container)
        results = list(files)
        if limit:
            results = results[:limit]
        headers = get_container_headers(target, results)
        extra_headers = self.get_path_metadata(container, prefix)
        if "content-type" in extra_headers:
            headers["content_type"] = extra_headers.pop("content-type")
        headers.update(extra_headers)
        if (
            query_string
            and query_string == "symlink=get"
            and "X-Symlink-Target" in extra_headers
        ):
            headers = extra_headers
            target_container, _, target = (
                extra_headers["X-Symlink-Target"].lstrip("/").partition("/")
            )
            headers.update(
                get_container_headers(
                    target_container, obj=prefix, query_string=query_string
                )
            )
        return headers, results

    def get_object(
        self,
        container: str,
        obj: str,
        resp_chunk_size: Optional[int] = None,
        query_string: Optional[str] = None,
        response_dict: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, str], bytes]:
        path = self.get_path(container, key=obj)
        if not path.exists():
            from swiftclient.exceptions import ClientException

            raise ClientException(f"No such path: {path!s}")
        headers = self.head_object(
            container, obj, headers=headers, query_string=query_string
        )
        print(headers, file=sys.stderr)
        print(f"Container: {container}\tobject: {obj}", file=sys.stderr)
        if "X-Symlink-Target" in headers:
            container, _, obj = headers["X-Symlink-Target"].lstrip("/").partition("/")
            container = container.strip("/")
            obj = obj.lstrip("/")
            path = self.get_path(container, key=obj)
            print(f"Container: {container}\tobject: {obj}", file=sys.stderr)
        resp = MockRetryBody(container, obj, path, resp_chunk_size=resp_chunk_size)
        if not resp_chunk_size:
            content = resp.read()
            return headers, content
        return headers, resp

    def head_object(
        self,
        container: str,
        obj: str,
        headers: Optional[Dict[str, str]] = None,
        query_string: Optional[str] = None,
    ) -> Dict[str, Union[datetime.datetime, str]]:
        path = self.get_path(container, key=obj)
        if not path.exists():
            from swiftclient.exceptions import ClientException

            raise ClientException(f"No such path: {path!s}")
        try:
            max_date = max(path.stat().st_mtime, path.stat().st_ctime)
            current_timestamp = get_swift_object_date(datetime.datetime.utcnow())
            path_contents = path.read_bytes()
        except Exception:
            from swiftclient.exceptions import ClientException

            raise ClientException(f"Not a file: {path!s}")
        name = str(self.get_relative_path(container, path))
        mimetype, encoding = mimetypes.guess_type(name)
        if mimetype is None:
            mimetype = "application/octet-stream"
        if encoding is not None:
            mimetype = f"{mimetype}; encoding={encoding}"
        transaction_id = generate_requestid()
        extra_headers = self.get_path_metadata(container.strip("/"), name.lstrip("/"))
        print(extra_headers, file=sys.stderr)
        headers = {
            "date": current_timestamp,
            "server": "Apache/2.4.29 (Ubuntu)",
            "content-length": "{}".format(path.stat().st_size),
            "accept-ranges": "bytes",
            "last-modified": get_swift_object_date(
                datetime.datetime.utcfromtimestamp(path.stat().st_mtime)
            ),
            "etag": hashlib.md5(path_contents).hexdigest(),
            "x-timestamp": f"{max_date}",
            "x-object-meta-mtime": f"{path.stat().st_mtime}",
            "content_type": extra_headers.get("content-type", mimetype),
            "x-trans-id": transaction_id,
            "x-openstack-request-id": transaction_id,
        }
        headers.update(extra_headers)
        if (
            query_string
            and query_string == "symlink=get"
            and "X-Symlink-Target" in extra_headers
        ):
            target_container, _, target = (
                extra_headers["X-Symlink-Target"].lstrip("/").partition("/")
            )
            headers.update(
                self.head_object(
                    target_container, obj=target, query_string=query_string
                )
            )
        print(headers, file=sys.stderr)
        return headers

    def post_object(
        self,
        container: str,
        obj: str,
        headers: Dict[str, Any],
        response_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        path = self.get_path(container, key=obj)
        path.touch()

    def copy_object(
        self,
        container: str,
        obj: str,
        destination: str,
        headers: Optional[Dict[str, str]] = None,
        fresh_metadata: Any = None,
        response_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        # destination path always starts with container/
        dest_container, _, dest_path = destination.partition("/")
        dest = self.get_path(dest_container, key=dest_path)
        base = self.get_path(container, key=obj)
        if not dest.parent.exists():
            dest.parent.mkdir(parents=True)
        dest.write_bytes(base.read_bytes())

    def put_object(
        self,
        container: str,
        obj: str,
        contents: Union[str, bytes],
        content_length: Optional[int] = None,
        etag: Any = None,
        chunk_size: Optional[int] = None,
        content_type: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        query_string: Optional[str] = None,
        response_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        dest = self.get_path(container, key=obj)
        print(f"writing path: {dest} : {contents}", file=sys.stderr)
        if not dest.parent.exists():
            dest.parent.mkdir(parents=True)
        if getattr(contents, "read", None):
            contents = contents.read()
        if content_type and not headers:
            metadata = self.get_path_metadata(container.strip("/"), obj.lstrip("/"))
            metadata["content-type"] = content_type
            self.write_metadata(container.strip("/"), obj.lstrip("/"), metadata)
        if headers and "X-Symlink-Target" in headers:
            src_container, _, src_path = (
                headers["X-Symlink-Target"].lstrip("/").partition("/")
            )
            metadata = self.get_path_metadata(container.strip("/"), obj.lstrip("/"))
            metadata.update(headers)
            if content_type:
                metadata["content-type"] = content_type
            self.write_metadata(container.strip("/"), obj.lstrip("/"), metadata)
            print(
                f"Container: {container}\tobject: {obj}\tmetadata: {metadata}",
                file=sys.stderr,
            )
            dest.write_bytes(b"")
        elif isinstance(contents, bytes):
            dest.write_bytes(contents)
        else:
            dest.write_text(contents)
        if etag:
            calculated_etag = hashlib.md5(dest.read_bytes()).hexdigest()
            if calculated_etag != etag:
                dest.unlink()
                from swiftclient.exceptions import ClientException

                raise ClientException(
                    f"ETag value mismatch: {calculated_etag} != {etag}"
                )
        if content_length and len(dest.read_bytes()) != content_length:
            from swiftclient.exceptions import ClientException

            dest.unlink()
            raise ClientException(
                f"Content does not match expected length: {len(dest.read_bytes())} != "
                f"{content_length}"
            )

    def delete_object(
        self,
        container: str,
        obj: str,
        query_string: Optional[str] = None,
        response_dict: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        target = self.get_path(container, key=obj)
        if not target.exists():
            from swiftclient.exceptions import ClientException

            raise ClientException(f"File does not exist: {target!s}")
        target.unlink()
        if not list(target.parent.iterdir()):
            target.parent.rmdir()

    def delete_container(
        self,
        container: str,
        response_dict: Optional[Dict[str, Any]] = None,
        query_string: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        target = self.get_path(container)
        if not target.exists():
            from swiftclient.exceptions import ClientException

            raise ClientException(f"No such container: {container}")
        shutil.rmtree(target)

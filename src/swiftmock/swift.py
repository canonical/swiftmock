import atexit
import datetime
import hashlib
import http
import io
import json
import logging
import mimetypes
import operator
import os
import pathlib
import shutil
import sys
import tempfile
import uuid
from collections.abc import Iterable
from functools import reduce
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from unittest import mock

import pytest
from requests.exceptions import RequestException, StreamConsumedError
from requests.utils import stream_decode_response_unicode

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


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
        yield {"count": 0, "bytes": 0}
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
        self.headers = {
            "Content-Length": os.stat(path).st_size,
            "ETag": hashlib.md5(path.read_bytes()).hexdigest(),
        }

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

    def getheader(self, header):
        return self.headers.get(header)

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
    def __init__(
        self,
        conn,
        container,
        obj,
        path,
        resp_chunk_size=None,
        query_string=None,
        response_dict=None,
        headers=None,
    ):
        self.conn = conn
        self.path = pathlib.Path(path)
        self.resp = MockResponse(path)
        if headers is None:
            headers = {}
        self.headers = headers
        self.response_dict = response_dict
        self.query_string = query_string
        # self.buffer = io.BytesIO(self.path.read_bytes())
        self.chunk_size = resp_chunk_size
        resp_length = self.resp.getheader("Content-Length") or self.path.stat().st_size
        self.expected_length = resp_length
        self.container = container
        self.obj = obj
        self.bytes_read = 0

    def read(self, length=None):
        buf = None
        try:
            buf = self.resp.read(length)
            self.bytes_read += len(buf)
        except OSError:
            raise
        except RequestException:
            if self.conn.attempts > self.conn.retries:
                raise
        if (
            not buf
            and self.bytes_read < self.expected_length
            and self.conn.attempts <= self.conn.retries
        ):
            self.headers["Range"] = f"bytes={self.bytes_read}-"
            self.headers["If-Match"] = self.resp.getheader("ETag")
            headers, body = self.conn.get_object(
                self.container,
                self.obj,
                resp_chunk_size=self.chunk_size,
                query_string=self.query_string,
                response_dict=self.response_dict,
                headers=self.headers,
                attempts=self.conn.attempts,
            )
            expected_range = (
                f"bytes {self.bytes_read}-{self.expected_length-1}/"
                f"{self.expected_length}"
            )
            if "content-range" not in headers:
                to_read = self.bytes_read
                while to_read > 0:
                    buf = body.resp.read(min(to_read, self.chunk_size))
                    to_read -= len(buf)
            elif headers["content-range"] != expected_range:
                from swiftclient.exceptions import ClientException

                msg = (
                    f'Expected range "{expected_range}" while retrying '
                    f"{self.container}/{self.obj} but got \"{headers['content-range']}\""
                )
                raise ClientException(msg)
            self.resp = body.resp
            buf = self.read(length)
        return buf

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()  # noqa:B305

    def next(self):
        buf = self.read(self.chunk_size)
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
        self.retries = kwargs.pop("retries", 5)
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
        self._account_metadata = {}
        self.attempts = 0
        _conn_mock = mock.MagicMock("swiftclient.client.Connection", autospec=True)
        _connection = _conn_mock()
        _connection.get_account.return_value = ("", "")
        self._connection = _connection

    def _ensure_strlike(self, value: Any) -> Optional[Union[str, bytes]]:
        if value is None:
            return value
        if isinstance(value, (float, int)):
            return str(value)
        return value

    def init_metadata(self) -> None:
        content = self.metadata_file.read_text()
        if not content:
            self.metadata_file.write_text("{}")

    def read_metadata(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        return json.loads(self.metadata_file.read_text())

    def get_path_metadata(
        self, container: str, path: Union[str, None]
    ) -> Dict[str, str]:
        if path is None:
            path = "__base__"
        metadata = self.read_metadata()
        container = container.strip("/")
        path = path.lstrip("/") if path else None
        if container not in metadata:
            metadata[container] = {}
        if path not in metadata[container]:
            metadata[container][path] = {}
        result = metadata[container][path].copy()
        return {k: self._ensure_strlike(v) for k, v in result.items()}

    def write_metadata(
        self,
        container: str,
        path: Union[str, None],
        data: Dict[str, str],
        fresh_metadata: bool = False,
    ) -> None:
        if path is None:
            path = "__base__"
        contents = self.read_metadata()
        container = container.strip("/")
        path = path.lstrip("/")
        if container not in contents:
            contents[container] = {}
        if path not in contents[container]:
            contents[container][path] = {}
        if not fresh_metadata:
            contents[container][path].update(data)
        else:
            contents[container][path] = data
        self.metadata_file.write_text(json.dumps(contents))

    def __getattr__(self, key: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return self.__getattribute__(key, *args, **kwargs)
        except AttributeError:
            return self.__getattribute__("_connection").getattr(key, *args, **kwargs)

    def _retry(self, reset_func, func, *args, **kwargs):
        kwargs.pop("response_dict", None)
        self.attempts = kwargs.get("attempts", 0)
        self.attempts += 1
        super().__getattribute__("_connection")._retry(
            reset_func, func, *args, **kwargs
        )

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
        content_type = extra_headers.get("content-type", mimetype)
        result_dict = {
            "bytes": len(data),
            "hash": hashlib.md5(data).hexdigest(),
            "name": type(path)(name.lstrip("/")),
            "content_type": content_type,
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
        from swiftclient.client import put_container

        self._retry(
            None, put_container, container, headers, response_dict, query_string
        )
        path = self.get_path(container)
        if path.exists():
            from swiftclient.exceptions import ClientException

            raise ClientException(f"Container {container} already exists!")
        if headers:
            self.write_metadata(container, None, headers)
        path.mkdir(parents=True)

    def head_container(
        self, container: str, headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        from swiftclient.client import head_container

        self._retry(None, head_container, container, headers)
        if container.startswith(self.base._flavour.sep):
            container = container.lstrip(self.base._flavour.sep)
        headers, _ = self._get_container(container, headers=headers)
        #  if "content-length" in headers:
        #  headers.pop("content-length")
        return headers

    def head_account(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        from swiftclient.client import head_account

        self._retry(None, head_account, headers=headers)
        headers, _ = self._get_account()
        #  if "content-length" in headers:
        #  headers.pop("content-length")
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
        from swiftclient.client import get_account

        self._retry(
            None,
            get_account,
            marker=marker,
            limit=limit,
            prefix=prefix,
            end_marker=end_marker,
            full_listing=full_listing,
            headers=headers,
            delimiter=delimiter,
        )
        return self._get_account(
            marker=marker,
            limit=limit,
            prefix=prefix,
            end_marker=end_marker,
            full_listing=full_listing,
            headers=headers,
            delimiter=delimiter,
        )

    def _get_account(
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
        containers = {}
        is_in_markers = False if marker else True
        results = []
        for container in sorted(container_paths, key=operator.attrgetter("name")):
            containers[container.name] = summarize_path(container)

            matches_prefix = (
                prefix and container.name.startswith(prefix)
            ) or not prefix
            # Markers operate as exclusive (i.e. the outer matches are not included)
            # so we set the value here before evaluating whether to include this path
            # to ensure we don't accidentally include the end marker path
            if end_marker is not None and container.name.startswith(end_marker):
                is_in_markers = False
                continue
            elif not is_in_markers and container.name.startswith(marker):
                is_in_markers = True
                continue
            elif marker and is_in_markers:
                if matches_prefix:
                    results.append(container.name)
                continue
            elif matches_prefix and (not marker or is_in_markers):
                results.append(container.name)
        account_headers = gen_account_headers(list(containers.values()))
        account_headers.update(
            {
                k: v
                for k, v in self._account_metadata.items()
                if k not in account_headers
            }
        )
        results = [containers[name] for name in results]
        if limit and not full_listing:
            results = results[:limit]
        return account_headers, results

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
        version_marker: Any = None,
        query_string: Optional[str] = None,
    ) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        from swiftclient.client import get_container

        self._retry(
            None,
            get_container,
            container,
            marker=marker,
            prefix=prefix,
            delimiter=delimiter,
            end_marker=end_marker,
            version_marker=version_marker,
            path=path,
            full_listing=full_listing,
            headers=headers,
            query_string=query_string,
        )
        return self._get_container(
            container,
            marker=marker,
            limit=limit,
            prefix=prefix,
            delimiter=delimiter,
            end_marker=end_marker,
            path=path,
            full_listing=full_listing,
            headers=headers,
            query_string=query_string,
        )

    def _get_container(
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

            raise ClientException(f"No such container: {container}")
        elif delimiter:
            files = self.iter_dir(target, recurse=False, container=container)
        else:
            files = self.iter_dir(target, recurse=True, container=container)
        results = []
        is_in_markers = False if marker else True
        for r in sorted(files, key=operator.itemgetter("name")):
            result = r.copy()
            name = str(result["name"]).lstrip("/")
            if not is_in_markers:
                if name.startswith(marker):
                    is_in_markers = True
                continue
            if end_marker is not None and name.startswith(end_marker):
                break
            result["name"] = name
            results.append(result)
        if limit and not full_listing:
            results = results[:limit]
        if headers is None:
            headers = {}
        if files:
            headers.update(get_container_headers(target, results))
        metadata_path = path if path else prefix
        metadata = self.get_path_metadata(container.strip("/"), metadata_path)
        if "content-type" in metadata:
            headers["content-type"] = metadata.pop("content-type")
        headers.update(metadata)
        if "X-Symlink-Target" in metadata and not (
            query_string and query_string == "symlink=get"
        ):
            headers = metadata
            target_container, _, target = (
                metadata["X-Symlink-Target"].lstrip("/").partition("/")
            )
            headers.update(
                get_container_headers(
                    target_container, obj=prefix, query_string=query_string
                )
            )
        if limit and not full_listing:
            results = results[:limit]
        return headers, results

    def get_object(
        self,
        container: str,
        obj: str,
        resp_chunk_size: Optional[int] = None,
        query_string: Optional[str] = None,
        response_dict: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        attempts: Optional[int] = None,
    ) -> Tuple[Dict[str, str], bytes]:
        from swiftclient.client import get_object

        self._retry(
            None,
            get_object,
            container,
            obj,
            resp_chunk_size=resp_chunk_size,
            query_string=query_string,
            response_dict=response_dict,
            headers=headers,
        )
        path = self.get_path(container, key=obj)
        if attempts is not None:
            self.attempts = attempts
        if not path.exists():
            from swiftclient.exceptions import ClientException

            raise ClientException(f"No such path: {path!s}")
        headers = self._head_object(
            container, obj, headers=headers, query_string=query_string
        )
        if "X-Symlink-Target" in headers:
            container, _, obj = headers["X-Symlink-Target"].lstrip("/").partition("/")
            container = container.strip("/")
            obj = obj.lstrip("/")
            path = self.get_path(container, key=obj)
        resp = MockRetryBody(
            self,
            container,
            obj,
            path,
            resp_chunk_size=resp_chunk_size,
            query_string=query_string,
            response_dict=response_dict,
            headers=headers,
        )
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
        from swiftclient.client import head_object

        self._retry(
            None,
            head_object,
            container,
            obj,
            headers=headers,
            query_string=query_string,
        )
        return self._head_object(
            container, obj, headers=headers, query_string=query_string
        )

    def _head_object(
        self,
        container: str,
        obj: str,
        headers: Optional[Dict[str, str]] = None,
        query_string: Optional[str] = None,
    ) -> Dict[str, Union[datetime.datetime, str]]:
        path = self.get_path(container, key=obj)
        if not path.exists():
            logger.info(f"Path does not exist: {path!s}")
            from swiftclient.exceptions import ClientException

            raise ClientException(f"No such path: {path!s}")
        try:
            max_date = max(path.stat().st_mtime, path.stat().st_ctime)
            current_timestamp = get_swift_object_date(datetime.datetime.utcnow())
            path_contents = path.read_bytes()
        except Exception:
            logger.error(f"Failed to read path contents: {path!s})")
            from swiftclient.exceptions import ClientException

            raise ClientException(f"Not a file: {path!s}")
        name = str(self.get_relative_path(container, path))
        mimetype, encoding = mimetypes.guess_type(name)
        if mimetype is None:
            mimetype = "application/octet-stream"
        if encoding is not None:
            mimetype = f"{mimetype}; encoding={encoding}"
        transaction_id = generate_requestid()
        extra_headers = self.get_path_metadata(container, name)
        logger.info(f"Extra metadata for path {path!s}: {extra_headers}")
        content_type_key = next(
            iter(k for k in extra_headers if k.lower() == "content-type"), None
        )
        content_type = (
            extra_headers.pop(content_type_key) if content_type_key else mimetype
        )
        extra_headers["content-type"] = content_type
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
            "content_type": content_type,
            "x-trans-id": transaction_id,
            "x-openstack-request-id": transaction_id,
        }
        headers.update(extra_headers)
        # if the query string 'symlink=get' is provided, we should operate on the symlink
        # otherwise we should operate on the referenced object
        if "X-Symlink-Target" in extra_headers and not (
            query_string and query_string == "symlink=get"
        ):
            target_container, _, target = (
                extra_headers["X-Symlink-Target"].lstrip("/").partition("/")
            )
            headers.update(
                self._head_object(
                    target_container, obj=target, query_string=query_string
                )
            )
        return headers

    def post_account(
        self,
        headers: Dict[str, str],
        response_dict: Optional[Dict[str, str]] = None,
        query_string: Optional[str] = None,
        data: Optional[Dict[str, str]] = None,
    ) -> None:
        from swiftclient.client import post_account

        self._retry(
            None,
            post_account,
            headers,
            query_string=query_string,
            data=data,
            response_dict=response_dict,
        )
        self._account_metadata.update(headers)
        return

    def post_container(
        self,
        container: str,
        headers: Dict[str, str],
        response_dict: Dict[str, str] = None,
    ) -> None:
        from swiftclient.client import post_container

        self._retry(
            None, post_container, container, headers, response_dict=response_dict
        )
        path = self.get_path(container)
        if not path.exists():
            path.mkdir()
        if headers:
            metadata = self.get_path_metadata(container.strip("/"), None)
            metadata.update(headers)
            self.write_metadata(container.strip("/"), None, metadata)
        return

    def post_object(
        self,
        container: str,
        obj: str,
        headers: Dict[str, Any],
        response_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        from swiftclient.client import post_object

        self._retry(None, post_object, obj, headers, response_dict=response_dict)
        path = self.get_path(container, key=obj)
        path.touch()
        if headers:
            self.write_metadata(container, obj, headers)

    def copy_object(
        self,
        container: str,
        obj: str,
        destination: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        fresh_metadata: bool = False,
        response_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        from swiftclient.client import copy_object

        self._retry(
            None,
            copy_object,
            obj,
            destination,
            headers,
            fresh_metadata,
            response_dict=response_dict,
        )
        # destination path always starts with container/
        base = self.get_path(container, key=obj)
        if not destination:
            dest = base
            dest_container = container
            dest_path = obj
        else:
            destination = destination.lstrip("/")
            dest_container, _, dest_path = destination.partition("/")
            dest_container = dest_container.strip("/")
            dest_path = dest_path.lstrip("/")
            dest = self.get_path(dest_container, key=dest_path)
            src_headers = self.get_path_metadata(container, obj)
            if src_headers and not fresh_metadata:
                if headers is None:
                    headers = src_headers
                else:
                    src_headers.update(headers)
                    headers = src_headers.copy()
        if headers:
            self.write_metadata(
                dest_container, dest_path, headers, fresh_metadata=fresh_metadata
            )
        if dest == base:
            return
        if not dest.parent.exists():
            dest.parent.mkdir(parents=True)
        dest.write_bytes(base.read_bytes())

    def _validate_obj(
        self, path: pathlib.Path, etag: Optional[str], content_length: Optional[int]
    ) -> None:
        if etag:
            calculated_etag = hashlib.md5(path.read_bytes()).hexdigest()
            if calculated_etag != etag:
                from swiftclient.exceptions import ClientException

                raise ClientException(
                    f"ETag value mismatch: {calculated_etag} != {etag}"
                )
        if content_length and len(path.read_bytes()) != content_length:
            from swiftclient.exceptions import ClientException

            raise ClientException(
                f"Content does not match expected length: {len(path.read_bytes())} != "
                f"{content_length}"
            )
        return None

    def _merge_headers(
        self,
        content_type: Optional[str] = None,
        content_length: Optional[str] = None,
        etag: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        results = {}
        if headers is None:
            headers = {}
        header_content_type = next(
            iter(hdr for hdr in headers if hdr.lower() == "content-type"), None
        )
        header_etag = next(iter(hdr for hdr in headers if hdr.lower() == "etag"), None)
        if header_content_type is not None:
            if content_type is None:
                content_type = headers.pop(header_content_type)
            else:
                headers.pop(header_content_type)
        if header_etag is not None:
            if etag is None:
                etag = headers.pop(header_etag)
            else:
                headers.pop(header_etag)
        results.update(headers)
        if content_type:
            results["content-type"] = content_type
        if content_length:
            results["content-length"] = content_length
        if etag:
            results["etag"] = etag
        return results

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
        from swiftclient.client import put_object
        from swiftclient.exceptions import ClientException

        self._retry(
            None,
            put_object,
            container,
            obj,
            contents,
            content_length=content_length,
            etag=etag,
            chunk_size=chunk_size,
            content_type=content_type,
            headers=headers,
            query_string=query_string,
            response_dict=response_dict,
        )
        dest = self.get_path(container, key=obj)
        if not dest.parent.exists():
            dest.parent.mkdir(parents=True)
        if getattr(contents, "read", None):
            contents = contents.read()
        if content_type is not None and not headers:
            metadata = self.get_path_metadata(container, obj)
            headers = self._merge_headers(
                content_type=content_type,
                content_length=content_length,
                etag=etag,
                headers=headers,
            )
            metadata.update(headers)
            self.write_metadata(container, obj, metadata)
        if headers:
            if "X-Symlink-Target" in headers:
                src_container, _, src_path = (
                    headers["X-Symlink-Target"].lstrip("/").partition("/")
                )
                contents = b""
            metadata = self.get_path_metadata(container, obj)
            headers = self._merge_headers(
                content_type=content_type,
                content_length=content_length,
                etag=etag,
                headers=headers,
            )
            metadata.update(headers)
            self.write_metadata(container, obj, metadata)
        if isinstance(contents, bytes):
            dest.write_bytes(contents)
        elif isinstance(contents, str):
            dest.write_text(contents)
        elif isinstance(contents, Iterable) and not isinstance(contents, (str, bytes)):
            with dest.open("wb") as fh:
                for chunk in contents:
                    fh.write(chunk)
        elif isinstance(contents, io.FileIO):
            with dest.open("wb") as fh:
                shutil.copyfileobj(contents, fh)
        try:
            self._validate_obj(dest, etag=etag, content_length=content_length)
        except ClientException:
            dest.unlink()
            raise

    def delete_object(
        self,
        container: str,
        obj: str,
        query_string: Optional[str] = None,
        response_dict: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        from swiftclient.client import delete_object

        self._retry(
            None, delete_object, container, obj, headers, response_dict=response_dict
        )
        target = self.get_path(container, key=obj)
        if not target.exists():
            from swiftclient.exceptions import ClientException

            raise ClientException(f"File does not exist: {target!s}")
        target.unlink()
        if not list(target.parent.iterdir()) and not target.parent.parent == self.base:
            target.parent.rmdir()

    def delete_container(
        self,
        container: str,
        response_dict: Optional[Dict[str, Any]] = None,
        query_string: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        from swiftclient.client import delete_container

        self._retry(
            None,
            delete_container,
            container,
            response_dict=response_dict,
            query_string=query_string,
            headers=headers,
        )
        target = self.get_path(container)
        if not target.exists():
            from swiftclient.exceptions import ClientException

            raise ClientException(f"No such container: {container}")
        shutil.rmtree(target)

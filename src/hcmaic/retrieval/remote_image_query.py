"""Bounded, public-only remote image fetching for user-initiated queries."""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
from collections.abc import Callable
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit

MAX_REMOTE_IMAGE_URL_LENGTH = 2_048
MAX_REMOTE_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REMOTE_IMAGE_REDIRECTS = 3
REMOTE_IMAGE_TIMEOUT_SECONDS = 8.0
_READ_CHUNK_BYTES = 64 * 1024
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_ALLOWED_CONTENT_TYPES = frozenset({"image/jpeg", "image/png"})
_BLOCKED_HOSTNAMES = frozenset({"localhost", "localhost."})
_BLOCKED_HOST_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".intranet",
    ".test",
    ".invalid",
)


class RemoteImageFetchError(ValueError):
    """Safe, user-facing error for a rejected or unavailable remote image."""

    def __init__(self, status_code: int, code: str) -> None:
        self.status_code = status_code
        self.code = code
        super().__init__(code)


def _reject(code: str, status_code: int = 400) -> RemoteImageFetchError:
    return RemoteImageFetchError(status_code, code)


def _is_public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return bool(address.is_global)


def _resolve_public_addresses(host: str, port: int) -> list[str]:
    normalized_host = host.rstrip(".").lower()
    if normalized_host in _BLOCKED_HOSTNAMES or normalized_host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise _reject("REMOTE_IMAGE_URL_REJECTED")

    try:
        literal = ipaddress.ip_address(normalized_host)
    except ValueError:
        literal = None
    if literal is not None:
        if not literal.is_global:
            raise _reject("REMOTE_IMAGE_URL_REJECTED")
        return [str(literal)]

    try:
        records = socket.getaddrinfo(
            normalized_host,
            port,
            type=socket.SOCK_STREAM,
        )
    except (OSError, socket.gaierror) as exc:
        raise _reject("REMOTE_IMAGE_DNS_FAILED", 502) from exc

    addresses: list[str] = []
    for record in records:
        sockaddr = record[4]
        address = str(sockaddr[0])
        if address not in addresses:
            addresses.append(address)
    if not addresses or any(not _is_public_address(address) for address in addresses):
        # Reject mixed public/private answers as well as wholly private answers;
        # this avoids choosing a safe answer from a rebinding-capable response.
        raise _reject("REMOTE_IMAGE_URL_REJECTED")
    return addresses


def _validated_target(raw_url: str) -> tuple[SplitResult, list[str]]:
    value = str(raw_url or "").strip()
    if not value or len(value) > MAX_REMOTE_IMAGE_URL_LENGTH:
        raise _reject("REMOTE_IMAGE_URL_REJECTED")
    if any(character in value for character in "\r\n"):
        raise _reject("REMOTE_IMAGE_URL_REJECTED")
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
        username = parsed.username
        password = parsed.password
    except ValueError as exc:
        raise _reject("REMOTE_IMAGE_URL_REJECTED") from exc
    if scheme not in {"http", "https"} or not hostname or username or password:
        raise _reject("REMOTE_IMAGE_URL_REJECTED")
    if parsed.fragment:
        raise _reject("REMOTE_IMAGE_URL_REJECTED")
    if port is not None and not 1 <= port <= 65_535:
        raise _reject("REMOTE_IMAGE_URL_REJECTED")
    resolved_port = port if port is not None else (443 if scheme == "https" else 80)
    addresses = _resolve_public_addresses(hostname, resolved_port)
    return parsed, addresses


def validate_remote_image_url(raw_url: str) -> SplitResult:
    """Validate a URL and its current DNS answers without fetching bytes."""

    parsed, _addresses = _validated_target(raw_url)
    return parsed


def _host_header(parsed: SplitResult) -> str:
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    port = parsed.port
    default_port = 443 if parsed.scheme.lower() == "https" else 80
    return f"{hostname}:{port}" if port and port != default_port else hostname


def _connect_remote_response(
    parsed: SplitResult,
    address: str,
    timeout: float,
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    hostname = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    raw_socket = socket.create_connection((address, port), timeout=timeout)
    connection: http.client.HTTPConnection | None = None
    try:
        connected_socket = raw_socket
        if parsed.scheme.lower() == "https":
            context = ssl.create_default_context()
            connected_socket = context.wrap_socket(raw_socket, server_hostname=hostname)
        connection = http.client.HTTPConnection(hostname, port, timeout=timeout)
        connection.sock = connected_socket
        target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection.putrequest("GET", target, skip_host=True, skip_accept_encoding=True)
        connection.putheader("Host", _host_header(parsed))
        connection.putheader("Accept", "image/jpeg,image/png")
        connection.putheader("Accept-Encoding", "identity")
        connection.putheader("Connection", "close")
        connection.putheader("User-Agent", "HCMAIC-image-query/1.0")
        connection.endheaders()
        return connection, connection.getresponse()
    except Exception:
        if connection is not None:
            connection.close()
        else:
            raw_socket.close()
        raise


def _read_bounded_body(response: http.client.HTTPResponse) -> bytes:
    content_length = response.getheader("Content-Length")
    if content_length:
        try:
            declared_length = int(content_length)
        except ValueError:
            declared_length = 0
        if declared_length < 0 or declared_length > MAX_REMOTE_IMAGE_BYTES:
            raise _reject("REMOTE_IMAGE_TOO_LARGE", 413)

    chunks: list[bytes] = []
    total = 0
    while total <= MAX_REMOTE_IMAGE_BYTES:
        chunk = response.read(min(_READ_CHUNK_BYTES, MAX_REMOTE_IMAGE_BYTES - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_REMOTE_IMAGE_BYTES:
            raise _reject("REMOTE_IMAGE_TOO_LARGE", 413)
        chunks.append(chunk)
    return b"".join(chunks)


def fetch_remote_image(
    raw_url: str,
    *,
    connector: Callable[..., tuple[http.client.HTTPConnection, http.client.HTTPResponse]]
    | None = None,
) -> tuple[bytes, str]:
    """Fetch one public JPEG/PNG with pinned DNS, bounded redirects and bytes."""

    open_response = connector or _connect_remote_response
    current_url = raw_url
    for redirect_count in range(MAX_REMOTE_IMAGE_REDIRECTS + 1):
        parsed, addresses = _validated_target(current_url)
        connection: http.client.HTTPConnection | None = None
        response: http.client.HTTPResponse | None = None
        last_error: Exception | None = None
        for address in addresses:
            try:
                connection, response = open_response(
                    parsed,
                    address,
                    REMOTE_IMAGE_TIMEOUT_SECONDS,
                )
                break
            except (OSError, http.client.HTTPException) as exc:
                last_error = exc
        if response is None:
            raise _reject("REMOTE_IMAGE_FETCH_FAILED", 502) from last_error

        try:
            status = int(response.status)
            if status in _REDIRECT_STATUSES:
                if redirect_count >= MAX_REMOTE_IMAGE_REDIRECTS:
                    raise _reject("REMOTE_IMAGE_REDIRECT_LIMIT", 502)
                location = response.getheader("Location")
                if not location:
                    raise _reject("REMOTE_IMAGE_FETCH_FAILED", 502)
                current_url = urljoin(current_url, location)
                continue
            if not 200 <= status < 300:
                raise _reject("REMOTE_IMAGE_FETCH_FAILED", 502)

            content_type = (
                str(response.getheader("Content-Type") or "")
                .split(";", 1)[0]
                .strip()
                .lower()
            )
            if content_type not in _ALLOWED_CONTENT_TYPES:
                raise _reject("REMOTE_IMAGE_UNSUPPORTED_TYPE", 415)
            return _read_bounded_body(response), content_type
        except RemoteImageFetchError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            raise _reject("REMOTE_IMAGE_FETCH_FAILED", 502) from exc
        finally:
            if connection is not None:
                connection.close()

    raise _reject("REMOTE_IMAGE_REDIRECT_LIMIT", 502)

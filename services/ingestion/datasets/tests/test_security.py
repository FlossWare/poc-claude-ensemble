"""Hostile YAML, static no-execution checks and SSRF guard coverage."""
from __future__ import annotations

import ast
import socket
from pathlib import Path

import pytest

from ingestion.datasets import model
from ingestion.datasets.netguard import NetguardError, RawResponse, Transport

from ingestion.datasets.tests.helpers import StubTransport, make_config, stub_resolver


def test_python_object_tag_is_rejected(tmp_path):
    config = make_config(tmp_path)
    raw = b"!!python/object/apply:os.system ['echo pwned']\n"
    with pytest.raises(model.EntryError) as error:
        model.load_yaml_entry(raw, config)
    assert error.value.code == "catalog_parse_error"


def test_alias_is_rejected_by_default(tmp_path):
    config = make_config(tmp_path)
    raw = b"a: &a [x, x]\nb: [*a, *a]\n"
    with pytest.raises(model.EntryError) as error:
        model.load_yaml_entry(raw, config)
    assert error.value.code == "limits_exceeded"


def test_alias_expansion_is_capped_when_aliases_are_allowed(tmp_path):
    config = make_config(tmp_path).replace(yaml_max_aliases=100, yaml_max_nodes=12)
    raw = (
        b"a: &a [x, x, x, x, x]\n"
        b"b: [*a, *a, *a, *a, *a, *a, *a, *a, *a, *a]\n"
    )
    with pytest.raises(model.EntryError) as error:
        model.load_yaml_entry(raw, config)
    assert error.value.code == "limits_exceeded"


def test_depth_bomb_is_rejected(tmp_path):
    config = make_config(tmp_path)
    raw = ("a: " + "[" * 100 + "x" + "]" * 100 + "\n").encode("ascii")
    with pytest.raises(model.EntryError) as error:
        model.load_yaml_entry(raw, config)
    assert error.value.code == "limits_exceeded"


def test_oversized_yaml_is_rejected(tmp_path):
    config = make_config(tmp_path).replace(catalog_max_file_bytes=32)
    with pytest.raises(model.EntryError) as error:
        model.load_yaml_entry(b"title: " + b"x" * 100 + b"\n", config)
    assert error.value.code == "limits_exceeded"


def test_static_check_no_execution_or_unsafe_loader():
    import ingestion.datasets as package

    package_dir = Path(package.__file__).parent
    forbidden_modules = {"pickle", "marshal", "shelve", "subprocess"}
    forbidden_calls = {"eval", "exec", "system", "popen", "extractall"}
    unsafe_yaml_attributes = {"Loader", "UnsafeLoader", "CLoader", "full_load", "unsafe_load"}
    # Recurse: the PR 3 adapters are a subpackage and must be covered too.
    for path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden_modules, path
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in forbidden_modules, path
            elif isinstance(node, ast.Call):
                function = node.func
                name = (
                    function.id
                    if isinstance(function, ast.Name)
                    else function.attr
                    if isinstance(function, ast.Attribute)
                    else None
                )
                assert name not in forbidden_calls, (path, name)
            elif isinstance(node, ast.Attribute):
                assert node.attr not in unsafe_yaml_attributes, (path, node.attr)
    model_source = (package_dir / "model.py").read_text(encoding="utf-8")
    assert "yaml.safe_load" in model_source
    assert "yaml.SafeLoader" in model_source


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "0.0.0.0",
        "100.64.0.1",
        "192.0.2.1",
        "::1",
        "fc00::1",
        "fe80::1",
    ],
)
def test_ssrf_guard_rejects_non_global_addresses(address, tmp_path):
    config = make_config(tmp_path)
    transport = Transport(config, resolver=stub_resolver({"example.com": address}))
    with pytest.raises(NetguardError) as error:
        transport.validate_url("https://example.com/")
    assert error.value.code == "ssrf_blocked"


def test_metadata_hostname_is_blocked(tmp_path):
    config = make_config(tmp_path)
    transport = Transport(
        config,
        resolver=stub_resolver({"metadata.google.internal": "169.254.169.254"}),
    )
    with pytest.raises(NetguardError) as error:
        transport.validate_url("http://metadata.google.internal/computeMetadata/v1/")
    assert error.value.code == "ssrf_blocked"


def test_public_address_is_accepted(tmp_path):
    config = make_config(tmp_path)
    transport = Transport(config, resolver=stub_resolver({"example.com": "93.184.216.34"}))
    validated = transport.validate_url("https://example.com/path?q=1")
    assert validated.address == "93.184.216.34"
    assert validated.path_query == "/path?q=1"


def test_url_validation_refusals(tmp_path):
    config = make_config(tmp_path)
    transport = Transport(config, resolver=stub_resolver({"example.com": "93.184.216.34"}))
    cases = {
        "https://user:pass@example.com/": "invalid_url",
        "https://example.com:8443/": "unsafe_port",
        "ftp://example.com/file": "unsupported_scheme",
        "hhttps://example.com/": "unsupported_scheme",
        "file:///etc/passwd": "unsupported_scheme",
        "https:///nohost": "invalid_url",
    }
    for url, code in cases.items():
        with pytest.raises(NetguardError) as error:
            transport.validate_url(url)
        assert error.value.code == code, url


def test_dns_failure(tmp_path):
    config = make_config(tmp_path)

    def resolver(*args, **kwargs):
        raise socket.gaierror("no such host")

    transport = Transport(config, resolver=resolver)
    with pytest.raises(NetguardError) as error:
        transport.validate_url("https://missing.example/")
    assert error.value.code == "dns_failure"


def test_redirect_hop_is_revalidated(tmp_path):
    config = make_config(tmp_path)
    responses = {
        "https://public.example/start": RawResponse(
            status=302, headers={"location": "https://127.0.0.1/secret"}
        )
    }
    transport = StubTransport(
        config,
        stub_resolver(
            {"public.example": "93.184.216.34", "127.0.0.1": "127.0.0.1"}
        ),
        responses,
    )
    with pytest.raises(NetguardError) as error:
        transport.request("https://public.example/start", method="HEAD")
    assert error.value.code == "ssrf_blocked"


def test_https_to_http_downgrade_is_refused(tmp_path):
    config = make_config(tmp_path)
    responses = {
        "https://public.example/start": RawResponse(
            status=302, headers={"location": "http://other.example/next"}
        )
    }
    transport = StubTransport(
        config,
        stub_resolver(
            {"public.example": "93.184.216.34", "other.example": "93.184.216.35"}
        ),
        responses,
    )
    with pytest.raises(NetguardError) as error:
        transport.request("https://public.example/start", method="HEAD")
    assert error.value.code == "unsafe_redirect"


def test_dns_rebinding_between_hops_is_blocked(tmp_path):
    config = make_config(tmp_path)
    answers = iter(["93.184.216.34", "10.0.0.5"])

    def resolver(host, port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), 0))
        ]

    responses = {
        "https://rebind.example/start": RawResponse(
            status=302, headers={"location": "https://rebind.example/next"}
        )
    }
    transport = StubTransport(config, resolver, responses)
    with pytest.raises(NetguardError) as error:
        transport.request("https://rebind.example/start", method="HEAD")
    assert error.value.code == "ssrf_blocked"


def test_redirect_limit_and_missing_location(tmp_path):
    config = make_config(tmp_path)
    resolver = stub_resolver({"public.example": "93.184.216.34"})
    responses = {
        f"https://public.example/{index}": RawResponse(
            status=302, headers={"location": f"https://public.example/{index + 1}"}
        )
        for index in range(5)
    }
    transport = StubTransport(config, resolver, responses)
    with pytest.raises(NetguardError) as error:
        transport.request("https://public.example/0", method="HEAD")
    assert error.value.code == "too_many_redirects"

    missing = StubTransport(
        config,
        resolver,
        {"https://public.example/x": RawResponse(status=302, headers={})},
    )
    with pytest.raises(NetguardError) as error:
        missing.request("https://public.example/x", method="HEAD")
    assert error.value.code == "invalid_url"


def test_netguard_matches_existing_ingestion_guard(tmp_path, monkeypatch):
    from ingestion import ingestion_service

    config = make_config(tmp_path)
    addresses = [
        "127.0.0.1",
        "10.1.2.3",
        "169.254.169.254",
        "::1",
        "93.184.216.34",
        "8.8.8.8",
    ]
    for address in addresses:
        def fake_getaddrinfo(host, port, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))]

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        service_ok = True
        try:
            ingestion_service._global_address("example.com")
        except ingestion_service.IngestionError:
            service_ok = False
        transport = Transport(config)
        netguard_ok = True
        try:
            transport.validate_url("https://example.com/")
        except NetguardError:
            netguard_ok = False
        assert service_ok == netguard_ok, address

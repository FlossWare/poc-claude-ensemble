"""Hostile archive fixtures: traversal, links, bombs, encryption, nesting."""
from __future__ import annotations

import io
import zipfile

import pytest

from ingestion.datasets import archive

from ingestion.datasets.tests.helpers import (
    device,
    directory,
    hardlink,
    make_config,
    regular,
    symlink,
    tar_bytes,
)


def _zip_bytes(entries, *, compression=zipfile.ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as handle:
        for name, data, *rest in entries:
            info = zipfile.ZipInfo(name)
            info.compress_type = compression
            if rest:
                info.external_attr = rest[0]
            handle.writestr(info, data)
    return buffer.getvalue()


def _inspect(tmp_path, data: bytes, *, name="data.zip", config=None):
    config = config or make_config(tmp_path)
    path = tmp_path / "archive.bin"
    path.write_bytes(data)
    return archive.inspect(path, name=name, archive_format=None, config=config)


def test_zip_with_macosx_junk_member_is_inspected(tmp_path):
    data = _zip_bytes(
        [
            ("data/table.csv", b"a,b\n1,2\n"),
            ("__MACOSX/._table.csv", b"\x00\x05\x16\x07junk"),
        ]
    )
    inspection = _inspect(tmp_path, data)
    assert inspection.archive_format == "zip"
    assert inspection.junk == ("__MACOSX/._table.csv",)
    assert [member.name for member in inspection.data_members] == ["data/table.csv"]


def _patch_headers(data: bytes, offset_in_local: int, offset_in_central: int, mutate) -> bytes:
    """Apply ``mutate`` to a 2-byte field in every local and central header."""
    buffer = bytearray(data)
    for signature, field_offset in ((b"PK\x03\x04", offset_in_local), (b"PK\x01\x02", offset_in_central)):
        start = 0
        while True:
            index = buffer.find(signature, start)
            if index < 0:
                break
            field = index + field_offset
            buffer[field : field + 2] = mutate(bytes(buffer[field : field + 2]))
            start = index + 4
    return bytes(buffer)


@pytest.mark.parametrize(
    "name",
    [
        "../escape.csv",
        "/absolute.csv",
        "C:/drive.csv",
        "nested\\backslash.csv",
    ],
)
def test_zip_unsafe_member_paths_are_rejected(tmp_path, name):
    data = _zip_bytes([(name, b"a,b\n1,2\n")])
    with pytest.raises(archive.ArchiveError) as error:
        _inspect(tmp_path, data)
    assert error.value.code == "archive_unsafe_path"


def test_zip_member_with_a_nul_byte_never_reaches_the_validator(tmp_path):
    # ``zipfile`` truncates a name at the first NUL when reading, so the NUL
    # cannot reach ``Member.name``; the validator still refuses one if another
    # code path ever produces it (see ``test_member_name_validation_...``).
    data = _zip_bytes([("nulXbyte.csv", b"a,b\n1,2\n")])
    data = data.replace(b"nulXbyte.csv", b"nul\x00byte.csv")
    inspection = _inspect(tmp_path, data)
    assert inspection.members[0].name == "nul"
    assert all("\x00" not in member.name for member in inspection.members)


def test_zip_symlink_member_is_rejected(tmp_path):
    # 0o120777 is a unix symlink in the external attributes.
    data = _zip_bytes([("link.csv", b"/etc/passwd", (0o120777 << 16))])
    with pytest.raises(archive.ArchiveError) as error:
        _inspect(tmp_path, data)
    assert error.value.code == "archive_link_or_device"


def test_zip_encrypted_member_is_rejected(tmp_path):
    # ``writestr`` overwrites flag_bits, so the "encrypted" bit is patched into
    # the returned bytes instead.
    data = _zip_bytes([("secret.csv", b"a,b\n1,2\n")])
    encrypted = _patch_headers(
        data,
        6,
        8,
        lambda field: bytes([field[0] | 0x01, field[1]]),
    )
    with pytest.raises(archive.ArchiveError) as error:
        _inspect(tmp_path, encrypted)
    assert error.value.code == "archive_encrypted"


def test_zip_with_too_many_members_is_rejected(tmp_path):
    config = make_config(tmp_path).replace(archive_max_entries=2)
    data = _zip_bytes([(f"f{index}.csv", b"a\n") for index in range(3)])
    with pytest.raises(archive.ArchiveError) as error:
        _inspect(tmp_path, data, config=config)
    assert error.value.code == "archive_too_many_entries"


def test_zip_compression_ratio_bomb_is_rejected(tmp_path):
    config = make_config(tmp_path).replace(archive_max_ratio=10)
    data = _zip_bytes([("bomb.csv", b"a" * 1_000_000)])
    with pytest.raises(archive.ArchiveError) as error:
        _inspect(tmp_path, data, config=config)
    assert error.value.code == "archive_bomb"


def test_zip_uncompressed_cap_is_enforced(tmp_path):
    config = make_config(tmp_path).replace(
        archive_max_uncompressed=1024, archive_max_ratio=1_000_000
    )
    data = _zip_bytes([("big.csv", b"a" * 4096)], compression=zipfile.ZIP_STORED)
    with pytest.raises(archive.ArchiveError) as error:
        _inspect(tmp_path, data, config=config)
    assert error.value.code == "archive_bomb"


def test_truncated_zip_is_reported_as_parse_error(tmp_path):
    data = _zip_bytes([("a.csv", b"a,b\n1,2\n")])
    with pytest.raises(archive.ArchiveError) as error:
        _inspect(tmp_path, data[: len(data) // 2])
    assert error.value.code == "parse_error"


def test_tar_links_and_devices_are_rejected(tmp_path):
    for info in (symlink("root/link", "/etc/passwd"), hardlink("root/h", "root/t"), device("root/null")):
        data = tar_bytes([(directory("root/"), None), (info, None)])
        with pytest.raises(archive.ArchiveError) as error:
            _inspect(tmp_path, data, name="data.tar")
        assert error.value.code == "archive_link_or_device"


def test_tar_traversal_is_rejected(tmp_path):
    data = tar_bytes([(directory("root/"), None), (regular("root/../evil.csv", b"x"), b"x")])
    with pytest.raises(archive.ArchiveError) as error:
        _inspect(tmp_path, data, name="data.tar")
    assert error.value.code == "archive_unsafe_path"


def test_duplicate_member_names_are_rejected(tmp_path):
    data = _zip_bytes([("dup.csv", b"a\n"), ("dup.csv", b"b\n")])
    with pytest.raises(archive.ArchiveError) as error:
        _inspect(tmp_path, data)
    assert error.value.code == "archive_unsafe_path"


def test_nested_archive_member_is_not_extracted(tmp_path):
    inner = _zip_bytes([("inner.csv", b"a,b\n1,2\n")])
    outer = _zip_bytes([("bundle.zip", inner), ("data.csv", b"a,b\n1,2\n")])
    inspection = _inspect(tmp_path, outer)
    assert inspection.nested == ("bundle.zip",)
    assert [member.name for member in inspection.data_members] == ["data.csv"]


def test_archive_with_only_nested_archives_is_refused(tmp_path):
    inner = _zip_bytes([("inner.csv", b"a,b\n1,2\n")])
    outer = _zip_bytes([("bundle.zip", inner)])
    path = tmp_path / "outer.zip"
    path.write_bytes(outer)
    config = make_config(tmp_path)
    with pytest.raises(archive.ArchiveError) as error:
        with archive.extracted_members(
            path, name="outer.zip", archive_format=None, config=config
        ):
            pass
    assert error.value.code == "archive_nested"


def test_safe_members_are_extracted_under_the_temp_root(tmp_path):
    data = _zip_bytes(
        [("data/table.csv", b"a,b\n1,2\n"), ("data/other.tsv", b"a\tb\n1\t2\n")]
    )
    path = tmp_path / "bundle.zip"
    path.write_bytes(data)
    config = make_config(tmp_path)
    with archive.extracted_members(
        path, name="bundle.zip", archive_format=None, config=config
    ) as members:
        assert sorted(member.name for member in members) == [
            "data/other.tsv",
            "data/table.csv",
        ]
        for member in members:
            assert member.path.is_file()
            assert str(member.path).startswith(str(config.data_root))
            assert member.path.stat().st_mode & 0o777 == 0o600
        captured = [member.path for member in members]
    assert all(not path.exists() for path in captured)
    # The extraction root is gone after the block.
    assert list((config.data_root / "tmp").iterdir()) == []


def test_gzip_single_member_is_inspected_and_extracted(tmp_path):
    import gzip

    payload = b"a,b\n1,2\n"
    data = gzip.compress(payload)
    inspection = _inspect(tmp_path, data, name="table.csv.gz")
    assert inspection.archive_format == "gz"
    assert inspection.members[0].name == "table.csv"
    path = tmp_path / "table.csv.gz"
    path.write_bytes(data)
    config = make_config(tmp_path)
    with archive.extracted_members(
        path, name="table.csv.gz", archive_format=None, config=config
    ) as members:
        assert [member.name for member in members] == ["table.csv"]
        assert members[0].path.read_bytes() == payload


def test_format_detection_prefers_name_then_media_type():
    assert archive.archive_format_for("x.tar.gz") == "tar.gz"
    assert archive.archive_format_for("x.zip") == "zip"
    assert archive.archive_format_for(None, "application/zip") == "zip"
    assert archive.archive_format_for(None, None, b"PK\x03\x04rest") == "zip"
    assert archive.archive_format_for("x.csv", "text/csv", b"a,b\n") is None


def test_member_name_validation_matches_the_catalog_rules():
    assert archive.unsafe_member_reason("a/b.csv") is None
    assert archive.unsafe_member_reason("/a.csv") is not None
    assert archive.unsafe_member_reason("a/../../b") is not None


def test_gzip_without_a_tar_header_falls_back_to_a_single_member(tmp_path):
    import gzip

    payload = b"name,value\nalpha,1\n"
    path = tmp_path / "download.bin"
    path.write_bytes(gzip.compress(payload))
    config = make_config(tmp_path)
    # The name carries no .gz suffix, so the format comes from the gzip magic.
    inspection = archive.inspect(
        path, name="download", archive_format=None, config=config
    )
    assert inspection.archive_format == "gz"
    assert [member.name for member in inspection.data_members] == ["download"]
    with archive.extracted_members(
        path, name="download", archive_format=None, config=config
    ) as members:
        assert members[0].path.read_bytes() == payload


def test_gzip_tar_is_still_detected_by_its_ustar_header(tmp_path):
    data = tar_bytes([(directory("root/"), None), (regular("root/a.csv", b"a\n1\n"), b"a\n1\n")])
    path = tmp_path / "bundle.gz"
    path.write_bytes(data)
    config = make_config(tmp_path)
    inspection = archive.inspect(
        path, name="bundle.gz", archive_format=None, config=config
    )
    assert inspection.archive_format == "tar.gz"
    assert [member.name for member in inspection.data_members] == ["root/a.csv"]

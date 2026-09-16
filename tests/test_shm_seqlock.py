from __future__ import annotations

import uuid

import pytest

from vision_pipeline.runtime.shm_seqlock import SeqlockReader, SeqlockWriter


def _channel_name() -> str:
    return f"vision-pipeline-test-{uuid.uuid4().hex}"


def test_reader_sees_none_before_the_first_write() -> None:
    name = _channel_name()
    writer = SeqlockWriter(name, payload_size=8)
    try:
        reader = SeqlockReader(name, payload_size=8)
        try:
            assert reader.read() is None
        finally:
            reader.close()
    finally:
        writer.close()


def test_reader_sees_the_latest_write_and_the_sequence_advances() -> None:
    name = _channel_name()
    writer = SeqlockWriter(name, payload_size=4)
    try:
        reader = SeqlockReader(name, payload_size=4)
        try:
            writer.write(b"abcd")
            first = reader.read()
            assert first == (2, b"abcd")

            writer.write(b"efgh")
            second = reader.read()
            assert second == (4, b"efgh")
        finally:
            reader.close()
    finally:
        writer.close()


def test_write_rejects_a_payload_of_the_wrong_size() -> None:
    name = _channel_name()
    writer = SeqlockWriter(name, payload_size=8)
    try:
        with pytest.raises(ValueError, match="8"):
            writer.write(b"short")
    finally:
        writer.close()


def test_reader_rejects_a_channel_created_with_a_different_payload_size() -> None:
    name = _channel_name()
    writer = SeqlockWriter(name, payload_size=8)
    try:
        with pytest.raises(ValueError, match="expected"):
            SeqlockReader(name, payload_size=16)
    finally:
        writer.close()


def test_creating_over_a_stale_segment_reclaims_it() -> None:
    name = _channel_name()
    first = SeqlockWriter(name, payload_size=4)
    # Simulate a crash: the segment is left behind without close()/unlink().

    second = SeqlockWriter(name, payload_size=4)
    try:
        second.write(b"ok!!")
        reader = SeqlockReader(name, payload_size=4)
        try:
            assert reader.read() == (2, b"ok!!")
        finally:
            reader.close()
    finally:
        second.close()
    del first

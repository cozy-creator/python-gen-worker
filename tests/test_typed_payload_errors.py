"""Verify typed payload-size errors expose structured fields.

Issue #4 `typed-payload-errors`: callers should be able to write
`except OutputTooLargeError as e: ... e.size_bytes ...` without
regex-matching the message string.
"""


def test_output_too_large_typed():
    from gen_worker import OutputTooLargeError

    err = OutputTooLargeError(size_bytes=200, max_bytes=100)
    assert err.size_bytes == 200
    assert err.max_bytes == 100
    assert isinstance(err, Exception)


def test_input_too_large_typed():
    from gen_worker import InputTooLargeError

    err = InputTooLargeError(size_bytes=200, max_bytes=100, source="input file")
    assert err.size_bytes == 200
    assert err.max_bytes == 100
    assert err.source == "input file"
    assert isinstance(err, Exception)


def test_input_too_large_default_source():
    from gen_worker import InputTooLargeError

    err = InputTooLargeError(size_bytes=10, max_bytes=5)
    assert err.source == "input"


def test_typed_errors_subclass_validation_error():
    from gen_worker import (
        InputTooLargeError,
        OutputTooLargeError,
        ValidationError,
    )

    assert issubclass(OutputTooLargeError, ValidationError)
    assert issubclass(InputTooLargeError, ValidationError)

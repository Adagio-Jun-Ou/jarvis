"""Tests for file upload security validation."""
import pytest

from core.security.upload_validation import (
    validate_upload,
    sanitize_filename,
    validate_filename,
    validate_mime_type,
    validate_extension_for_mime,
    validate_magic_bytes,
    get_file_category,
    get_max_file_size,
    UploadValidationResult,
    ALLOWED_IMAGE_TYPES,
    ALL_ALLOWED_TYPES,
)


class TestSanitizeFilename:
    def test_normal_filename(self):
        assert sanitize_filename("photo.jpg") == "photo.jpg"

    def test_path_traversal_removed(self):
        result = sanitize_filename("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_backslash_removed(self):
        result = sanitize_filename("folder\\file.txt")
        assert "\\" not in result

    def test_special_chars_replaced(self):
        result = sanitize_filename("file@#$%.txt")
        assert "@" not in result
        assert "#" not in result

    def test_empty_filename(self):
        assert sanitize_filename("") == "unnamed_file"

    def test_long_filename_truncated(self):
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name)
        assert len(result) <= 200

    def test_dots_only(self):
        result = sanitize_filename("...")
        assert result == "unnamed_file"

    def test_double_underscore_collapse(self):
        result = sanitize_filename("a___b.txt")
        assert "___" not in result


class TestValidateFilename:
    def test_valid_filename(self):
        assert validate_filename("photo.jpg") is True

    def test_path_traversal(self):
        assert validate_filename("../../etc/passwd") is False

    def test_backslash(self):
        assert validate_filename("folder\\file.txt") is False

    def test_dangerous_chars(self):
        assert validate_filename('file<>:"|?.txt') is False

    def test_empty_filename(self):
        assert validate_filename("") is False

    def test_windows_reserved(self):
        assert validate_filename("CON") is False
        assert validate_filename("PRN") is False
        assert validate_filename("AUX") is False

    def test_normal_filenames(self):
        assert validate_filename("data.csv") is True
        assert validate_filename("report.pdf") is True
        assert validate_filename("image_01.png") is True


class TestValidateMimeType:
    def test_valid_image_types(self):
        for mt in ALLOWED_IMAGE_TYPES:
            assert validate_mime_type(mt) is True

    def test_invalid_mime(self):
        assert validate_mime_type("application/x-executable") is False
        assert validate_mime_type("text/html") is False
        assert validate_mime_type("application/octet-stream") is False

    def test_empty_mime(self):
        assert validate_mime_type("") is False


class TestValidateExtensionForMime:
    def test_jpeg_extensions(self):
        assert validate_extension_for_mime("image/jpeg", ".jpg") is True
        assert validate_extension_for_mime("image/jpeg", ".jpeg") is True

    def test_png_extension(self):
        assert validate_extension_for_mime("image/png", ".png") is True
        assert validate_extension_for_mime("image/png", ".jpg") is False

    def test_extension_without_dot(self):
        assert validate_extension_for_mime("image/jpeg", "jpg") is True

    def test_invalid_mime_type(self):
        assert validate_extension_for_mime("application/x-executable", ".exe") is False


class TestValidateMagicBytes:
    def test_jpeg_magic(self):
        assert validate_magic_bytes(b"\xff\xd8\xff\xe0", "image/jpeg") is True

    def test_png_magic(self):
        assert validate_magic_bytes(b"\x89PNG\r\n\x1a\n", "image/png") is True

    def test_gif87a_magic(self):
        assert validate_magic_bytes(b"GIF87a", "image/gif") is True

    def test_gif89a_magic(self):
        assert validate_magic_bytes(b"GIF89a", "image/gif") is True

    def test_pdf_magic(self):
        assert validate_magic_bytes(b"%PDF-1.4", "application/pdf") is True

    def test_wrong_magic(self):
        assert validate_magic_bytes(b"\x89PNG\r\n\x1a\n", "image/jpeg") is False

    def test_unknown_mime_passes(self):
        assert validate_magic_bytes(b"anything", "text/plain") is True


class TestGetFileCategory:
    def test_image_category(self):
        assert get_file_category("image/jpeg") == "image"

    def test_document_category(self):
        assert get_file_category("application/pdf") == "document"

    def test_audio_category(self):
        assert get_file_category("audio/wav") == "audio"

    def test_video_category(self):
        assert get_file_category("video/mp4") == "video"

    def test_unknown_category(self):
        assert get_file_category("application/x-unknown") == "unknown"


class TestGetMaxFileSize:
    def test_image_max(self):
        assert get_max_file_size("image/jpeg") == 20 * 1024 * 1024

    def test_document_max(self):
        assert get_max_file_size("application/pdf") == 50 * 1024 * 1024

    def test_unknown_max(self):
        assert get_max_file_size("application/x-unknown") == 10 * 1024 * 1024


class TestValidateUpload:
    def test_valid_jpeg_upload(self):
        result = validate_upload(
            filename="photo.jpg",
            mime_type="image/jpeg",
            file_size=1024,
            file_data=b"\xff\xd8\xff\xe0" + b"\x00" * 60,
        )
        assert result.valid is True
        assert result.result == UploadValidationResult.VALID
        assert result.category == "image"

    def test_valid_png_upload(self):
        result = validate_upload(
            filename="image.png",
            mime_type="image/png",
            file_size=5000,
            file_data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 56,
        )
        assert result.valid is True

    def test_invalid_mime_type(self):
        result = validate_upload(
            filename="evil.exe",
            mime_type="application/x-executable",
            file_size=1024,
        )
        assert result.valid is False
        assert result.result == UploadValidationResult.INVALID_MIME

    def test_mime_extension_mismatch(self):
        result = validate_upload(
            filename="image.png",
            mime_type="image/jpeg",
            file_size=1024,
        )
        assert result.valid is False
        assert result.result == UploadValidationResult.MIME_EXTENSION_MISMATCH

    def test_file_too_large(self):
        result = validate_upload(
            filename="huge.jpg",
            mime_type="image/jpeg",
            file_size=100 * 1024 * 1024,
        )
        assert result.valid is False
        assert result.result == UploadValidationResult.FILE_TOO_LARGE

    def test_dangerous_filename(self):
        result = validate_upload(
            filename="../../etc/passwd",
            mime_type="image/jpeg",
            file_size=1024,
        )
        assert result.valid is False
        assert result.result == UploadValidationResult.DANGEROUS_FILENAME

    def test_magic_byte_mismatch(self):
        result = validate_upload(
            filename="fake.jpg",
            mime_type="image/jpeg",
            file_size=1024,
            file_data=b"NOT_A_JPEG_FILE_HEADER_DATA_HERE_FAKE",
        )
        assert result.valid is False
        assert result.result == UploadValidationResult.MAGIC_BYTE_MISMATCH

    def test_custom_allowed_types(self):
        result = validate_upload(
            filename="doc.pdf",
            mime_type="application/pdf",
            file_size=1024,
            allowed_types={"image/jpeg": {".jpg"}},
        )
        assert result.valid is False
        assert result.result == UploadValidationResult.INVALID_MIME

    def test_no_file_data_skips_magic_check(self):
        result = validate_upload(
            filename="photo.jpg",
            mime_type="image/jpeg",
            file_size=1024,
            file_data=None,
        )
        assert result.valid is True

    def test_to_dict(self):
        result = validate_upload(
            filename="photo.jpg",
            mime_type="image/jpeg",
            file_size=1024,
        )
        d = result.to_dict()
        assert "valid" in d
        assert "result" in d
        assert "mime_type" in d

    def test_valid_pdf_upload(self):
        result = validate_upload(
            filename="report.pdf",
            mime_type="application/pdf",
            file_size=50000,
            file_data=b"%PDF-1.4 fake content for test",
        )
        assert result.valid is True
        assert result.category == "document"

    def test_valid_wav_upload(self):
        result = validate_upload(
            filename="audio.wav",
            mime_type="audio/wav",
            file_size=60000,
            file_data=b"RIFF" + b"\x00" * 60,
        )
        assert result.valid is True
        assert result.category == "audio"

    def test_svg_upload_no_magic(self):
        result = validate_upload(
            filename="diagram.svg",
            mime_type="image/svg+xml",
            file_size=2000,
            file_data=b"<?xml version=",
        )
        assert result.valid is True
"""File Upload Security for Jarvis.

Validates uploaded files against:
1. Allowed MIME types (whitelist-based)
2. File extension consistency with MIME type
3. File size limits per type
4. Filename sanitization (path traversal, special chars)
5. Magic byte verification for image types
"""
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {
    "image/jpeg": {".jpg", ".jpeg"},
    "image/png": {".png"},
    "image/gif": {".gif"},
    "image/webp": {".webp"},
    "image/svg+xml": {".svg"},
    "image/bmp": {".bmp"},
    "image/tiff": {".tiff", ".tif"},
}

ALLOWED_DOCUMENT_TYPES = {
    "application/pdf": {".pdf"},
    "text/plain": {".txt", ".md", ".csv", ".log"},
    "application/json": {".json"},
    "text/csv": {".csv"},
}

ALLOWED_AUDIO_TYPES = {
    "audio/wav": {".wav"},
    "audio/mpeg": {".mp3"},
    "audio/ogg": {".ogg"},
    "audio/webm": {".weba"},
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4": {".mp4"},
    "video/webm": {".webm"},
}

ALL_ALLOWED_TYPES = {
    **ALLOWED_IMAGE_TYPES,
    **ALLOWED_DOCUMENT_TYPES,
    **ALLOWED_AUDIO_TYPES,
    **ALLOWED_VIDEO_TYPES,
}

MAX_FILE_SIZES = {
    "image": 20 * 1024 * 1024,
    "document": 50 * 1024 * 1024,
    "audio": 100 * 1024 * 1024,
    "video": 500 * 1024 * 1024,
    "default": 10 * 1024 * 1024,
}

MAGIC_BYTES = {
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/gif": [b"GIF87a", b"GIF89a"],
    "image/bmp": [b"BM"],
    "image/webp": [b"RIFF"],
    "application/pdf": [b"%PDF"],
    "audio/wav": [b"RIFF"],
    "audio/mpeg": [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"ID3"],
}


class UploadValidationResult(Enum):
    VALID = "valid"
    INVALID_MIME = "invalid_mime"
    INVALID_EXTENSION = "invalid_extension"
    MIME_EXTENSION_MISMATCH = "mime_extension_mismatch"
    FILE_TOO_LARGE = "file_too_large"
    DANGEROUS_FILENAME = "dangerous_filename"
    MAGIC_BYTE_MISMATCH = "magic_byte_mismatch"


@dataclass
class UploadValidation:
    valid: bool
    result: UploadValidationResult
    mime_type: str = ""
    extension: str = ""
    category: str = "unknown"
    max_size: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "result": self.result.value,
            "mime_type": self.mime_type,
            "extension": self.extension,
            "category": self.category,
            "reason": self.reason,
        }


def sanitize_filename(filename: str) -> str:
    if not filename:
        return "unnamed_file"
    filename = filename.split("/")[-1].split("\\")[-1]
    filename = re.sub(r'[^\w\.\-]', '_', filename)
    filename = re.sub(r'_+', '_', filename)
    filename = filename.strip('_.')
    if not filename or filename == '.':
        return "unnamed_file"
    max_len = 200
    if len(filename) > max_len:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = f"{name[:max_len - len(ext) - 1]}.{ext}" if ext else filename[:max_len]
    return filename


def get_file_category(mime_type: str) -> str:
    if mime_type in ALLOWED_IMAGE_TYPES:
        return "image"
    elif mime_type in ALLOWED_DOCUMENT_TYPES:
        return "document"
    elif mime_type in ALLOWED_AUDIO_TYPES:
        return "audio"
    elif mime_type in ALLOWED_VIDEO_TYPES:
        return "video"
    return "unknown"


def get_max_file_size(mime_type: str) -> int:
    category = get_file_category(mime_type)
    return MAX_FILE_SIZES.get(category, MAX_FILE_SIZES["default"])


def validate_mime_type(mime_type: str) -> bool:
    return mime_type in ALL_ALLOWED_TYPES


def validate_extension_for_mime(mime_type: str, extension: str) -> bool:
    if mime_type not in ALL_ALLOWED_TYPES:
        return False
    ext = extension.lower()
    if not ext.startswith('.'):
        ext = f".{ext}"
    return ext in ALL_ALLOWED_TYPES[mime_type]


def validate_magic_bytes(data: bytes, mime_type: str) -> bool:
    if mime_type not in MAGIC_BYTES:
        return True
    magic_list = MAGIC_BYTES[mime_type]
    return any(data.startswith(magic) for magic in magic_list)


def validate_filename(filename: str) -> bool:
    if not filename:
        return False
    dangerous_patterns = [
        r'\.\.', r'[\\]', r'[/]$', r'^[/]',
        r'[<>:"|?*]', r'^(CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])$',
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, filename, re.IGNORECASE):
            return False
    return True


def validate_upload(
    filename: str,
    mime_type: str,
    file_size: int,
    file_data: Optional[bytes] = None,
    allowed_types: Optional[dict] = None,
) -> UploadValidation:
    types_to_check = allowed_types or ALL_ALLOWED_TYPES

    if not validate_filename(filename):
        return UploadValidation(
            valid=False,
            result=UploadValidationResult.DANGEROUS_FILENAME,
            reason=f"Filename contains dangerous characters or patterns",
        )

    if not mime_type or mime_type not in types_to_check:
        return UploadValidation(
            valid=False,
            result=UploadValidationResult.INVALID_MIME,
            mime_type=mime_type,
            reason=f"MIME type '{mime_type}' is not allowed. Allowed: {list(types_to_check.keys())}",
        )

    ext = ""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()

    if ext and not validate_extension_for_mime(mime_type, ext):
        return UploadValidation(
            valid=False,
            result=UploadValidationResult.MIME_EXTENSION_MISMATCH,
            mime_type=mime_type,
            extension=ext,
            reason=f"Extension '{ext}' does not match MIME type '{mime_type}'",
        )

    max_size = get_max_file_size(mime_type)
    if file_size > max_size:
        return UploadValidation(
            valid=False,
            result=UploadValidationResult.FILE_TOO_LARGE,
            mime_type=mime_type,
            max_size=max_size,
            reason=f"File size {file_size} exceeds maximum {max_size} for category '{get_file_category(mime_type)}'",
        )

    if file_data and not validate_magic_bytes(file_data, mime_type):
        return UploadValidation(
            valid=False,
            result=UploadValidationResult.MAGIC_BYTE_MISMATCH,
            mime_type=mime_type,
            reason=f"File content does not match expected format for '{mime_type}'",
        )

    return UploadValidation(
        valid=True,
        result=UploadValidationResult.VALID,
        mime_type=mime_type,
        extension=ext,
        category=get_file_category(mime_type),
        max_size=max_size,
    )
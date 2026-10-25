"""Stack Bug Defense for Jarvis.

Provides defense against stack-based vulnerabilities including:
- Buffer overflow protection
- Stack canary validation
- Input validation and sanitization
- Memory safety checks
- Format string attack prevention
- Integer overflow detection
- Use-after-free prevention
"""

import re
import struct
from core.helpers import atomic_write_json, safe_error_message
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class VulnerabilityType(str, Enum):
    """Types of stack vulnerabilities."""
    BUFFER_OVERFLOW = "buffer_overflow"
    INTEGER_OVERFLOW = "integer_overflow"
    FORMAT_STRING = "format_string"
    USE_AFTER_FREE = "use_after_free"
    NULL_POINTER = "null_pointer"
    OUT_OF_BOUNDS = "out_of_bounds"
    TYPE_CONFUSION = "type_confusion"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    FORMAT_SPECIFIER = "format_specifier"


class DefenseAction(str, Enum):
    """Actions taken when vulnerability detected."""
    LOGGED = "logged"
    SANITIZED = "sanitized"
    BLOCKED = "blocked"
    QUARANTINED = "quarantined"


@dataclass
class SecurityViolation:
    """Represents a detected security violation."""
    timestamp: datetime
    vulnerability_type: VulnerabilityType
    description: str
    input_data: str  # Sanitized input snippet
    action: DefenseAction
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "vulnerability_type": self.vulnerability_type.value,
            "description": self.description,
            "input_data": self.input_data[:100],  # Truncate for safety
            "action": self.action.value,
            "context": self.context,
        }


class SafeBuffer:
    """A safe buffer implementation with bounds checking."""

    def __init__(self, size: int, canary: bool = True):
        self._size = size
        self._enable_canary = canary
        self._buffer = bytearray(size)

        self._canary_value = self._generate_canary() if canary else 0
        self._canary_position = 0
        if canary and size >= 8:
            self._canary_position = size - 8
            canary_bytes = struct.pack("Q", self._canary_value)
            self._buffer[self._canary_position:self._canary_position + 8] = canary_bytes

    def _generate_canary(self) -> int:
        """Generate a random canary value."""
        import os
        return struct.unpack("Q", os.urandom(8))[0]

    def write(self, data: bytes, offset: int = 0) -> bool:
        if offset < 0:
            return False

        end_pos = offset + len(data)
        if end_pos > self._size:
            return False

        if self._enable_canary and self._canary_position > 0:
            if offset < self._canary_position + 8 and end_pos > self._canary_position:
                return False

        self._buffer[offset:offset + len(data)] = data
        return True

    def read(self, offset: int = 0, size: int | None = None) -> bytes | None:
        """Read data from buffer with bounds checking.

        Args:
            offset: Offset to read from
            size: Number of bytes to read (default: rest of buffer)

        Returns:
            Bytes read, or None if bounds violation
        """
        if offset < 0 or offset >= self._size:
            return None

        read_end = offset + (size if size is not None else self._size - offset)
        if self._enable_canary and self._canary_position > 0:
            if offset < self._canary_position + 8 and read_end > self._canary_position:
                return None

        if size is None:
            return bytes(self._buffer[offset:])

        if offset + size > self._size:
            return None

        return bytes(self._buffer[offset:offset + size])

    def check_canary(self) -> bool:
        if not self._enable_canary:
            return True
        if self._canary_position <= 0 or self._size < self._canary_position + 8:
            return True
        stored_canary = struct.unpack("Q", bytes(self._buffer[self._canary_position:self._canary_position + 8]))[0]
        return stored_canary == self._canary_value

    def clear(self) -> None:
        """Securely clear the buffer."""
        # Overwrite with zeros
        for i in range(self._size):
            self._buffer[i] = 0

    @property
    def size(self) -> int:
        """Get buffer size."""
        return self._size

    def __len__(self) -> int:
        return self._size


class InputValidator:
    """Comprehensive input validation and sanitization."""

    # Dangerous patterns for various attack types
    DANGEROUS_PATTERNS = {
        # Command injection
        VulnerabilityType.COMMAND_INJECTION: [
            re.compile(r';\s*(?:rm|chmod|chown|wget|curl|nc|bash|sh|python|perl|ruby|node)\b', re.IGNORECASE),
            re.compile(r'\|\s*(?:rm|chmod|chown|wget|curl|nc|bash|sh|python|perl|ruby|node)\b', re.IGNORECASE),
            re.compile(r'\$\([^)]*\)'),
            re.compile(r'`[^`]+`'),
            re.compile(r'&&\s*(?:rm|chmod|chown|wget|curl|nc|bash|sh)\b', re.IGNORECASE),
            re.compile(r'>\s*/(?:dev|etc|proc|sys|tmp)\b'),
            re.compile(r'<\s*(?:/dev/tcp|/etc/passwd|/proc/self)\b'),
        ],
        # Path traversal
        VulnerabilityType.PATH_TRAVERSAL: [
            re.compile(r'\.\.\/'),
            re.compile(r'\.\.\\'),
            re.compile(r'%2e%2e%2f', re.IGNORECASE),
            re.compile(r'%2e%2e/', re.IGNORECASE),
            re.compile(r'\.\.%2f', re.IGNORECASE),
            re.compile(r'%252e%252e%252f', re.IGNORECASE),
        ],
        # Format string
        VulnerabilityType.FORMAT_STRING: [
            re.compile(r'%[sdxn]'),
            re.compile(r'%[0-9]+\$[sdxn]'),
            re.compile(r'%\*[0-9]+\$[sdxn]'),
        ],
        # Buffer overflow indicators
            VulnerabilityType.BUFFER_OVERFLOW: [
            re.compile(r'[Aa]{100,}'),
            re.compile(r'%n%n%n%n'),
            re.compile(r'(?:\\x[0-9a-fA-F]{2}){10,}'),
        ],
        # Null byte injection
        VulnerabilityType.NULL_POINTER: [
            re.compile(r'%00'),
            re.compile(r'\x00'),
        ],
    }

    # Maximum safe lengths for various input types
    MAX_LENGTHS = {
        "string": 10000,
        "command": 1000,
        "path": 4096,
        "url": 2048,
        "email": 256,
        "filename": 255,
        "identifier": 128,
    }

    def __init__(
        self,
        strict_mode: bool = True,
        max_length: int = 10000,
        audit_callback: Callable[[SecurityViolation], None] | None = None,
    ):
        self.strict_mode = strict_mode
        self.max_length = max_length
        self.audit_callback = audit_callback
        self._violations: list[SecurityViolation] = []
        self._max_violations = 1000

    def _add_violation(self, violation: SecurityViolation) -> None:
        """Add a violation with capacity enforcement."""
        self._violations.append(violation)
        if len(self._violations) > self._max_violations:
            self._violations = self._violations[-self._max_violations:]
        if self.audit_callback:
            self.audit_callback(violation)

    def validate_string(
        self,
        input_data: str,
        min_length: int = 0,
        max_length: int | None = None,
        allowed_chars: str | None = None,
    ) -> tuple[bool, str, list[SecurityViolation]]:
        """Validate and sanitize a string input.

        Args:
            input_data: Input string to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            allowed_chars: Regex pattern of allowed characters

        Returns:
            Tuple of (is_valid, sanitized_data, violations)
        """
        violations = []

        # Check length
        if max_length is None:
            max_length = self.max_length

        if len(input_data) < min_length:
            violations.append(SecurityViolation(
                timestamp=datetime.now(),
                vulnerability_type=VulnerabilityType.OUT_OF_BOUNDS,
                description=f"Input too short: {len(input_data)} < {min_length}",
                input_data=input_data[:50],
                action=DefenseAction.BLOCKED,
            ))

        if len(input_data) > max_length:
            violations.append(SecurityViolation(
                timestamp=datetime.now(),
                vulnerability_type=VulnerabilityType.BUFFER_OVERFLOW,
                description=f"Input too long: {len(input_data)} > {max_length}",
                input_data=input_data[:50],
                action=DefenseAction.SANITIZED if not self.strict_mode else DefenseAction.BLOCKED,
            ))
            if self.strict_mode:
                return False, "", violations
            input_data = input_data[:max_length]

        for vuln_type, patterns in self.DANGEROUS_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(input_data)
                if match:
                    violations.append(SecurityViolation(
                        timestamp=datetime.now(),
                        vulnerability_type=vuln_type,
                        description=f"Dangerous pattern detected: {vuln_type.value}",
                        input_data=match.group()[:50],
                        action=DefenseAction.SANITIZED if not self.strict_mode else DefenseAction.BLOCKED,
                    ))

        if allowed_chars:
            escaped = re.escape(allowed_chars)
            try:
                if not re.match(f"^[{escaped}]*$", input_data):
                    violations.append(SecurityViolation(
                        timestamp=datetime.now(),
                        vulnerability_type=VulnerabilityType.OUT_OF_BOUNDS,
                        description="Invalid characters in input",
                        input_data=input_data[:50],
                        action=DefenseAction.SANITIZED if not self.strict_mode else DefenseAction.BLOCKED,
                    ))
            except re.error:
                violations.append(SecurityViolation(
                    timestamp=datetime.now(),
                    vulnerability_type=VulnerabilityType.OUT_OF_BOUNDS,
                    description="Invalid allowed_chars pattern",
                    input_data=input_data[:50],
                    action=DefenseAction.BLOCKED,
                ))

        # Sanitize if not in strict mode
        sanitized = input_data
        if not self.strict_mode:
            sanitized = self._sanitize_input(input_data, violations)

        # Record violations
        for v in violations:
            self._add_violation(v)
            

        is_valid = len([v for v in violations if v.action == DefenseAction.BLOCKED]) == 0
        return is_valid, sanitized, violations

    def validate_path(
        self,
        path: str,
        base_dir: str | None = None,
        allow_absolute: bool = False,
    ) -> tuple[bool, str, list[SecurityViolation]]:
        """Validate a file path for security.

        Args:
            path: Path to validate
            base_dir: Base directory for relative paths
            allow_absolute: Allow absolute paths

        Returns:
            Tuple of (is_valid, sanitized_path, violations)
        """
        violations = []
        sanitized = path

        # Check for path traversal
        traversal_patterns = self.DANGEROUS_PATTERNS[VulnerabilityType.PATH_TRAVERSAL]
        for pattern in traversal_patterns:
            match = pattern.search(path)
            if match:
                violations.append(SecurityViolation(
                    timestamp=datetime.now(),
                    vulnerability_type=VulnerabilityType.PATH_TRAVERSAL,
                    description="Path traversal attempt detected",
                    input_data=match.group()[:50],
                    action=DefenseAction.BLOCKED,
                ))
                self._add_violation(violations[-1])
                return False, "", violations

        # Check for null bytes
        if "\x00" in path or "%00" in path:
            violations.append(SecurityViolation(
                timestamp=datetime.now(),
                vulnerability_type=VulnerabilityType.NULL_POINTER,
                description="Null byte in path",
                input_data=path[:50],
                action=DefenseAction.BLOCKED,
            ))
            self._add_violation(violations[-1])
            
            return False, "", violations

        from pathlib import Path as PathLib
        if not allow_absolute and PathLib(path).is_absolute():
            violations.append(SecurityViolation(
                timestamp=datetime.now(),
                vulnerability_type=VulnerabilityType.OUT_OF_BOUNDS,
                description="Absolute path not allowed",
                input_data=path[:50],
                action=DefenseAction.BLOCKED if self.strict_mode else DefenseAction.SANITIZED,
            ))
            if self.strict_mode:
                self._add_violation(violations[-1])
                return False, "", violations

        # Normalize path
        try:
            normalized = PathLib(path).resolve()
            if base_dir:
                base = PathLib(base_dir).resolve()
                try:
                    normalized.relative_to(base)
                except ValueError:
                    violations.append(SecurityViolation(
                        timestamp=datetime.now(),
                        vulnerability_type=VulnerabilityType.PATH_TRAVERSAL,
                        description="Path escapes base directory",
                        input_data=path[:50],
                        action=DefenseAction.BLOCKED,
                    ))
                    self._add_violation(violations[-1])
                    return False, "", violations

            sanitized = str(normalized)

        except Exception as e:
            violations.append(SecurityViolation(
                timestamp=datetime.now(),
                vulnerability_type=VulnerabilityType.OUT_OF_BOUNDS,
                description=f"Invalid path: {safe_error_message(e)}",
                input_data=path[:50],
                action=DefenseAction.BLOCKED,
            ))
            self._add_violation(violations[-1])
            
            return False, "", violations

        is_valid = len([v for v in violations if v.action == DefenseAction.BLOCKED]) == 0
        return is_valid, sanitized, violations

    def validate_command(
        self,
        command: str,
        allowed_commands: set[str] | None = None,
        blocked_commands: set[str] | None = None,
    ) -> tuple[bool, str, list[SecurityViolation]]:
        """Validate a shell command for security.

        Args:
            command: Command to validate
            allowed_commands: Set of allowed command names (if set, only these are allowed)
            blocked_commands: Set of blocked command names

        Returns:
            Tuple of (is_valid, sanitized_command, violations)
        """
        violations = []

        injection_patterns = self.DANGEROUS_PATTERNS[VulnerabilityType.COMMAND_INJECTION]
        for pattern in injection_patterns:
            match = pattern.search(command)
            if match:
                violations.append(SecurityViolation(
                    timestamp=datetime.now(),
                    vulnerability_type=VulnerabilityType.COMMAND_INJECTION,
                    description="Command injection pattern detected",
                    input_data=match.group()[:50],
                    action=DefenseAction.BLOCKED,
                ))

        # Parse command name
        command_parts = command.split()
        if command_parts:
            cmd_name = command_parts[0].split("/")[-1]  # Get base command name

            if allowed_commands and cmd_name not in allowed_commands:
                violations.append(SecurityViolation(
                    timestamp=datetime.now(),
                    vulnerability_type=VulnerabilityType.COMMAND_INJECTION,
                    description=f"Command not in allowed list: {cmd_name}",
                    input_data=cmd_name[:50],
                    action=DefenseAction.BLOCKED,
                ))

            if blocked_commands and cmd_name in blocked_commands:
                violations.append(SecurityViolation(
                    timestamp=datetime.now(),
                    vulnerability_type=VulnerabilityType.COMMAND_INJECTION,
                    description=f"Blocked command: {cmd_name}",
                    input_data=cmd_name[:50],
                    action=DefenseAction.BLOCKED,
                ))

        # Record violations
        for v in violations:
            self._add_violation(v)
            

        is_valid = len([v for v in violations if v.action == DefenseAction.BLOCKED]) == 0
        sanitized = self._sanitize_command(command) if not self.strict_mode else command
        return is_valid, sanitized, violations

    def validate_integer(
        self,
        value: int | str,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> tuple[bool, int, list[SecurityViolation]]:
        """Validate an integer value.

        Args:
            value: Integer or string to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Tuple of (is_valid, value, violations)
        """
        violations = []

        # Convert to int
        try:
            if isinstance(value, str):
                # Check for format specifiers
                if "%" in value:
                    violations.append(SecurityViolation(
                        timestamp=datetime.now(),
                        vulnerability_type=VulnerabilityType.FORMAT_SPECIFIER,
                        description="Format specifier in integer input",
                        input_data=value[:50],
                        action=DefenseAction.BLOCKED,
                    ))
                    self._add_violation(violations[-1])
                    return False, 0, violations

                int_value = int(value, 10)  # Force base 10
            else:
                int_value = value
        except ValueError:
            violations.append(SecurityViolation(
                timestamp=datetime.now(),
                vulnerability_type=VulnerabilityType.TYPE_CONFUSION,
                description="Invalid integer value",
                input_data=str(value)[:50],
                action=DefenseAction.BLOCKED,
            ))
            self._add_violation(violations[-1])
            
            return False, 0, violations

        # Check bounds
        if min_value is not None and int_value < min_value:
            violations.append(SecurityViolation(
                timestamp=datetime.now(),
                vulnerability_type=VulnerabilityType.INTEGER_OVERFLOW,
                description=f"Integer underflow: {int_value} < {min_value}",
                input_data=str(int_value),
                action=DefenseAction.SANITIZED if not self.strict_mode else DefenseAction.BLOCKED,
            ))
            if self.strict_mode:
                self._add_violation(violations[-1])
                return False, 0, violations
            int_value = min_value

        if max_value is not None and int_value > max_value:
            violations.append(SecurityViolation(
                timestamp=datetime.now(),
                vulnerability_type=VulnerabilityType.INTEGER_OVERFLOW,
                description=f"Integer overflow: {int_value} > {max_value}",
                input_data=str(int_value),
                action=DefenseAction.SANITIZED if not self.strict_mode else DefenseAction.BLOCKED,
            ))
            if self.strict_mode:
                self._add_violation(violations[-1])
                return False, 0, violations
            int_value = max_value

        # Record violations
        for v in violations:
            self._add_violation(v)
            

        is_valid = len([v for v in violations if v.action == DefenseAction.BLOCKED]) == 0
        return is_valid, int_value, violations

    def _sanitize_input(self, input_data: str, violations: list[SecurityViolation]) -> str:
        """Sanitize input by removing/escaping dangerous patterns."""
        sanitized = input_data

        # Remove null bytes
        sanitized = sanitized.replace("\x00", "")
        sanitized = sanitized.replace("%00", "")

        # Escape shell metacharacters
        shell_chars = [';', '|', '&', '`', '$', '(', ')', '<', '>']
        for char in shell_chars:
            sanitized = sanitized.replace(char, f"\\{char}")

        # Remove path traversal sequences
        sanitized = re.sub(r'\.\.\/', '', sanitized)
        sanitized = re.sub(r'\.\.\\', '', sanitized)

        return sanitized

    def _sanitize_command(self, command: str) -> str:
        """Sanitize a command string."""
        # Remove shell operators
        sanitized = re.sub(r'[;&|`$]', '', command)
        return sanitized

    def get_violations(
        self,
        vuln_type: VulnerabilityType | None = None,
        since: datetime | None = None,
    ) -> list[SecurityViolation]:
        """Get recorded violations."""
        violations = self._violations

        if vuln_type:
            violations = [v for v in violations if v.vulnerability_type == vuln_type]
        if since:
            violations = [v for v in violations if v.timestamp >= since]

        return violations


class StackDefense:
    """Main stack defense coordinator."""

    def __init__(
        self,
        storage_path: Path | None = None,
        strict_mode: bool = True,
        audit_callback: Callable[[SecurityViolation], None] | None = None,
    ):
        self.storage_path = storage_path
        self.strict_mode = strict_mode
        self.audit_callback = audit_callback

        self.validator = InputValidator(
            strict_mode=strict_mode,
            audit_callback=audit_callback,
        )

        # Memory tracking for use-after-free prevention
        self._allocated_memory: dict[int, dict[str, Any]] = {}
        self._freed_addresses: set[int] = set()
        self._next_allocation_id = 1

    def allocate_buffer(self, size: int) -> tuple[int, SafeBuffer]:
        """Allocate a safe buffer with tracking.

        Args:
            size: Buffer size

        Returns:
            Tuple of (allocation_id, buffer)
        """
        buffer = SafeBuffer(size, canary=True)
        alloc_id = self._next_allocation_id
        self._next_allocation_id += 1

        self._allocated_memory[alloc_id] = {
            "buffer": buffer,
            "size": size,
            "allocated_at": datetime.now(),
        }

        return alloc_id, buffer

    def free_buffer(self, allocation_id: int) -> bool:
        """Free a buffer with use-after-free protection.

        Args:
            allocation_id: Buffer allocation ID

        Returns:
            True if freed successfully
        """
        if allocation_id not in self._allocated_memory:
            return False

        # Clear the buffer securely
        buffer_info = self._allocated_memory[allocation_id]
        buffer_info["buffer"].clear()

        # Mark as freed
        self._freed_addresses.add(allocation_id)
        del self._allocated_memory[allocation_id]

        return True

    def check_use_after_free(self, allocation_id: int) -> SecurityViolation | None:
        """Check if an allocation ID is a use-after-free attempt.

        Args:
            allocation_id: Buffer allocation ID to check

        Returns:
            SecurityViolation if use-after-free detected
        """
        if allocation_id in self._freed_addresses:
            violation = SecurityViolation(
                timestamp=datetime.now(),
                vulnerability_type=VulnerabilityType.USE_AFTER_FREE,
                description=f"Use-after-free attempt on allocation {allocation_id}",
                input_data=str(allocation_id),
                action=DefenseAction.BLOCKED,
            )
            self.validator._add_violation(violation)
            
            return violation
        return None

    def validate_stack_frame(
        self,
        frame_data: bytes,
        expected_size: int,
        canary_value: int,
    ) -> tuple[bool, SecurityViolation | None]:
        """Validate a stack frame for integrity.

        Args:
            frame_data: Stack frame data
            expected_size: Expected size
            canary_value: Expected canary value

        Returns:
            Tuple of (is_valid, violation)
        """
        if len(frame_data) != expected_size:
            violation = SecurityViolation(
                timestamp=datetime.now(),
                vulnerability_type=VulnerabilityType.BUFFER_OVERFLOW,
                description=f"Stack frame size mismatch: {len(frame_data)} != {expected_size}",
                input_data=f"{len(frame_data)} bytes",
                action=DefenseAction.BLOCKED,
            )
            return False, violation

        # Check canary (last 8 bytes)
        if len(frame_data) >= 8:
            frame_canary = struct.unpack("Q", frame_data[-8:])[0]
            if frame_canary != canary_value:
                violation = SecurityViolation(
                    timestamp=datetime.now(),
                    vulnerability_type=VulnerabilityType.BUFFER_OVERFLOW,
                    description="Stack canary corrupted - buffer overflow detected",
                    input_data="canary mismatch",
                    action=DefenseAction.BLOCKED,
                )
                return False, violation

        return True, None

    def get_stats(self) -> dict[str, Any]:
        """Get defense statistics."""
        violations = self.validator.get_violations()

        by_type: dict[str, int] = {}
        for vuln_type in VulnerabilityType:
            by_type[vuln_type.value] = len([
                v for v in violations if v.vulnerability_type == vuln_type
            ])

        by_action: dict[str, int] = {}
        for action in DefenseAction:
            by_action[action.value] = len([
                v for v in violations if v.action == action
            ])

        return {
            "total_violations": len(violations),
            "by_type": by_type,
            "by_action": by_action,
            "strict_mode": self.strict_mode,
            "active_allocations": len(self._allocated_memory),
            "freed_allocations": len(self._freed_addresses),
        }

    def save_state(self) -> None:
        """Save state to storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        violations_file = self.storage_path / "stack_violations.json"
        atomic_write_json(violations_file, [v.to_dict() for v in self.validator._violations[-100:]])

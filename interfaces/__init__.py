"""Jarvis Interfaces Module"""
from .cli import CLIInterface
from .voice import VoiceInterface
from .web_react import WebInterface, run_web_interface

__all__ = ["CLIInterface", "VoiceInterface", "WebInterface", "run_web_interface"]

"""Terminal interface for Jarvis - OpenCode-style CLI experience."""
import asyncio
import json
import logging
import re
from pathlib import Path

from core.agent import JarvisAgent
from core.helpers import create_tracked_task

logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"


class TerminalInterface:
    """OpenCode-style terminal interface for Jarvis."""

    def __init__(self, agent: JarvisAgent, working_dir: str | None = None):
        self.agent = agent
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.running = False
        self.last_response = ""
        self.command_history: list[str] = []
        self.history_file = Path.home() / ".jarvis" / "terminal_history.json"
        self._load_history()

    def _load_history(self):
        """Load command history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, encoding='utf-8') as f:
                    self.command_history = json.load(f)[-100:]  # Keep last 100
        except Exception as e:
            logger.warning(f"Failed to load command history: {e}")

    def _save_history(self):
        """Save command history to file."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.command_history[-100:], f)
        except Exception as e:
            logger.warning(f"Failed to save command history: {e}")

    def _print_header(self):
        """Print the welcome header."""
        print()
        print(f"{Colors.BRIGHT_CYAN}╭──────────────────────────────────────────────────────────╮{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  {Colors.BOLD}🧠 JARVIS{Colors.RESET} - AI Assistant Terminal              {Colors.BRIGHT_CYAN}│{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  Model: {Colors.YELLOW}{self.agent.llm.model}{Colors.RESET}  •  Tools: {Colors.GREEN}{len(self.agent.tools)}{Colors.RESET}  •  Type {Colors.DIM}/help{Colors.RESET} for commands  {Colors.BRIGHT_CYAN}│{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}╰──────────────────────────────────────────────────────────╯{Colors.RESET}")
        print()

    def _print_status(self, status: str, detail: str = ""):
        """Print a status indicator."""
        if status == "thinking":
            print(f"\n{Colors.DIM}◇ Thinking...{Colors.RESET}", end="", flush=True)
        elif status == "tool":
            print(f"\r{Colors.YELLOW}🔧 {detail}{Colors.RESET}                    ", end="", flush=True)
        elif status == "tool_done":
            print(f"\r{Colors.GREEN}✓ {detail}{Colors.RESET}                     ", flush=True)
        elif status == "tool_error":
            print(f"\r{Colors.RED}✗ {detail}{Colors.RESET}                     ", flush=True)
        elif status == "done":
            print(f"\r{' ' * 50}\r", end="")  # Clear status line

    def _format_output(self, text: str) -> str:
        """Format output with syntax highlighting and styling."""
        # Bold markdown headers
        text = re.sub(r'^### (.+)$', f'{Colors.BOLD}\\1{Colors.RESET}', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', f'{Colors.BOLD}{Colors.BRIGHT_CYAN}\\1{Colors.RESET}', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.+)$', f'{Colors.BOLD}{Colors.BRIGHT_MAGENTA}\\1{Colors.RESET}', text, flags=re.MULTILINE)

        # Code blocks
        text = re.sub(r'```(\w*)\n', f'{Colors.DIM}───\\1───{Colors.RESET}\n', text)
        text = re.sub(r'```', f'{Colors.DIM}───{Colors.RESET}', text)

        # Inline code
        text = re.sub(r'`([^`]+)`', f'{Colors.DIM}\\1{Colors.RESET}', text)

        # Bold and italic
        text = re.sub(r'\*\*([^*]+)\*\*', f'{Colors.BOLD}\\1{Colors.RESET}', text)
        text = re.sub(r'\*([^*]+)\*', f'{Colors.ITALIC}\\1{Colors.RESET}', text)

        # Bullet points
        text = re.sub(r'^(\s*[-•]) ', f'{Colors.CYAN}\\1{Colors.RESET} ', text, flags=re.MULTILINE)

        # Numbers
        text = re.sub(r'^(\s*\d+\.) ', f'{Colors.CYAN}\\1{Colors.RESET} ', text, flags=re.MULTILINE)

        # URLs
        text = re.sub(r'(https?://[^\s]+)', f'{Colors.BLUE}{Colors.UNDERLINE}\\1{Colors.RESET}', text)

        return text

    def _print_tool_usage(self, tool_name: str, args: dict | None = None):
        """Print tool usage indicator."""
        args_str = ""
        if args:
            # Truncate long args
            args_str = json.dumps(args, indent=None)[:50]
            if len(json.dumps(args)) > 50:
                args_str += "..."
        print(f"\n{Colors.YELLOW}🔧 Using: {Colors.BOLD}{tool_name}{Colors.RESET}{Colors.DIM}{args_str}{Colors.RESET}")

    def _print_tool_result(self, tool_name: str, success: bool, output: str | None = None):
        """Print tool result."""
        if success:
            print(f"{Colors.GREEN}✓ {tool_name} completed{Colors.RESET}")
            if output and len(output) < 500:
                print(f"{Colors.DIM}{output[:500]}{'...' if len(output) > 500 else ''}{Colors.RESET}")
        else:
            print(f"{Colors.RED}✗ {tool_name} failed{Colors.RESET}")

    def _get_input(self, prompt: str = "You") -> str:
        """Get user input with readline-like support."""
        try:
            # Show working directory in prompt
            dir_name = self.working_dir.name
            prompt_str = f"\n{Colors.BRIGHT_CYAN}{dir_name}{Colors.RESET} {Colors.DIM}❯{Colors.RESET} "

            # Use simple input for now
            line = input(prompt_str)
            return line.strip()
        except EOFError:
            return "/exit"
        except KeyboardInterrupt:
            print()
            return ""

    def _process_command(self, cmd: str) -> bool:
        """Process slash commands. Returns True if command was handled."""
        cmd_lower = cmd.lower().strip()

        if cmd_lower in ["/exit", "/quit", "exit", "quit"]:
            print(f"\n{Colors.DIM}Goodbye!{Colors.RESET}\n")
            self.running = False
            return True

        if cmd_lower == "/help":
            self._show_help()
            return True

        if cmd_lower == "/clear":
            print("\033[2J\033[H", end="")  # Clear screen
            self._print_header()
            return True

        if cmd_lower == "/tools":
            self._list_tools()
            return True

        if cmd_lower == "/status":
            self._show_status()
            return True

        if cmd_lower == "/memory":
            self._show_memory()
            return True

        if cmd_lower.startswith("/cd "):
            new_dir = cmd[4:].strip()
            try:
                if new_dir == "..":
                    self.working_dir = self.working_dir.parent
                else:
                    path = Path(new_dir)
                    if not path.is_absolute():
                        path = self.working_dir / path
                    if path.exists() and path.is_dir():
                        self.working_dir = path.resolve()
                    else:
                        print(f"{Colors.RED}Directory not found: {new_dir}{Colors.RESET}")
                print(f"{Colors.DIM}Working directory: {self.working_dir}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.RESET}")
            return True

        if cmd_lower == "/history":
            self._show_history()
            return True

        if cmd_lower == "/pwd":
            print(f"{self.working_dir}")
            return True

        if cmd_lower.startswith("/run "):
            # Run a file
            filename = cmd[5:].strip()
            self._run_file(filename)
            return True

        if cmd_lower.startswith("/save "):
            # Save last response to file
            filename = cmd[6:].strip()
            self._save_response(filename)
            return True

        return False

    def _show_help(self):
        """Show help information."""
        print(f"""
{Colors.BOLD}Commands:{Colors.RESET}
  /help          Show this help message
  /exit, /quit   Exit Jarvis
  /clear         Clear the terminal
  /tools         List available tools
  /status        Show session status
  /memory        Show memory statistics
  /history       Show command history
  /pwd           Print working directory
  /cd <dir>      Change working directory
  /run <file>    Run a file as input
  /save <file>   Save last response to file

{Colors.BOLD}Features:{Colors.RESET}
  • Natural language interaction
  • Tool execution with auto-approval
  • File read/write capabilities
  • Web search and fetch
  • Code execution
  • Persistent memory

{Colors.BOLD}Tips:{Colors.RESET}
  - Be specific about what you want
  - Ask follow-up questions
  - Use /tools to see what Jarvis can do
  - Jarvis remembers context from the conversation
""")

    def _list_tools(self):
        """List available tools."""
        tools = sorted(self.agent.tools.keys())
        print(f"\n{Colors.BOLD}Available Tools ({len(tools)}):{Colors.RESET}\n")

        # Group tools by category
        categories = {
            "File": ["read_file", "write_file", "list_files", "delete_file", "search_files"],
            "Web": ["web_search", "web_fetch"],
            "System": ["shell", "system_info"],
            "Memory": ["remember", "recall", "forget"],
            "Analysis": ["analyze_data", "sql_query", "statistics"],
            "Evolution": ["evolution_status", "propose_gene", "gene_list", "practice_tool"],
        }

        shown = set()
        for category, tools_in_cat in categories.items():
            matching = [t for t in tools if any(t.startswith(prefix) for prefix in tools_in_cat)]
            if matching:
                print(f"{Colors.CYAN}{category}:{Colors.RESET}")
                for t in matching:
                    shown.add(t)
                    print(f"  • {t}")

        # Other tools
        other = [t for t in tools if t not in shown]
        if other:
            print(f"\n{Colors.CYAN}Other:{Colors.RESET}")
            for t in other[:20]:  # Limit output
                print(f"  • {t}")
            if len(other) > 20:
                print(f"  {Colors.DIM}... and {len(other) - 20} more{Colors.RESET}")

    def _show_status(self):
        """Show session status."""
        print(f"\n{Colors.BOLD}Session Status:{Colors.RESET}")
        print(f"  Model: {Colors.YELLOW}{self.agent.llm.model}{Colors.RESET}")
        print(f"  Tools: {Colors.GREEN}{len(self.agent.tools)}{Colors.RESET}")
        print(f"  Turns: {self.agent._turn_count}")
        print(f"  Working: {self.working_dir}")

        # Memory stats
        try:
            messages = self.agent.memory.get_messages_for_llm()
            print(f"  Context: {len(messages)} messages")
        except Exception as e:
            logger.debug(f"Could not get memory stats: {e}")

    def _show_memory(self):
        """Show memory statistics."""
        print(f"\n{Colors.BOLD}Memory:{Colors.RESET}")
        try:
            messages = self.agent.memory.get_messages_for_llm()
            print(f"  Conversation: {len(messages)} messages")

            # Estimate tokens
            total_chars = sum(len(str(m)) for m in messages)
            print(f"  ~{total_chars // 4:,} tokens estimated")
        except Exception as e:
            print(f"  Error: {e}")

        # Persistent memory
        try:
            from core.persistent_memory import get_persistent_memory
            pm = get_persistent_memory()
            print(f"  Long-term: {len(pm.long_term_memories)} memories")
            print(f"  User facts: {len(pm.user_facts)} facts")
        except ImportError:
            print("  Long-term: Not available")
        except Exception as e:
            logger.debug(f"Could not get persistent memory: {e}")

    def _show_history(self):
        """Show command history."""
        print(f"\n{Colors.BOLD}Recent Commands:{Colors.RESET}\n")
        for i, cmd in enumerate(self.command_history[-10:], 1):
            # Truncate long commands
            display_cmd = cmd[:60] + "..." if len(cmd) > 60 else cmd
            print(f"  {Colors.DIM}{i:2}.{Colors.RESET} {display_cmd}")

    def _run_file(self, filename: str):
        """Run a file as input."""
        try:
            path = self.working_dir / filename
            if not path.exists():
                print(f"{Colors.RED}File not found: {filename}{Colors.RESET}")
                return

            with open(path, encoding='utf-8') as f:
                content = f.read()

            print(f"{Colors.DIM}Running {filename}...{Colors.RESET}")
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self._process_message(content))
            else:
                # Tracked so the task isn't garbage-collected before it runs,
                # which would silently drop the user's file.
                create_tracked_task(
                    self._process_message(content),
                    name=f"terminal:run_file:{filename}",
                )
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")

    def _save_response(self, filename: str):
        """Save last response to file."""
        if not self.last_response:
            print(f"{Colors.RED}No response to save{Colors.RESET}")
            return

        try:
            path = self.working_dir / filename
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.last_response)
            print(f"{Colors.GREEN}Saved to {filename}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Error saving: {e}{Colors.RESET}")

    async def _process_message(self, user_input: str):
        """Process a message and stream the response."""
        self._print_status("thinking")

        tool_in_progress = None
        response_parts = []

        try:
            async for token in self.agent.run_step_stream(user_input):
                # Track tool usage
                if "🔧 **Using" in token or "🔧 **Executing" in token:
                    match = re.search(r'\*\*([^*]+)\*\*', token)
                    if match:
                        tool_in_progress = match.group(1)
                        self._print_status("tool", f"Using {tool_in_progress}...")

                # Track tool completion
                if "✅ **" in token:
                    match = re.search(r'\*\*([^*]+)\*\*', token)
                    if match:
                        self._print_status("tool_done", f"{match.group(1)} done")

                if "❌ **" in token:
                    match = re.search(r'\*\*([^*]+)\*\*', token)
                    if match:
                        self._print_status("tool_error", f"{match.group(1)} failed")
                    tool_in_progress = None

                # Clear status and print content
                if token.strip() and not any(x in token for x in ["🔧 **", "✅ **", "❌ **", "📊 **"]):
                    self._print_status("done")
                    formatted = self._format_output(token)
                    print(formatted, end="", flush=True)
                    response_parts.append(token)

            self._print_status("done")

            # Ensure newline at end
            if response_parts and not response_parts[-1].endswith('\n'):
                print()

            self.last_response = ''.join(response_parts)

        except Exception as e:
            self._print_status("done")
            print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
            logger.error(f"Error processing message: {e}")

    async def run(self):
        """Run the terminal interface."""
        self.running = True
        self._print_header()

        # Check if LLM is ready
        try:
            ready = await self.agent.is_ready()
            if not ready:
                print(f"{Colors.RED}Warning: LLM may not be ready. Waiting...{Colors.RESET}")
                await asyncio.sleep(2)
        except Exception as e:
            print(f"{Colors.YELLOW}Note: Could not verify LLM readiness: {e}{Colors.RESET}")

        while self.running:
            try:
                # Get user input
                user_input = self._get_input()

                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    if self._process_command(user_input):
                        if not self.running:
                            break
                        continue

                # Save to history
                self.command_history.append(user_input)
                self._save_history()

                # Process with agent
                await self._process_message(user_input)

            except KeyboardInterrupt:
                print(f"\n{Colors.DIM}Interrupted. Type /exit to quit.{Colors.RESET}")
                continue
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")

        # Cleanup
        try:
            if hasattr(self.agent, 'stop'):
                await self.agent.stop()
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")

    async def start(self):
        """Start the terminal interface."""
        await self.run()

    def run_sync(self):
        """Run the terminal interface (sync wrapper for non-async contexts)."""
        asyncio.run(self.run())


def run_terminal_interface(agent: JarvisAgent, working_dir: str = None):
    """Run the terminal interface."""
    interface = TerminalInterface(agent, working_dir)
    return interface.start()

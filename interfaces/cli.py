"""CLI interface for Jarvis."""
import logging
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from core.agent import JarvisAgent
from core.permissions import PermissionLevel

logger = logging.getLogger(__name__)


class CLIInterface:
    """Terminal interface for Jarvis."""

    def __init__(self, agent: JarvisAgent):
        self.agent = agent
        self.console = Console()
        self.running = False
        self.show_suggestions = True
        self._refine_manager = None
        self._self_refine_agent = None
        self._daydream_events: list[dict] = []
        self._daydream_active = False

    async def _get_refine_manager(self):
        """Lazy load the auto-refine manager."""
        if self._refine_manager is None:
            try:
                from core.auto_refine import AutoRefineManager
                self._refine_manager = AutoRefineManager(
                    llm_client=self.agent.llm,
                    base_path=self.agent.config.get_storage_path().parent if hasattr(self.agent.config, 'get_storage_path') else None
                )
            except Exception as e:
                self.console.print(f"[yellow]Auto-refine not available: {e}[/yellow]")
        return self._refine_manager

    async def _get_self_refine_agent(self):
        """Lazy load the self-refine agent."""
        if self._self_refine_agent is None:
            try:
                from core.self_refine import get_self_refine_agent
                self._self_refine_agent = get_self_refine_agent(self.agent.llm)
            except Exception as e:
                self.console.print(f"[yellow]Self-refine agent not available: {e}[/yellow]")
        return self._self_refine_agent

    async def start(self) -> None:
        """Start the CLI interface."""
        self.running = True

        # Connect to daydream engine for activity display
        self._connect_daydream_engine()

        # Welcome message
        name = self.agent.config.identity.name
        self.console.print(Panel.fit(
            f"[bold cyan]{name.upper()}[/bold cyan] - Thoughtful AI Companion\n"
            "[dim]Type 'exit' or 'quit' to stop[/dim]\n"
            "[dim]Type /help for commands[/dim]",
            border_style="cyan",
        ))

        # Check if LLM is ready
        if not await self.agent.is_ready():
            self.console.print("[red]Error: Cannot connect to Ollama. Make sure Ollama is running with the glm-5 model.[/red]")
            return

        self.console.print("[green]Ready![/green] How can I help you today?\n")

        # Main loop
        while self.running:
            try:
                # Get user input
                user_input = Prompt.ask("[bold blue]You[/bold blue]")

                if not user_input.strip():
                    continue

                # Check for exit commands
                if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                    self.console.print("\n[cyan]Goodbye! Have a great day![/cyan]")
                    self.running = False
                    continue

                # Check for special commands
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue

                # Process with agent
                await self._process_input(user_input)

                # Check for proactive suggestions
                if self.show_suggestions:
                    await self._check_for_suggestions()

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            except EOFError:
                self.console.print("\n[cyan]Goodbye![/cyan]")
                self.running = False

    async def _process_input(self, user_input: str) -> None:
        """Process user input with the agent."""
        with self.console.status("[cyan]Thinking...[/cyan]"):
            response = await self.agent.run_step(user_input)

        # Check for permission request
        if response.startswith("[PERMISSION_REQUIRED]"):
            await self._handle_permission_request(response)
            return

        # Display response
        self._display_response(response)

    def _display_response(self, response: str) -> None:
        """Display agent response."""
        self.console.print()
        name = self.agent.config.identity.name
        # Try to render as markdown, fall back to plain text
        try:
            self.console.print(Panel(Markdown(response), title=f"[bold green]{name}[/bold green]", border_style="green"))
        except Exception:
            self.console.print(Panel(response, title=f"[bold green]{name}[/bold green]", border_style="green"))
        self.console.print()

    async def _check_for_suggestions(self) -> None:
        """Check for and display any proactive suggestions."""
        suggestion = self.agent.get_pending_suggestion()
        if suggestion:
            self._display_suggestion(suggestion)

    def _display_suggestion(self, suggestion: Any) -> None:
        """Display a proactive suggestion."""
        formatted = self.agent.awareness.format_suggestion(
            suggestion.content,
            suggestion.confidence,
            suggestion.suggestion_type.value,
        )
        self.console.print()
        self.console.print(Panel(formatted, title="[bold magenta]💭 Suggestion[/bold magenta]", border_style="magenta"))
        self.console.print()

    def _connect_daydream_engine(self) -> None:
        """Connect to daydream engine for activity monitoring."""
        try:
            from core.daydream_engine import get_daydream_engine
            engine = get_daydream_engine()

            engine.set_callbacks(
                on_start=self._on_daydream_start,
                on_end=self._on_daydream_end,
                daydream_action=self._on_daydream_action,
            )
            self.console.print("[dim]Daydream engine connected[/dim]")
        except Exception as e:
            self.console.print(f"[dim]Daydream engine not available: {e}[/dim]")

    async def _on_daydream_start(self) -> None:
        """Callback when daydream starts."""
        self._daydream_active = True
        self._daydream_events.append({
            "type": "start",
            "timestamp": datetime.now().isoformat(),
        })
        self.console.print("\n[dim cyan]💫 Entering daydream mode...[/dim cyan]")

    async def _on_daydream_end(self) -> None:
        """Callback when daydream ends."""
        self._daydream_active = False
        self._daydream_events.append({
            "type": "end",
            "timestamp": datetime.now().isoformat(),
        })

        # Show summary of daydream activity
        if self._daydream_events:
            recent = [e for e in self._daydream_events[-5:] if e.get("type") == "insight"]
            if recent:
                self.console.print("\n[dim green]✨ Daydream complete. Insights gained.[/dim green]")

    async def _on_daydream_action(self) -> None:
        """Callback during daydream actions."""
        try:
            from core.daydream_engine import get_daydream_engine

            # Get status
            engine = get_daydream_engine()
            status = engine.get_status()

            # Show activity
            if status.get("running"):
                insights_count = status.get("daydream_count", 0)
                self.console.print(f"[dim purple]  🧠 Exploring... (dream #{insights_count})[/dim purple]")
        except Exception as e:
            logger.debug(f"Failed to update daydream status: {e}")

    async def _handle_permission_request(self, response: str) -> None:
        """Handle a permission request from the agent."""
        # Parse permission request
        # Format: [PERMISSION_REQUIRED] Tool: <name>, Args: <args>
        parts = response.replace("[PERMISSION_REQUIRED]", "").strip()
        tool_name = None
        tool_args = {}

        if "Tool:" in parts:
            tool_part = parts.split("Tool:")[1].split(",")[0].strip()
            tool_name = tool_part

        if "Args:" in parts:
            import json
            args_str = parts.split("Args:")[1].strip()
            try:
                tool_args = json.loads(args_str)
            except json.JSONDecodeError:
                tool_args = {}

        self.console.print("\n[yellow]⚠ Permission Required[/yellow]")
        self.console.print(f"Tool: [bold]{tool_name}[/bold]")
        if tool_args:
            self.console.print(f"Arguments: {tool_args}")

        # Show ethical reminder for sensitive operations
        sensitive_tools = ["shell", "write_file", "delete_file", "docker", "n8n"]
        if tool_name in sensitive_tools:
            self.console.print("[dim]This action affects your system. Please confirm you understand the implications.[/dim]")

        # Get permission
        choice = Prompt.ask(
            "Allow this action?",
            choices=["o", "s", "a", "d", "cancel"],
            default="o",
            show_choices=True,
            show_default=True,
        )

        if choice == "cancel":
            self.console.print("[yellow]Action cancelled.[/yellow]")
            return

        level_map = {
            "o": PermissionLevel.ALLOW_ONCE,
            "s": PermissionLevel.ALLOW_SESSION,
            "a": PermissionLevel.ALLOW_ALWAYS,
            "d": PermissionLevel.DENY,
        }

        level = level_map.get(choice)
        if level:
            await self.agent.grant_permission(tool_name or "", tool_args, level)

            if level == PermissionLevel.DENY:
                self.console.print("[yellow]Permission denied.[/yellow]")
                return

            # Re-execute with permission granted
            with self.console.status("[cyan]Executing...[/cyan]"):
                response = await self.agent.run_step(f"Retry with permission: {tool_name}")
            self._display_response(response)

    async def _handle_command(self, command: str) -> None:
        """Handle slash commands."""
        cmd = command.lower().strip()
        cmd_parts = cmd.split()
        cmd_name = cmd_parts[0] if cmd_parts else ""

        if cmd_name in ["/help", "/?"]:
            self._show_help()
        elif cmd_name in ["/clear", "/cls"]:
            self.agent.memory.clear()
            self.console.clear()
            self.console.print("[green]Conversation cleared.[/green]")
        elif cmd_name == "/history":
            history = self.agent.memory.get_history()
            for msg in history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:100]
                self.console.print(f"[dim]{role}:[/dim] {content}...")
        elif cmd_name == "/tools":
            self._show_tools()
        elif cmd_name == "/status":
            await self._show_status()
        elif cmd_name == "/focus":
            focus = self.agent.get_current_focus()
            self.console.print(Panel(focus, title="Current Focus", border_style="blue"))
        elif cmd_name == "/awareness":
            if len(cmd_parts) > 1:
                level = cmd_parts[1]
                self.agent.set_awareness_level(level)
                self.console.print(f"[green]Awareness level set to: {level}[/green]")
            else:
                self.console.print("[yellow]Usage: /awareness <level>[/yellow]")
                self.console.print("[dim]Levels: minimal, normal, detailed, debug[/dim]")
        elif cmd_name == "/suggestions":
            if len(cmd_parts) > 1 and cmd_parts[1] in ["on", "off"]:
                self.show_suggestions = cmd_parts[1] == "on"
                state = "enabled" if self.show_suggestions else "disabled"
                self.console.print(f"[green]Proactive suggestions {state}.[/green]")
            else:
                self.console.print("[yellow]Usage: /suggestions on|off[/yellow]")
        elif cmd_name == "/refine":
            await self._handle_refine_command(cmd_parts[1:] if len(cmd_parts) > 1 else [])
        elif cmd_name == "/scan":
            await self._handle_scan_command(cmd_parts[1:] if len(cmd_parts) > 1 else [])
        elif cmd_name == "/tasks":
            await self._handle_tasks_command(cmd_parts[1:] if len(cmd_parts) > 1 else [])
        elif cmd_name == "/evolve":
            await self._handle_evolve_command(cmd_parts[1:] if len(cmd_parts) > 1 else [])
        elif cmd_name == "/memory":
            await self._handle_memory_command(cmd_parts[1:] if len(cmd_parts) > 1 else [])
        elif cmd_name == "/think":
            if len(cmd_parts) > 1:
                topic = " ".join(cmd_parts[1:])
                await self._handle_think_command(topic)
            else:
                self.console.print("[yellow]Usage: /think <topic>[/yellow]")
        elif cmd_name == "/daydream":
            await self._handle_daydream_command(cmd_parts[1:] if len(cmd_parts) > 1 else [])
        elif cmd_name == "/metrics":
            self._show_metrics()
        else:
            self.console.print(f"[yellow]Unknown command: {command}[/yellow]")
            self._show_help()

    def _show_tools(self) -> None:
        """Display available tools in a table."""
        tools = self.agent.tools
        table = Table(title=f"Available Tools ({len(tools)})")
        table.add_column("Tool", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Permission", style="yellow")

        for name, tool in sorted(tools.items()):
            desc = getattr(tool, 'description', 'No description')[:50]
            perm = "⚠️ Required" if getattr(tool, 'requires_permission', False) else ""
            table.add_row(name, desc, perm)

        self.console.print(table)

    async def _show_status(self) -> None:
        """Show detailed agent status."""
        ready = await self.agent.is_ready()
        stats = self.agent.memory.get_context_stats()
        awareness = self.agent.get_awareness_status()

        table = Table(title="Jarvis Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="white")

        table.add_row("LLM Ready", "[green]Yes[/green]" if ready else "[red]No[/red]")
        table.add_row("Messages", str(stats.get('total_messages', 0)))
        table.add_row("Long-term Memories", str(stats.get('long_term_memories', 0)))
        table.add_row("Awareness Level", awareness.get('level', 'normal'))
        table.add_row("Tools Available", str(len(self.agent.tools)))

        if hasattr(self.agent, 'unified_memory'):
            try:
                mem_stats = self.agent.unified_memory.get_stats()
                table.add_row("Vector Memories", str(mem_stats.get('total_memories', 0)))
            except Exception as e:
                logger.debug(f"Failed to get unified memory stats: {e}")
            cache_stats = self.agent.tool_cache.get_stats()
            hit_rate = cache_stats.get('hit_rate', 0) * 100
            table.add_row("Cache Hit Rate", f"{hit_rate:.1f}%")

        self.console.print(table)

    async def _handle_refine_command(self, args: list[str]) -> None:
        """Handle /refine commands for self-improvement."""
        if not args:
            self._show_refine_help()
            return

        subcmd = args[0]

        if subcmd == "scan":
            await self._do_refine_scan(ai="--ai" in args or "-a" in args)
        elif subcmd == "list":
            await self._do_refine_list()
        elif subcmd == "apply":
            change_id = args[1] if len(args) > 1 else None
            await self._do_refine_apply(change_id)
        elif subcmd == "rollback":
            if len(args) > 1:
                await self._do_refine_rollback(args[1])
            else:
                self.console.print("[yellow]Usage: /refine rollback <change_id>[/yellow]")
        elif subcmd == "report":
            await self._do_refine_report()
        else:
            self._show_refine_help()

    async def _do_refine_scan(self, ai: bool = False) -> None:
        """Run a codebase scan."""
        self.console.print("[cyan]Scanning codebase for issues...[/cyan]")
        manager = await self._get_refine_manager()
        if not manager:
            return

        with self.console.status("[cyan]Analyzing...[/cyan]"):
            result = await manager.scan_codebase(include_ai=ai)

        if result is None:
            self.console.print("\n[yellow]Scan produced no results. Try again.[/yellow]")
            return

        self.console.print("\n[green]Scan complete![/green]")
        self.console.print(f"  Files scanned: {getattr(result, 'files_scanned', 0)}")
        self.console.print(f"  Issues found: {len(getattr(result, 'issues_found', []))}")
        self.console.print(f"  Duration: {getattr(result, 'duration_seconds', 0):.1f}s")

        if getattr(result, 'issues_found', []):
            critical = sum(1 for i in result.issues_found if i.severity.value == "critical")
            high = sum(1 for i in result.issues_found if i.severity.value == "high")
            if critical or high:
                self.console.print(f"\n[yellow]⚠️ {critical} critical, {high} high priority issues found[/yellow]")

    async def _do_refine_list(self) -> None:
        """List pending changes."""
        manager = await self._get_refine_manager()
        if not manager:
            return

        pending = manager.get_pending_changes()
        if not pending:
            self.console.print("[dim]No pending changes.[/dim]")
            return

        table = Table(title=f"Pending Changes ({len(pending)})")
        table.add_column("ID", style="cyan")
        table.add_column("File", style="white")
        table.add_column("Risk", style="yellow")
        table.add_column("Status", style="green")

        for change in pending[:20]:
            table.add_row(
                change.id[:20],
                change.file_path[:40],
                change.risk_level,
                change.status.value
            )

        self.console.print(table)

    async def _do_refine_apply(self, change_id: str | None) -> None:
        """Apply a change or all approved changes."""
        manager = await self._get_refine_manager()
        if not manager:
            return

        if change_id:
            success, msg = manager.apply_change(change_id)
            if success:
                self.console.print(f"[green]✓ {msg}[/green]")
            else:
                self.console.print(f"[red]✗ {msg}[/red]")
        else:
            results = await manager.apply_approved_changes(auto_permission=False)
            for cid, success, msg in results:
                status = "[green]✓[/green]" if success else "[red]✗[/red]"
                self.console.print(f"{status} {cid}: {msg}")

    async def _do_refine_rollback(self, change_id: str) -> None:
        """Rollback a change."""
        manager = await self._get_refine_manager()
        if not manager:
            return

        success, msg = manager.rollback_change(change_id)
        if success:
            self.console.print(f"[green]✓ {msg}[/green]")
        else:
            self.console.print(f"[red]✗ {msg}[/red]")

    async def _do_refine_report(self) -> None:
        """Show the refine report."""
        manager = await self._get_refine_manager()
        if not manager:
            return

        report = manager.get_summary_report()
        self.console.print(Panel(Markdown(report), title="Refine Report", border_style="blue"))

    def _show_refine_help(self) -> None:
        """Show refine command help."""
        help_text = """
[bold]Refine Commands:[/bold]
  /refine scan [--ai]    Scan codebase for issues (use --ai for deep analysis)
  /refine list           List pending changes
  /refine apply [id]     Apply a specific change or all approved changes
  /refine rollback <id>  Rollback an applied change
  /refine report         Show summary report

[bold]Examples:[/bold]
  /refine scan           Quick pattern-based scan
  /refine scan --ai      Deep AI-powered analysis (slower)
  /refine apply          Apply all approved changes
        """
        self.console.print(Panel(help_text, title="Refine Help", border_style="yellow"))

    async def _handle_scan_command(self, args: list[str]) -> None:
        """Handle /scan commands for self-refine agent."""
        if not args:
            self._show_scan_help()
            return

        subcmd = args[0]

        if subcmd == "status":
            agent = await self._get_self_refine_agent()
            if agent:
                stats = agent.get_stats()
                table = Table(title="Self-Refine Status")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="white")
                for k, v in stats.items():
                    table.add_row(k, str(v))
                self.console.print(table)
        elif subcmd == "tasks":
            agent = await self._get_self_refine_agent()
            if agent:
                tasks = agent.get_pending_tasks(limit=10)
                if not tasks:
                    self.console.print("[dim]No pending tasks.[/dim]")
                else:
                    table = Table(title=f"Pending Tasks ({len(tasks)})")
                    table.add_column("ID", style="cyan")
                    table.add_column("Title", style="white")
                    table.add_column("Priority", style="yellow")
                    for task in tasks:
                        table.add_row(task.id[:15], task.title[:50], task.priority.name)
                    self.console.print(table)
        else:
            self._show_scan_help()

    def _show_scan_help(self) -> None:
        """Show scan command help."""
        self.console.print(Panel("""
[bold]Scan Commands:[/bold]
  /scan status    Show self-refine agent status
  /scan tasks     List pending refinement tasks
        """, title="Scan Help", border_style="yellow"))

    async def _handle_tasks_command(self, args: list[str]) -> None:
        """Handle /tasks commands."""
        agent = await self._get_self_refine_agent()
        if not agent:
            self.console.print("[yellow]Self-refine agent not available.[/yellow]")
            return

        if not args or args[0] == "list":
            tasks = agent.get_pending_tasks(limit=20)
            if not tasks:
                self.console.print("[dim]No pending tasks.[/dim]")
                return

            table = Table(title="Refinement Tasks")
            table.add_column("ID", style="cyan")
            table.add_column("Category", style="blue")
            table.add_column("Title", style="white")
            table.add_column("Priority", style="yellow")

            for task in tasks:
                table.add_row(
                    task.id[:12],
                    task.category,
                    task.title[:40],
                    task.priority.name
                )

            self.console.print(table)

    async def _handle_evolve_command(self, args: list[str]) -> None:
        """Handle /evolve commands for evolution orchestrator."""
        if not args:
            self._show_evolve_help()
            return

        try:
            from core.evolution_orchestrator import get_evolution_orchestrator
            orchestrator = get_evolution_orchestrator()
        except Exception as e:
            self.console.print(f"[yellow]Evolution orchestrator not available: {e}[/yellow]")
            return

        subcmd = args[0]

        if subcmd == "status":
            status = orchestrator.get_status()
            table = Table(title="Evolution Status")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            for k, v in status.items():
                table.add_row(k, str(v))
            self.console.print(table)

        elif subcmd == "history":
            status = orchestrator.get_status()
            history = status.get("recent_proposals", [])
            if not history:
                self.console.print("[dim]No evolution history yet.[/dim]")
            else:
                table = Table(title="Evolution History")
                table.add_column("Time", style="cyan")
                table.add_column("Title", style="blue")
                table.add_column("Status", style="white")
                for entry in history:
                    table.add_row(
                        entry.get('timestamp', 'N/A')[:19],
                        entry.get('title', 'N/A')[:30],
                        entry.get('status', 'N/A')
                    )
                self.console.print(table)
        else:
            self._show_evolve_help()

    def _show_evolve_help(self) -> None:
        """Show evolve command help."""
        self.console.print(Panel("""
[bold]Evolution Commands:[/bold]
  /evolve status    Show evolution system status
  /evolve history   Show recent evolution history
        """, title="Evolve Help", border_style="yellow"))

    async def _handle_memory_command(self, args: list[str]) -> None:
        """Handle /memory commands."""
        if not args:
            self._show_memory_help()
            return

        subcmd = args[0]

        if subcmd == "stats":
            try:
                from core.unified_memory import get_unified_memory
                memory = get_unified_memory()
                stats = memory.get_stats()
                table = Table(title="Memory Stats")
                table.add_column("Type", style="cyan")
                table.add_column("Count", style="white")
                for k, v in stats.items():
                    table.add_row(k, str(v))
                self.console.print(table)
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

        elif subcmd == "clear":
            if len(args) > 1 and args[1] == "confirm":
                try:
                    self.agent.memory.clear()
                    self.console.print("[green]Conversation memory cleared.[/green]")
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
            else:
                self.console.print("[yellow]Use /memory clear confirm to clear memory.[/yellow]")

        elif subcmd == "search":
            if len(args) > 1:
                query = " ".join(args[1:])
                try:
                    from core.unified_memory import get_unified_memory
                    memory = get_unified_memory()
                    results = memory.search(query, limit=5)
                    if results:
                        table = Table(title=f"Search Results for '{query}'")
                        table.add_column("Content", style="white")
                        table.add_column("Score", style="cyan")
                        for entry in results:
                            table.add_row(entry.content[:80], f"{entry.importance:.2f}")
                        self.console.print(table)
                    else:
                        self.console.print("[dim]No results found.[/dim]")
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
            else:
                self.console.print("[yellow]Usage: /memory search <query>[/yellow]")
        else:
            self._show_memory_help()

    def _show_memory_help(self) -> None:
        """Show memory command help."""
        self.console.print(Panel("""
[bold]Memory Commands:[/bold]
  /memory stats           Show memory statistics
  /memory clear confirm   Clear conversation memory
  /memory search <query>  Search stored memories
        """, title="Memory Help", border_style="yellow"))

    async def _handle_think_command(self, topic: str) -> None:
        """Handle /think command for focused thinking."""
        self.console.print(f"[cyan]Thinking about: {topic}[/cyan]")

        if self.agent.thinking_engine:
            try:
                from core.thinking import ConversationContext
                context = ConversationContext(
                    recent_messages=[{"role": "user", "content": topic}],
                    topics=[topic],
                )
                thought = await self.agent.thinking_engine.think(context)
                if thought:
                    self.console.print(Panel(
                        thought.content or "No thought generated",
                        title=f"💭 Thinking: {topic}",
                        border_style="blue"
                    ))
                else:
                    self.console.print("[dim]No thought generated.[/dim]")
            except Exception as e:
                self.console.print(f"[red]Thinking error: {e}[/red]")
        else:
            self.console.print("[yellow]Thinking engine not available.[/yellow]")

    def _show_metrics(self) -> None:
        """Show performance metrics."""
        table = Table(title="Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        if hasattr(self.agent, 'tool_cache'):
            stats = self.agent.tool_cache.get_stats()
            table.add_row("Cache Size", str(stats.get('size', 0)))
            table.add_row("Cache Hits", str(stats.get('hits', 0)))
            table.add_row("Cache Misses", str(stats.get('misses', 0)))
            hit_rate = stats.get('hit_rate', 0) * 100
            table.add_row("Cache Hit Rate", f"{hit_rate:.1f}%")

        table.add_row("Tools Available", str(len(self.agent.tools)))
        table.add_row("Current Session", self.agent._session_id[:12] if hasattr(self.agent, '_session_id') else 'N/A')
        table.add_row("Turn Count", str(self.agent._turn_count) if hasattr(self.agent, '_turn_count') else '0')

        self.console.print(table)

    async def _handle_daydream_command(self, args: list[str]) -> None:
        """Handle /daydream commands."""
        if not args or args[0] == "status":
            await self._show_daydream_status()
        elif args[0] == "events":
            self._show_daydream_events()
        elif args[0] == "trigger":
            await self._trigger_daydream()
        elif args[0] == "practice":
            await self._practice_tools()
        else:
            self._show_daydream_help()

    async def _show_daydream_status(self) -> None:
        """Show daydream engine status."""
        try:
            from core.daydream_engine import get_daydream_engine
            from core.tool_learning import get_tool_learning_engine

            engine = get_daydream_engine()
            learning = get_tool_learning_engine()

            status = engine.get_status()
            competencies = learning.get_all_competencies()
            priorities = learning.get_practice_priorities()

            table = Table(title="Daydream Engine Status")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Running", "[green]Yes[/green]" if status.get("running") else "[red]No[/red]")
            table.add_row("Can Daydream", "[green]Yes[/green]" if status.get("can_daydream") else "[yellow]No[/yellow]")
            table.add_row("Daydream Count", str(status.get("daydream_count", 0)))
            table.add_row("Total Daydream Time", f"{status.get('total_daydream_time', 0):.0f}s")
            table.add_row("Throttled", "[yellow]Yes[/yellow]" if status.get("throttled") else "[green]No[/green]")

            if status.get("last_daydream"):
                table.add_row("Last Daydream", status["last_daydream"][:19])

            self.console.print(table)

            # Show tool competency summary
            if competencies:
                comp_table = Table(title="Tool Competencies")
                comp_table.add_column("Tool", style="cyan")
                comp_table.add_column("Level", style="white")
                comp_table.add_column("Uses", style="dim")
                comp_table.add_column("Success", style="green")

                for name, comp in list(competencies.items())[:10]:
                    rate = comp.successful_uses / max(1, comp.total_uses) * 100
                    comp_table.add_row(name[:20], comp.level.value, str(comp.total_uses), f"{rate:.0f}%")

                self.console.print(comp_table)

            if priorities:
                self.console.print(f"\n[dim]Tools needing practice: {', '.join(priorities[:5])}[/dim]")

        except Exception as e:
            self.console.print(f"[red]Daydream engine not available: {e}[/red]")

    def _show_daydream_events(self) -> None:
        """Show recent daydream events."""
        if not self._daydream_events:
            self.console.print("[dim]No daydream events recorded this session.[/dim]")
            return

        table = Table(title=f"Recent Daydream Events ({len(self._daydream_events)})")
        table.add_column("Time", style="cyan")
        table.add_column("Type", style="white")
        table.add_column("Details", style="dim")

        for event in self._daydream_events[-20:]:
            timestamp = event.get("timestamp", "")[11:19]
            event_type = event.get("type", "unknown")
            details = event.get("content", "")[:40] if event.get("content") else ""
            table.add_row(timestamp, event_type, details)

        self.console.print(table)

    async def _trigger_daydream(self) -> None:
        """Manually trigger a daydream session."""
        try:
            from core.daydream_engine import get_daydream_engine

            engine = get_daydream_engine()
            if not engine.can_daydream():
                self.console.print("[yellow]Cannot daydream right now (busy or not idle)[/yellow]")
                return

            self.console.print("[cyan]Triggering daydream session...[/cyan]")
            await engine._enter_daydream()
            self.console.print("[green]Daydream session complete.[/green]")

        except Exception as e:
            self.console.print(f"[red]Failed to trigger daydream: {e}[/red]")

    async def _practice_tools(self) -> None:
        """Manually trigger tool practice."""
        try:
            from core.daydream_engine import get_daydream_engine

            engine = get_daydream_engine()
            result = await engine.practice_tools()

            practiced = result.get("practiced", 0)
            if practiced > 0:
                self.console.print(f"[green]Practiced {practiced} tools[/green]")
                for tool, details in result.get("results", {}).items():
                    level = details.get("new_level", "unknown")
                    self.console.print(f"  [dim]{tool}: {level}[/dim]")
            else:
                self.console.print("[dim]No tools to practice right now[/dim]")

        except Exception as e:
            self.console.print(f"[red]Failed to practice tools: {e}[/red]")

    def _show_daydream_help(self) -> None:
        """Show daydream command help."""
        self.console.print(Panel("""
[bold]Daydream Commands:[/bold]
  /daydream status    Show daydream engine status
  /daydream events    Show recent daydream events
  /daydream trigger   Manually trigger a daydream session
  /daydream practice  Practice tools to improve competency
        """, title="Daydream Help", border_style="yellow"))

    def _show_help(self) -> None:
        """Show help message."""
        help_text = """
[bold]Basic Commands:[/bold]
  /help, /?           Show this help message
  /clear, /cls        Clear conversation history
  /history            Show conversation history
  /exit, quit         Exit Jarvis

[bold]Tools & Status:[/bold]
  /tools              List available tools (with descriptions)
  /status             Show detailed agent status
  /metrics            Show performance metrics
  /focus              Show current focus topics

[bold]Memory & Awareness:[/bold]
  /memory stats       Show memory statistics
  /memory search <q>  Search stored memories
  /memory clear       Clear conversation memory
  /awareness <level>  Set awareness level (minimal, normal, detailed, debug)
  /suggestions on/off Toggle proactive suggestions

[bold]Self-Improvement:[/bold]
  /refine scan [--ai] Scan codebase for issues and improvements
  /refine list        List pending code changes
  /refine apply [id]  Apply approved changes
  /refine report      Show improvement report
  /scan status        Show self-refine agent status
  /scan tasks         List pending refinement tasks
  /tasks              View background refinement tasks
  /evolve status      Show evolution system status
  /evolve history     View evolution history

[bold]Thinking:[/bold]
  /think <topic>      Trigger focused thinking on a topic

[bold]Daydream & Learning:[/bold]
  /daydream status    Show daydream engine and tool learning status
  /daydream events    Show recent daydream events
  /daydream trigger   Manually trigger a daydream session
  /daydream practice  Practice tools to improve competency

[bold]Tips:[/bold]
  • Ask questions naturally - Jarvis will understand
  • Request file operations, web searches, or system commands
  • Permission will be asked for sensitive actions
  • Use /refine scan to find code improvements
  • Use /status to see what's running
        """
        self.console.print(Panel(help_text, title="Jarvis CLI Help", border_style="blue"))

    async def stop(self) -> None:
        """Stop the CLI interface."""
        self.running = False

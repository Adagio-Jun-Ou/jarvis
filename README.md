# Jarvis - Personal AI Assistant

A thoughtful AI companion that anticipates needs, offers proactive suggestions, and integrates with your digital life.

## Features

- **Multiple Interfaces**: CLI, Web, and Voice interfaces
- **Proactive Thinking**: Jarvis thinks alongside you, offering suggestions at natural pauses
- **Tool Integration**: Web search, file operations, shell commands, Docker, n8n workflows
- **Self-Reflection**: Ability to reflect on interactions and design agentic processes
- **Memory**: Persistent conversation memory with importance tracking
- **Multimodal**: Vision and audio processing capabilities
- **Permission System**: Safe handling of sensitive operations

## Quick Start

### Prerequisites

- Python 3.11+
- Ollama (local) or Ollama Cloud API key

### Installation

```bash
# Clone or navigate to the jarvis directory
cd jarvis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install voice dependencies
pip install faster-whisper pyttsx3 sounddevice openwakeword
```

### Configuration

Create a `config.yaml` file (or use the default):

```yaml
llm:
  model: "glm-5"
  use_cloud: true
  api_key: "your-api-key-here"  # Or set OLLAMA_API_KEY env var

agent:
  name: "Jarvis"
  max_history_turns: 20

interface:
  default: "cli"  # cli, voice, or web
  web_port: 8080

thinking:
  enabled: true
  mode: "supportive"  # supportive, analytical, creative
```

### Running Jarvis

```bash
# CLI interface (default)
python run.py

# Web interface
python run.py --interface web --port 8080

# Voice interface
python run.py --interface voice

# With debug logging
python run.py --debug

# Specify model
python run.py --model llama3.2
```

## Deployment

### Local Development

```bash
# Run with auto-reload (requires uvicorn for web)
python run.py --interface web --debug
```

### Production Deployment

#### Using systemd (Linux)

Create `/etc/systemd/system/jarvis.service`:

```ini
[Unit]
Description=Jarvis AI Assistant
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/jarvis
Environment="OLLAMA_API_KEY=your-api-key"
ExecStart=/path/to/jarvis/venv/bin/python run.py --interface web
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable jarvis
sudo systemctl start jarvis
```

#### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

ENV OLLAMA_API_KEY=""

CMD ["python", "run.py", "--interface", "web", "--port", "8080"]
```

Build and run:

```bash
docker build -t jarvis .
docker run -d -p 8080:8080 -e OLLAMA_API_KEY=your-key jarvis
```

#### Using Docker Compose

```yaml
version: '3.8'

services:
  jarvis:
    build: .
    ports:
      - "8080:8080"
    environment:
      - OLLAMA_API_KEY=${OLLAMA_API_KEY}
    volumes:
      - ./config.yaml:/app/config.yaml
      - jarvis-data:/root/.jarvis

volumes:
  jarvis-data:
```

### Cloud Deployment

#### Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Create fly.toml
fly apps create jarvis

# Deploy
fly deploy
```

#### Railway

```bash
# Install railway CLI
npm install -g @railway/cli

# Deploy
railway init
railway up
```

## Interfaces

### CLI Interface

Full-featured terminal interface with:
- Rich markdown rendering
- Proactive suggestions display
- Command history
- Slash commands (`/help`, `/tools`, `/status`, `/clear`)

### Web Interface

Modern web UI accessible at `http://localhost:8080`:
- Real-time WebSocket communication
- Markdown rendering
- Streaming responses
- Permission modal for sensitive actions
- Responsive design

### Voice Interface

Hands-free voice interaction:
- Wake word detection ("Hey Jarvis")
- Speech-to-text via Whisper
- Text-to-speech output
- Conversational speech model support

## Tools

Jarvis has access to various tools:

| Tool | Description | Permission |
|------|-------------|------------|
| `shell` | Execute shell commands | Required |
| `read_file` | Read file contents | Optional |
| `write_file` | Write to files | Required |
| `web_search` | Search the web | Optional |
| `docker` | Manage Docker containers | Required |
| `n8n` | Trigger n8n workflows | Required |
| `self_reflect` | Reflect on interactions | Optional |
| `design_process` | Design agentic processes | Optional |

## Configuration Reference

```yaml
# LLM Settings
llm:
  model: "glm-5"                    # Model to use
  host: "http://localhost:11434"    # Ollama host (for local)
  temperature: 0.7                  # Temperature for responses
  max_tokens: null                  # Max tokens (null = model max)
  use_cloud: true                   # Use Ollama Cloud
  api_key: null                     # API key (or use env var)

# Agent Settings
agent:
  name: "Jarvis"
  max_history_turns: 20             # Conversation history limit

# Thinking Engine
thinking:
  enabled: true
  mode: "supportive"                # supportive, analytical, creative
  suggestion_interval: 30           # Seconds between thinking
  confidence_threshold: 0.7         # Minimum confidence to suggest

# Memory
memory:
  enabled: true
  max_context_tokens: null          # Context window (null = dynamic)
  long_term_storage: "~/.jarvis/memory"

# Automation
automation:
  docker:
    enabled: true
    socket: "/var/run/docker.sock"
  n8n:
    enabled: true
    url: "http://localhost:5678"
    api_key: null

# Interface
interface:
  default: "cli"
  web_port: 8080

# Voice
voice:
  enabled: true
  wake_word: "hey jarvis"
  tts_engine: "espeak"
  stt_model: "base"
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OLLAMA_API_KEY` | Ollama Cloud API key |
| `GOOGLE_API_KEY` | Google Search API key (optional) |
| `GOOGLE_ENGINE_ID` | Google Search Engine ID (optional) |
| `N8N_API_KEY` | n8n API key (optional) |

## API Endpoints (Web Interface)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check |
| `/status` | GET | Agent status |
| `/tools` | GET | List available tools |
| `/ws` | WebSocket | Chat communication |

## WebSocket Protocol

Send messages as JSON:

```json
{"type": "chat", "content": "Hello Jarvis!"}
{"type": "command", "command": "status"}
```

Receive responses:

```json
{"type": "response", "content": "Hello! How can I help?"}
{"type": "stream", "content": "...", "done": false}
{"type": "permission_request", "content": "..."}
{"type": "error", "content": "..."}
```

## Project Structure

```
jarvis/
├── main.py              # Entry point
├── run.py               # Launcher with logging
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
├── core/
│   ├── agent.py         # Main agent logic
│   ├── config.py        # Configuration management
│   ├── llm.py           # LLM client
│   ├── memory.py        # Conversation memory
│   ├── thinking.py      # Proactive thinking engine
│   ├── prompts.py       # System prompts
│   ├── permissions.py   # Permission management
│   ├── awareness.py     # Context visibility
│   ├── meta.py          # Self-reflection
│   └── multimodal.py    # Vision and audio
├── interfaces/
│   ├── cli.py           # CLI interface
│   ├── web.py           # Web interface
│   └── voice.py         # Voice interface
├── tools/
│   ├── base.py          # Tool base class
│   ├── system.py        # System tools
│   ├── web.py           # Web tools
│   ├── productivity.py  # Productivity tools
│   ├── home.py          # Home automation
│   ├── docker.py        # Docker integration
│   ├── n8n.py           # n8n integration
│   └── meta_tools.py    # Self-reflection tools
└── utils/
    └── audio.py         # Audio utilities
```

## License

MIT License
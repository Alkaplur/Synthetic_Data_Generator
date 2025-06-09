# 🤖 Synthetic Data Generator - Basic MVP

AI-powered synthetic data generation using OpenAI Agents, SDV, and Nemotron.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key

### Installation

1. **Clone and setup:**
```bash
cd synthetic_data_generator_basic
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Set up environment:**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### Running the Application

#### Option 1: Web API (Recommended)
```bash
python main.py
```
- API will be available at `http://localhost:8000`
- Interactive docs at `http://localhost:8000/docs`

#### Option 2: CLI Mode (Testing)
```bash
python main.py cli
```

## 🎯 How It Works

### Three AI Agents System

1. **🎛️ Orchestrator Agent** - Routes users to the right specialist
2. **📊 Sample Data Agent** - Generates data from existing datasets using SDV
3. **✨ Pure Synthetic Agent** - Creates data from scratch using LLM

### User Flows

#### Flow 1: I Have Sample Data
```
User: "I have customer data and want to generate more similar data"
→ Orchestrator routes to Sample Data Agent
→ Agent asks for file path
→ Agent analyzes data and reports findings
→ Agent asks how many rows to generate
→ Agent uses SDV to generate synthetic data
→ User gets download link
```

#### Flow 2: I Need Data From Scratch
```
User: "I need synthetic insurance customer data for testing"
→ Orchestrator routes to Pure Synthetic Agent  
→ Agent asks about business context (industry, geography, etc.)
→ Agent shows examples of proposed data structure
→ Agent asks how many rows to generate
→ Agent uses LLM to generate synthetic data
→ User gets download link
```

## 📡 API Endpoints

### Core Endpoints

- **POST `/chat`** - Main conversation endpoint
- **POST `/chat-with-file`** - Chat with file upload support
- **GET `/download/{file_id}?session_id={session_id}`** - Download generated files

### Management Endpoints

- **GET `/session/{session_id}`** - Session information
- **DELETE `/session/{session_id}`** - Clean up session
- **GET `/sessions`** - List active sessions
- **GET `/health`** - Health check

### Example Usage

#### Start Conversation
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need synthetic customer data",
    "user_id": "user123"
  }'
```

#### Continue Conversation
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I have existing data to base it on",
    "user_id": "user123",
    "session_id": "session-id-from-previous-response"
  }'
```

#### Upload File
```bash
curl -X POST "http://localhost:8000/chat-with-file" \
  -F "message=Here is my customer data file" \
  -F "user_id=user123" \
  -F "session_id=your-session-id" \
  -F "file=@/path/to/your/data.csv"
```

## 🏗️ Architecture

```
synthetic_data_generator_basic/
├── main.py                    # FastAPI app + CLI mode
├── agents.py                  # All OpenAI agents + context
├── tools.py                   # Function tools for agents
├── conversation_handler.py    # Conversation management
└── requirements.txt           # Dependencies
```

### Key Components

- **OpenAI Agents**: Handle conversation flow and decision making
- **SDV (Synthetic Data Vault)**: Generate data from samples using statistical modeling
- **LangChain + Nemotron**: Generate data from scratch using LLM
- **FastAPI**: Web interface and file management
- **Context Management**: Session state and file handling

## 🔧 Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=10
DEFAULT_TEMP_DIR=/tmp/synthetic_data
```

### Model Settings
- **SDV Model**: GaussianCopula (simple, fast, good for MVP)
- **LLM Model**: nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1 (4B parameters, runs locally)

## 🧪 Testing

### CLI Testing
```bash
python main.py cli
```

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Start conversation
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "user_id": "test"}'
```

## 📝 Example Conversations

### Sample Data Flow
```
User: I want to generate synthetic data from my CSV
Agent: I'll help you generate synthetic data from your existing dataset. Please provide the file path to your CSV data file.
User: /path/to/customers.csv
Agent: [Analyzes data] Found 1,000 rows with 8 columns: customer_id, age, gender... No critical issues detected. How many rows would you like to generate?
User: 2000 rows
Agent: ✅ Synthetic data generated successfully! File ID: abc-123-def
```

### Pure Synthetic Flow
```
User: I need fake customer data for my insurance app
Agent: I'll help you create synthetic data from scratch. What type of business is this for?
User: Car insurance company in Spain
Agent: Perfect! What type of data do you need - customers, policies, claims, or something else?
User: Customer profiles
Agent: Here's what the data structure could look like: customer_id, age, gender, location, premium_annual, policy_type, risk_score. Does this meet your needs?
User: Yes, generate 500 rows
Agent: ✅ Synthetic data generated successfully! File ID: def-456-ghi
```

## 🛠️ Extending the MVP

### Adding New Data Types
1. Update `llm_generate_tool` in `tools.py`
2. Add new data structure templates
3. Update agent instructions in `agents.py`

### Adding New Models
1. Add model initialization in `tools.py`
2. Create new function tools
3. Add tools to appropriate agents

### Adding Authentication
1. Add auth middleware to `main.py`
2. Update conversation handler to include user validation
3. Add user management endpoints

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Check `.env` file exists and has correct key
   - Verify key has sufficient credits

2. **File Not Found**
   - Ensure file paths are absolute
   - Check file permissions

3. **Memory Issues with Nemotron**
   - Model loads lazily, may take time on first use
   - Reduce model size or use OpenAI fallback

4. **Session Not Found**
   - Sessions are in-memory only (MVP limitation)
   - Use persistent storage for production

### Logs
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py
```

## 📋 MVP Limitations

- **In-memory sessions** (no persistence)
- **Simple file storage** (local temp files)
- **Basic data validation** (CSV format only)
- **Single-user focused** (no authentication)
- **Simplified LLM generation** (example data for MVP)

## 🚀 Next Steps

1. **Persistent Storage**: Add database for sessions and files
2. **Authentication**: User management and API keys
3. **Advanced SDV**: More sophisticated models and parameters
4. **Real Nemotron Integration**: Full LLM data generation
5. **File Formats**: Support JSON, Parquet, databases
6. **Data Quality**: Advanced validation and cleaning
7. **Batch Processing**: Handle large datasets
8. **Monitoring**: Metrics and logging dashboard

---

**Ready to generate synthetic data!** 🎉
# LangGraph Math Agent

An intelligent agent built with LangGraph that seamlessly handles both mathematical operations and general knowledge queries using LLM reasoning.

## 🎯 Features

- **Intelligent Query Routing**: Automatically detects mathematical vs. general queries
- **Custom Math Tools**: Addition, subtraction, multiplication, and division
- **LLM Integration**: Uses Groq's meta-llama/llama-4-scout-17b-16e-instruct model
- **Error Handling**: Includes validation for edge cases (e.g., division by zero)
- **Graph-Based Architecture**: Built using LangGraph's StateGraph
- **✨ Interactive Conversations**: Full conversation memory with ongoing dialogue support
- **✨ Context-Aware Responses**: Remembers previous questions and answers
- **✨ User-Friendly Interface**: Real-time input with graceful exit options
- **✨ Graph Visualization**: Automatic generation of workflow diagram

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API key ([Get one here](https://console.groq.com/keys))

## 🚀 Quick Start

### 1. Clone or Download Files

Ensure you have the following files:
- `langgraph_math_agent.py` (main implementation)
- `requirements.txt` (dependencies)
- `.env.example` (environment template)
- `REPORT.md` (detailed documentation)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in the project directory:

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your_actual_api_key_here
```

### 4. Run the Agent

**Interactive Mode (Default):**
```bash
python langgraph_math_agent.py
```

This launches an interactive conversation where you can:
- Ask unlimited questions
- Have ongoing conversations with memory
- Exit anytime with `<QUIT>` or `Ctrl+C`

**Demo Mode:**
```bash
python langgraph_math_agent.py --demo
```

This runs predefined test queries to demonstrate capabilities.

## 💬 Interactive Conversation

The agent now supports full conversation memory! Here's an example:

```
You: What is 15 plus 7?
Agent: 15 plus 7 equals 22

You: Multiply that by 3
Agent: 22 multiplied by 3 equals 66

You: What was my first question?
Agent: Your first question was "What is 15 plus 7?"

You: What is the capital of France?
Agent: The capital of France is Paris

You: <QUIT>
👋 Thank you for chatting! Goodbye!
```

The agent remembers the entire conversation context, allowing for natural follow-up questions!

## 💡 Usage Examples

The agent can handle both mathematical and general queries with full conversation memory:

### Mathematical Queries
```
"What is 25 plus 17?"
"Calculate 100 minus 45"
"What is 8 multiplied by 7?"
"Divide 144 by 12"
```

### General Queries
```
"What is the capital of France?"
"Explain what artificial intelligence is"
"Who wrote Romeo and Juliet?"
```

### Follow-up Questions (Using Memory)
```
"What is 10 plus 5?"
→ "Now multiply that by 2"
→ "What was the first number I mentioned?"
```

## 📊 Architecture

```
START → CHATBOT (LLM) → [TOOLS or END]
                ↑            ↓
                └────────────┘
```

The agent uses a graph-based architecture:
1. **Chatbot Node**: LLM analyzes the query
2. **Conditional Routing**: Decides if tools are needed
3. **Tools Node**: Executes mathematical operations
4. **Feedback Loop**: Returns to chatbot for final response

**Visual Representation:**  
When you run the agent, it automatically generates graph visualizations in two formats:
- **Mermaid text file** (`.mmd`) - Always works, can be viewed at [mermaid.live](https://mermaid.live/edit)
- **PNG image** (`.png`) - Optional, requires system graphviz (no extra setup needed if installation succeeded) This shows the complete workflow with all nodes, edges, and routing logic.

## 📚 Documentation

For detailed implementation explanation, see [REPORT.md](REPORT.md) which includes:
- Architecture overview
- LLM integration details
- Complete code breakdown
- Program flow explanation
- Example outputs

**Having issues with graph visualization?** See [TROUBLESHOOTING_GRAPH_VIZ.md](TROUBLESHOOTING_GRAPH_VIZ.md)

## 🛠️ Technical Stack

- **LangGraph**: Graph-based agent framework
- **LangChain**: LLM integration and tool management
- **Groq API**: Fast LLM inference
- **Python**: Core implementation

## 📝 Assignment Details

This project is part of the Analytics Vidhya "Building Advanced AI Agents with LangGraph" assignment.

**Key Requirements Met:**
- ✅ Custom mathematical functions (plus, subtract, multiply, divide)
- ✅ LLM integration for general queries
- ✅ Graph-based architecture with conditional routing
- ✅ Tool binding and execution
- ✅ Error handling
- ✅ Comprehensive documentation
- ✅ **Interactive conversation mode with full memory**
- ✅ **Dynamic user input with graceful exit**
- ✅ **Context-aware responses for follow-up questions**

## 🔍 Code Structure

```
langgraph_math_agent.py
├── Custom Functions (@tool decorators)
│   ├── plus(a, b)
│   ├── subtract(a, b)
│   ├── multiply(a, b)
│   └── divide(a, b)
├── State Schema (AgentState)
├── LLM Setup (Groq with tool binding)
├── Graph Nodes
│   ├── chatbot(state)
│   └── tool_node
├── Routing Logic (should_continue)
├── Graph Construction (create_graph)
└── Demo Function (main)
```

## 🎓 Learning Outcomes

This project demonstrates:
- Building agentic systems with LangGraph
- Integrating LLMs with custom tools
- Implementing conditional routing
- Managing state in multi-step workflows
- Production-ready error handling

## 📄 License

This is an educational project for Analytics Vidhya assignment.

## 🙋 Support

For questions about the implementation, refer to the detailed [REPORT.md](REPORT.md) document.
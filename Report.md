# LangGraph Math Agent - Implementation Report

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [LLM Integration](#llm-integration)
4. [How the Agent Works](#how-the-agent-works)
5. [Interactive Conversation Mode](#interactive-conversation-mode)
6. [Code Breakdown](#code-breakdown)
7. [Program Flow](#program-flow)
8. [Example Outputs](#example-outputs)
9. [Conclusion](#conclusion)

---

## Introduction

This project implements an intelligent agent using **LangGraph** that seamlessly handles both mathematical operations and general knowledge queries. The agent leverages the power of Large Language Models (LLMs) for reasoning and decision-making, while delegating mathematical computations to custom Python functions.

### Key Features
- **Dual Functionality**: Handles both math operations and general questions
- **Intelligent Routing**: LLM decides when to use tools vs. answer directly
- **Error Handling**: Includes validation for edge cases (e.g., division by zero)
- **Modular Design**: Clean separation between LLM reasoning and tool execution

### Technology Stack
- **LangGraph**: For building the agentic workflow graph
- **LangChain**: For LLM integration and tool management
- **Groq API**: High-performance LLM inference (meta-llama/llama-4-scout-17b-16e-instruct)
- **Python**: Core implementation language

---

## Architecture Overview

The agent follows a graph-based architecture with three main components:

```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌─────────────┐
│   CHATBOT   │◄──────┐
│ (LLM Node)  │       │
└──────┬──────┘       │
       │              │
       │ (conditional │
       │   routing)   │
       │              │
   ┌───▼────┬────┐    │
   │        │    │    │
   ▼        ▼    │    │
┌──────┐  ┌───┐ │    │
│TOOLS │  │END│ │    │
└───┬──┘  └───┘ │    │
    │           │    │
    └───────────┴────┘
```

### Graph Components

1. **Chatbot Node**: The LLM reasoning engine that analyzes queries and decides on actions
2. **Tools Node**: Executes mathematical operations (plus, subtract, multiply, divide)
3. **Conditional Edge**: Routes flow based on whether tools are needed
4. **State**: Maintains conversation history and message flow

---

## LLM Integration

### Groq API Setup

We use **Groq** as our LLM provider for its:
- Fast inference speed (optimal for agentic workflows)
- Excellent tool-calling capabilities
- Reliable performance with Llama models

**Model Used**: `meta-llama/llama-4-scout-17b-16e-instruct`

This model was specifically chosen because it:
- Supports tool calling natively
- Has strong reasoning capabilities
- Provides accurate function parameter extraction
- Is currently active and supported (unlike deprecated models)

### Tool Binding

The LLM is bound to our custom mathematical tools using LangChain's tool binding mechanism:

```python
llm_with_tools = llm.bind_tools(tools)
```

This enables the LLM to:
1. Understand available tools and their parameters
2. Decide when to call tools based on user queries
3. Generate properly formatted tool calls
4. Interpret tool results and formulate final answers

---

## How the Agent Works

### Step-by-Step Workflow

#### 1. **User Query Reception**
The agent receives a user query (e.g., "What is 25 plus 17?" or "What is AI?")

#### 2. **LLM Analysis (Chatbot Node)**
The LLM analyzes the query to determine:
- Is this a mathematical operation?
- Which tool (if any) should be called?
- What are the parameters?

#### 3. **Decision Point (Conditional Edge)**

**Scenario A: Mathematical Query**
- LLM identifies mathematical intent
- Generates tool calls with extracted parameters
- Routes to Tools Node

**Scenario B: General Query**
- LLM recognizes non-mathematical nature
- Provides direct answer from its knowledge
- Routes to END

#### 4. **Tool Execution (Tools Node)**
If tools are needed:
- The ToolNode executes the appropriate function
- Returns the computation result
- Routes back to Chatbot Node

#### 5. **Final Response**
- LLM receives tool results
- Formulates a natural language response
- Presents answer to the user

### Example Flows

**Mathematical Query Flow:**
```
User: "What is 25 plus 17?"
  ↓
Chatbot: Identifies "plus" operation, extracts (25, 17)
  ↓
Tools: Executes plus(25, 17) → 42
  ↓
Chatbot: Formats response "25 plus 17 equals 42"
  ↓
User sees: "25 plus 17 equals 42"
```

**General Query Flow:**
```
User: "What is the capital of France?"
  ↓
Chatbot: Recognizes no tools needed, uses knowledge
  ↓
User sees: "The capital of France is Paris"
```

---

## Interactive Conversation Mode

### Overview

The agent now supports **interactive conversations** with full memory retention. Users can have ongoing dialogues where the agent remembers previous questions and answers, making the interaction feel natural and contextual.

### Key Features

**1. Conversation Memory**
- The agent maintains full conversation history across multiple queries
- Uses LangGraph's `MessagesState` to preserve context
- Each new query adds to the existing conversation thread
- LLM can reference previous interactions

**2. Dynamic User Input**
- Real-time input from users (not hardcoded queries)
- Continuous conversation loop
- Graceful exit with `<QUIT>` command
- Error handling with conversation continuation

**3. Context-Aware Responses**
- Agent remembers what was discussed earlier
- Can answer follow-up questions naturally
- Example: "What is 5 plus 3?" → "Now multiply that by 2"

### How Conversation Memory Works

**State Persistence:**
```python
# Initialize empty state
conversation_state = None

# First query creates the state
response, conversation_state = run_agent("What is 5 plus 3?", graph, None)
# State now contains: [HumanMessage, AIMessage, ToolMessage, AIMessage]

# Second query appends to existing state
response, conversation_state = run_agent("Now add 10 to that", graph, conversation_state)
# State now contains: [previous messages..., HumanMessage, AIMessage, ToolMessage, AIMessage]
```

**Message Flow:**
1. User enters query → Appended as `HumanMessage` to existing state
2. Graph processes entire conversation history
3. LLM sees all previous context when reasoning
4. New response appended to conversation
5. Updated state returned for next iteration

### Interactive Loop Structure

```python
conversation_state = None  # Initialize

while True:
    user_input = input("You: ")
    
    if user_input == "<QUIT>":
        break
    
    # Pass conversation_state to maintain memory
    response, conversation_state = run_agent(user_input, graph, conversation_state)
    
    print(f"Agent: {response}")
```

### Example Conversation with Memory

```
You: What is 10 plus 5?
Agent: 10 plus 5 equals 15

You: Now multiply that by 3
Agent: 15 multiplied by 3 equals 45

You: What was my first question?
Agent: Your first question was "What is 10 plus 5?"

You: What is the capital of France?
Agent: The capital of France is Paris

You: Can you remind me what we were calculating before?
Agent: Before asking about Paris, we were doing math. You asked for 10 plus 5 (which equals 15), then asked to multiply that result by 3 (which equals 45).
```

### Usage Modes

**Interactive Mode (Default):**
```bash
python langgraph_math_agent.py
```
- Launches conversation interface
- User can ask unlimited questions
- Exit with `<QUIT>` or `Ctrl+C`

**Demo Mode:**
```bash
python langgraph_math_agent.py --demo
```
- Runs predefined test queries
- Shows agent capabilities
- Useful for testing and demonstrations

---

## Code Breakdown

### 1. Custom Mathematical Functions

```python
@tool
def plus(a: str, b: str) -> float:
    """Add two numbers together."""
    return float(a) + float(b)
```

Each function is decorated with `@tool` which:
- Converts it into a LangChain Tool
- Automatically generates a schema for the LLM
- Includes docstrings for LLM context

**Important Design Decision:**
The functions accept `string` parameters instead of `float` because LLMs often pass numeric values as strings in tool calls. By accepting strings and converting them internally with `float()`, we make the tools more robust and compatible with various LLM behaviors. This is a common production pattern for LLM-powered tools.

**Error Handling Example:**
```python
@tool
def divide(a: str, b: str) -> float:
    """Divide first number by second number."""
    num_a = float(a)
    num_b = float(b)
    if num_b == 0:
        raise ValueError("Cannot divide by zero!")
    return num_a / num_b
```

### 2. State Schema

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

**Purpose:**
- Maintains conversation history
- Uses `add_messages` reducer to append new messages
- Enables the agent to remember context

### 3. Chatbot Node

```python
def chatbot(state: AgentState) -> AgentState:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

**Functionality:**
- Receives current state (message history)
- Invokes LLM with tool bindings
- Returns updated state with LLM response
- LLM response may contain tool calls or direct answers

### 4. Tool Node

```python
tool_node = ToolNode(tools=tools)
```

**Purpose:**
- Pre-built LangGraph component
- Automatically executes tool calls from LLM
- Handles tool invocation and result formatting
- Returns ToolMessage objects with results

### 5. Conditional Routing

```python
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"
```

**Logic:**
- Checks last message for tool calls
- Routes to "tools" if tool calls present
- Routes to "end" if no tools needed

### 6. Graph Construction

```python
workflow = StateGraph(AgentState)
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", should_continue, {...})
workflow.add_edge("tools", "chatbot")
graph = workflow.compile()
```

**Structure:**
- Defines nodes (chatbot, tools)
- Connects START to chatbot
- Adds conditional routing from chatbot
- Creates feedback loop: tools → chatbot
- Compiles into executable graph

---

## Program Flow

### Detailed Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                               │
├─────────────────────────────────────────────────────────────────┤
│ • Load environment variables (GROQ_API_KEY)                     │
│ • Define custom math tools (@tool decorators)                   │
│ • Initialize Groq LLM with meta-llama/llama-4-scout model      │
│ • Bind tools to LLM (llm.bind_tools)                           │
│ • Create StateGraph with AgentState schema                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. USER INPUT PROCESSING                                        │
├─────────────────────────────────────────────────────────────────┤
│ • User submits query                                            │
│ • Create initial state: {"messages": [HumanMessage(query)]}    │
│ • Invoke graph with initial state                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. CHATBOT NODE EXECUTION                                       │
├─────────────────────────────────────────────────────────────────┤
│ • LLM receives message history                                  │
│ • Analyzes query intent and context                             │
│ • Decision:                                                     │
│   → Math query? Extract params, generate tool calls             │
│   → General query? Prepare direct answer                        │
│ • Return AIMessage (with or without tool calls)                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. CONDITIONAL ROUTING                                          │
├─────────────────────────────────────────────────────────────────┤
│ • should_continue() checks last message                         │
│ • Has tool_calls?                                               │
│   → YES: Route to "tools" node                                  │
│   → NO: Route to "end"                                          │
└─────────────────────────────────────────────────────────────────┘
         ↓                                    ↓
    [TO TOOLS]                            [TO END]
         ↓                                    ↓
┌──────────────────────────┐    ┌──────────────────────────┐
│ 5a. TOOLS NODE           │    │ 5b. END                  │
├──────────────────────────┤    ├──────────────────────────┤
│ • Execute tool function  │    │ • Return final state     │
│ • Calculate result       │    │ • Extract final message  │
│ • Create ToolMessage     │    │ • Display to user        │
│ • Append to state        │    └──────────────────────────┘
└──────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. RETURN TO CHATBOT (after tool execution)                     │
├─────────────────────────────────────────────────────────────────┤
│ • LLM receives tool results                                     │
│ • Formulates natural language response                          │
│ • Returns final AIMessage (without tool calls)                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. FINAL ROUTING TO END                                         │
├─────────────────────────────────────────────────────────────────┤
│ • should_continue() detects no tool calls                       │
│ • Routes to END                                                 │
│ • Graph execution completes                                     │
│ • Return final state with complete message history              │
└─────────────────────────────────────────────────────────────────┘
```

### State Evolution Example

**Query:** "What is 15 multiplied by 3?"

```python
# State 1: Initial
{
    "messages": [
        HumanMessage(content="What is 15 multiplied by 3?")
    ]
}

# State 2: After Chatbot (with tool call)
{
    "messages": [
        HumanMessage(content="What is 15 multiplied by 3?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "multiply", "args": {"a": 15, "b": 3}}]
        )
    ]
}

# State 3: After Tools
{
    "messages": [
        HumanMessage(content="What is 15 multiplied by 3?"),
        AIMessage(content="", tool_calls=[...]),
        ToolMessage(content="45", tool_call_id="...")
    ]
}

# State 4: After Final Chatbot
{
    "messages": [
        HumanMessage(content="What is 15 multiplied by 3?"),
        AIMessage(content="", tool_calls=[...]),
        ToolMessage(content="45", tool_call_id="..."),
        AIMessage(content="15 multiplied by 3 equals 45")
    ]
}
```

---

## Example Outputs

### Mathematical Queries

**Addition:**
```
Query: What is 25 plus 17?
Agent Response: 25 plus 17 equals 42
```

**Subtraction:**
```
Query: Calculate 100 minus 45
Agent Response: 100 minus 45 equals 55
```

**Multiplication:**
```
Query: What is 8 multiplied by 7?
Agent Response: 8 multiplied by 7 equals 56
```

**Division:**
```
Query: Divide 144 by 12
Agent Response: 144 divided by 12 equals 12
```

**Error Handling:**
```
Query: What is 5 divided by 0?
Agent Response: Error: Cannot divide by zero!
```

### General Knowledge Queries

**Geography:**
```
Query: What is the capital of France?
Agent Response: The capital of France is Paris
```

**Technology:**
```
Query: Explain what artificial intelligence is
Agent Response: Artificial Intelligence (AI) refers to the simulation 
of human intelligence in machines that are programmed to think and 
learn like humans. It includes machine learning, natural language 
processing, and computer vision.
```

**Literature:**
```
Query: Who wrote Romeo and Juliet?
Agent Response: Romeo and Juliet was written by William Shakespeare
```

---

## Conclusion

### Key Achievements

1. **Successful Integration**: Seamlessly combined LLM reasoning with custom tool execution
2. **Intelligent Routing**: Agent correctly identifies when to use tools vs. answer directly
3. **Error Handling**: Implemented proper error handling for edge cases
4. **Clean Architecture**: Modular, maintainable code following LangGraph best practices
5. **✨ Interactive Conversations**: Full conversation memory with ongoing dialogue support
6. **✨ User-Friendly Interface**: Real-time input with graceful exit options

### Technical Insights

**Why LangGraph?**
- Provides structured way to build agentic workflows
- Native support for conditional routing
- Built-in state management with automatic message persistence
- Easy to visualize and debug

**Why Tool Binding?**
- Enables LLM to understand available functions
- Automatic parameter extraction from natural language
- Type-safe function calls (with string-to-float conversion for robustness)
- Structured output format

**Why Conversation Memory?**
- Enhances user experience with context-aware responses
- Enables follow-up questions without repetition
- Makes the agent feel more natural and intelligent
- Achieved through state persistence across graph invocations

### Potential Enhancements

1. **Persistent Storage**: Save conversation history to database for long-term retention
2. **Complex Operations**: Support multi-step calculations (e.g., "Add 5 and 3, then multiply by 2")
3. **More Tools**: Extend with trigonometry, statistics, calculus functions
4. **Conversation Export**: Allow users to save conversation transcripts
5. **UI Enhancement**: Build a web interface using Streamlit or Gradio
6. **Multi-turn Planning**: Enable agent to plan and execute complex multi-step tasks

### Usage Instructions

**Setup:**
```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export GROQ_API_KEY="your-api-key-here"
# Or create .env file with: GROQ_API_KEY=your-api-key-here
```

**Run Interactive Mode (Default):**
```bash
python langgraph_math_agent.py
```
This launches an interactive conversation where you can:
- Ask unlimited questions
- Have the agent remember previous interactions
- Exit anytime with `<QUIT>` or `Ctrl+C`

**Run Demo Mode:**
```bash
python langgraph_math_agent.py --demo
```
This runs predefined queries to demonstrate capabilities.

**Example Interactive Session:**
```
You: What is 25 plus 17?
Agent: 25 plus 17 equals 42

You: Multiply that by 2
Agent: 42 multiplied by 2 equals 84

You: What is the capital of Japan?
Agent: The capital of Japan is Tokyo

You: <QUIT>
👋 Thank you for chatting! Goodbye!
```

### Learning Outcomes

This assignment demonstrates:
- Building agentic systems with LangGraph
- Integrating LLMs with custom tools
- Implementing conditional routing logic
- Managing state in multi-step workflows
- Error handling in production-ready agents

---

**Assignment Completed By**: Analytics Vidhya Student  
**Framework**: LangGraph + LangChain  
**LLM Provider**: Groq (meta-llama/llama-4-scout-17b-16e-instruct)  
**Date**: October 2025
"""
LangGraph Math Agent
====================
An intelligent agent that routes queries to either custom mathematical functions
or an LLM for general knowledge questions using LangGraph.

Author: Analytics Vidhya Assignment
Model: meta-llama/llama-4-scout-17b-16e-instruct (Groq)
"""

import os
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# STEP 1: Define Custom Mathematical Functions
# ============================================================================

@tool
def plus(a: str, b: str) -> float:
    """Add two numbers together.
    
    Args:
        a: First number (as string)
        b: Second number (as string)
    
    Returns:
        The sum of a and b
    """
    return float(a) + float(b)


@tool
def subtract(a: str, b: str) -> float:
    """Subtract second number from first number.
    
    Args:
        a: First number (as string)
        b: Second number (as string)
    
    Returns:
        The difference (a - b)
    """
    return float(a) - float(b)


@tool
def multiply(a: str, b: str) -> float:
    """Multiply two numbers together.
    
    Args:
        a: First number (as string)
        b: Second number (as string)
    
    Returns:
        The product of a and b
    """
    return float(a) * float(b)


@tool
def divide(a: str, b: str) -> float:
    """Divide first number by second number.
    
    Args:
        a: First number (numerator, as string)
        b: Second number (denominator, as string)
    
    Returns:
        The quotient (a / b)
    
    Raises:
        ValueError: If b is zero
    """
    num_a = float(a)
    num_b = float(b)
    if num_b == 0:
        raise ValueError("Cannot divide by zero!")
    return num_a / num_b


# ============================================================================
# STEP 2: Define State Schema
# ============================================================================

class AgentState(TypedDict):
    """State schema for the agent graph.
    
    Attributes:
        messages: List of messages in the conversation, with add_messages reducer
    """
    messages: Annotated[list, add_messages]


# ============================================================================
# STEP 3: Initialize LLM with Tools
# ============================================================================

# List of all mathematical tools
tools = [plus, subtract, multiply, divide]

# Initialize Groq LLM with tool binding
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Bind tools to the LLM so it can call them
llm_with_tools = llm.bind_tools(tools)


# ============================================================================
# STEP 4: Define Graph Nodes
# ============================================================================

def chatbot(state: AgentState) -> AgentState:
    """Chatbot node that uses LLM to reason and decide on tool usage.
    
    This node:
    1. Receives the current state with messages
    2. Passes messages to LLM
    3. LLM decides whether to call tools or answer directly
    4. Returns updated state with LLM response
    
    Args:
        state: Current agent state containing messages
    
    Returns:
        Updated state with LLM response added to messages
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Tool node is created using LangGraph's ToolNode helper
# It automatically executes tool calls and returns results
tool_node = ToolNode(tools=tools)


# ============================================================================
# STEP 5: Define Conditional Routing Logic
# ============================================================================

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Conditional edge function to route between tools and end.
    
    This function checks if the last message contains tool calls:
    - If yes: Route to "tools" node to execute the tools
    - If no: Route to "end" to finish the conversation
    
    Args:
        state: Current agent state
    
    Returns:
        "tools" if tool calls are present, "end" otherwise
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# ============================================================================
# STEP 6: Build the Graph
# ============================================================================

def create_graph():
    """Create and compile the LangGraph agent.
    
    Graph structure:
        START -> chatbot -> [tools -> chatbot] or END
    
    Flow:
    1. User input goes to chatbot node
    2. Chatbot (LLM) decides if tools are needed
    3. If tools needed: execute tools, return to chatbot for final answer
    4. If no tools needed: directly end with LLM's answer
    
    Returns:
        Compiled graph ready for execution
    """
    # Initialize the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes to the graph
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "chatbot")  # Start goes to chatbot
    
    # Conditional edge: chatbot decides next step
    workflow.add_conditional_edges(
        "chatbot",
        should_continue,
        {
            "tools": "tools",  # If tools needed, go to tools node
            "end": END         # If no tools needed, end
        }
    )
    
    # After tools execute, return to chatbot for final answer
    workflow.add_edge("tools", "chatbot")
    
    # Compile the graph
    graph = workflow.compile()
    
    return graph


def save_graph_visualization(graph, output_path="./graph_architecture"):
    """Save the LangGraph visualization as both PNG (if possible) and Mermaid text.
    
    Args:
        graph: Compiled LangGraph agent
        output_path: Path where to save the visualization (without extension)
    
    Returns:
        Tuple of (png_path or None, mermaid_path)
    """
    png_path = None
    mermaid_path = f"{output_path}.mmd"
    
    # Always generate Mermaid text (works without any special dependencies)
    try:
        mermaid_text = graph.get_graph().draw_mermaid()
        with open(mermaid_path, "w") as f:
            f.write(mermaid_text)
        print(f"✓ Graph Mermaid diagram saved to: {mermaid_path}")
        print(f"  View online at: https://mermaid.live/edit")
    except Exception as e:
        print(f"⚠️  Could not generate Mermaid text: {str(e)}")
        return None, None
    
    # Try to generate PNG (optional, requires additional dependencies)
    try:
        png_path = f"{output_path}.png"
        png_data = graph.get_graph().draw_mermaid_png()
        with open(png_path, "wb") as f:
            f.write(png_data)
        print(f"✓ Graph PNG image saved to: {png_path}")
    except Exception as e:
        print(f"ℹ️  PNG generation skipped (optional): {str(e)}")
        print(f"   You can visualize the .mmd file at https://mermaid.live/edit")
        png_path = None
    
    return png_path, mermaid_path


# ============================================================================
# STEP 7: Demo and Testing
# ============================================================================

def run_agent(query: str, graph, conversation_state: dict = None):
    """Run the agent with a user query while maintaining conversation history.
    
    Args:
        query: User's question or request
        graph: Compiled LangGraph agent
        conversation_state: Current conversation state (maintains memory)
    
    Returns:
        Tuple of (response, updated_state)
    """
    # If this is the first query, create initial state
    if conversation_state is None:
        conversation_state = {"messages": []}
    
    # Add the new user message to the conversation
    conversation_state["messages"].append(HumanMessage(content=query))
    
    # Run the graph with the full conversation history
    result = graph.invoke(conversation_state)
    
    # Extract the final response
    final_message = result["messages"][-1]
    response = final_message.content
    
    # Return both the response and the updated state
    return response, result


def main():
    """Main function for interactive conversation with the agent."""
    
    print("\n" + "="*70)
    print("🤖 LangGraph Math Agent - Interactive Mode")
    print("="*70)
    print("\nInitializing agent with Groq LLM (meta-llama/llama-4-scout-17b-16e-instruct)...")
    
    # Create the graph
    graph = create_graph()
    print("✓ Graph created successfully!")
    
    # Save graph visualization
    save_graph_visualization(graph)
    
    print("\n" + "="*70)
    print("INSTRUCTIONS:")
    print("- Ask me mathematical questions (e.g., 'What is 5 plus 3?')")
    print("- Ask me general questions (e.g., 'What is the capital of France?')")
    print("- I'll remember our conversation context!")
    print("- Type '<QUIT>' to exit the conversation")
    print("="*70 + "\n")
    
    # Initialize conversation state (maintains memory across queries)
    conversation_state = None
    
    # Start interactive conversation loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for quit command
            if user_input.upper() == "<QUIT>" or user_input.upper() == "QUIT":
                print("\n" + "="*70)
                print("👋 Thank you for chatting! Goodbye!")
                print("="*70 + "\n")
                break
            
            # Skip empty inputs
            if not user_input:
                print("⚠️  Please enter a question or type '<QUIT>' to exit.\n")
                continue
            
            # Run the agent with conversation memory
            print("\n🤔 Processing...\n")
            response, conversation_state = run_agent(user_input, graph, conversation_state)
            
            # Display the agent's response
            print(f"Agent: {response}\n")
            print("-" * 70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("👋 Conversation interrupted. Goodbye!")
            print("="*70 + "\n")
            break
        
        except Exception as e:
            print(f"\n❌ Error occurred: {str(e)}\n")
            print("Please try again or type '<QUIT>' to exit.\n")
            # Continue the loop even after errors


def demo_mode():
    """Demo function to demonstrate the agent with predefined queries."""
    
    print("\n" + "="*70)
    print("🤖 LangGraph Math Agent - Demo Mode")
    print("="*70)
    print("\nInitializing agent with Groq LLM (meta-llama/llama-4-scout-17b-16e-instruct)...")
    
    # Create the graph
    graph = create_graph()
    print("✓ Graph created successfully!")
    
    # Save graph visualization
    save_graph_visualization(graph)
    
    # Test queries
    test_queries = [
        # Mathematical queries
        "What is 25 plus 17?",
        "Calculate 100 minus 45",
        "What is 8 multiplied by 7?",
        "Divide 144 by 12",
        
        # General knowledge queries
        "What is the capital of France?",
        "Explain what artificial intelligence is in one sentence",
    ]
    
    print("\n" + "="*70)
    print("Running Demo Queries...")
    print("="*70)
    
    conversation_state = None
    
    for query in test_queries:
        try:
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print(f"{'='*70}")
            
            response, conversation_state = run_agent(query, graph, conversation_state)
            
            print(f"\nAgent Response: {response}")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"Error occurred: {str(e)}\n")
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    # Check if user wants demo mode
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_mode()
    else:
        # Default to interactive mode
        main()
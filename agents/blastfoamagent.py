import os
import sys
from typing import List

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

@tool
def create_directory(path: str) -> str:
    """Creates a directory at the specified path. Returns status message."""
    import json
    try:
        # Handle case where agent provides JSON string instead of raw path
        if path.strip().startswith('{'):
            try:
                data = json.loads(path)
                if 'path' in data:
                    path = data['path']
            except:
                pass
        
        # Strip quotes if present (common LLM mistake)
        path = path.strip().strip('"').strip("'")

        os.makedirs(path, exist_ok=True)
        return f"Successfully created directory: {path}"
    except Exception as e:
        return f"Failed to create directory {path}: {str(e)}"

@tool
def write_file(data: str) -> str:
    """Writes content to a file. Input must be a JSON string with 'file_path' and 'content' keys."""
    import json
    try:
        # Clean up the input string
        cleaned_data = data.strip()
        # Remove markdown code blocks if present
        if cleaned_data.startswith("```") and cleaned_data.endswith("```"):
            cleaned_data = cleaned_data.strip("`")
            if cleaned_data.startswith("json"):
                cleaned_data = cleaned_data[4:]
            cleaned_data = cleaned_data.strip()
        
        # Remove surrounding quotes if it's a stringified JSON
        if (cleaned_data.startswith('"') and cleaned_data.endswith('"')) or \
           (cleaned_data.startswith("'") and cleaned_data.endswith("'")):
            cleaned_data = cleaned_data[1:-1]

        # Parse the input JSON string
        params = json.loads(cleaned_data)
        file_path = params.get('file_path')
        content = params.get('content')
        
        if not file_path or content is None:
            return "Error: Input must contain 'file_path' and 'content'"

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to file: {file_path}"
    except json.JSONDecodeError:
        return f"Error: Input was not a valid JSON string. Received: {data[:100]}..."
    except Exception as e:
        return f"Failed to write to file: {str(e)}"

class BlastFoamAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            temperature=0
        )
        
        self.tools = [create_directory, write_file]
        
        # Define the prompt
        template = '''You are an expert OpenFOAM/BlastFoam simulation engineer.
Your task is to generate simulation cases based on user requirements and reference cases.
You have access to the file system to create directories and write configuration files.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action.
- For create_directory: provide ONLY the directory path as a string (e.g., /path/to/dir).
- For write_file: provide a JSON string with "file_path" and "content" keys.

Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Context from RAG (Reference Case):
{context}

Previous Conversation:
{chat_history}

User Request: {input}
Thought:{agent_scratchpad}'''

        self.prompt = PromptTemplate.from_template(template)
        
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=config.MAX_ITERATIONS
        )

    def run(self, user_request: str, context: str, chat_history: str = ""):
        """
        Run the agent with the given user request, context, and chat history.
        """
        print(f"Agent starting with request: {user_request}")
        try:
            result = self.agent_executor.invoke({
                "input": user_request, 
                "context": context,
                "chat_history": chat_history
            })
            return result["output"]
        except Exception as e:
            return f"Error executing agent: {str(e)}"

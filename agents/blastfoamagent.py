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
        if path.strip().startswith('{'):
            try:
                data = json.loads(path)
                if 'path' in data:
                    path = data['path']
            except:
                pass
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
        cleaned_data = data.strip()
        if cleaned_data.startswith("```") and cleaned_data.endswith("```"):
            cleaned_data = cleaned_data.strip("`")
            if cleaned_data.startswith("json"):
                cleaned_data = cleaned_data[4:]
            cleaned_data = cleaned_data.strip()
        
        if (cleaned_data.startswith('"') and cleaned_data.endswith('"')) or \
           (cleaned_data.startswith("'") and cleaned_data.endswith("'")):
            cleaned_data = cleaned_data[1:-1]

        params = json.loads(cleaned_data)
        file_path = params.get('file_path')
        content = params.get('content')
        
        if not file_path or content is None:
            return "Error: Input must contain 'file_path' and 'content'"

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
        
        template = '''You are an expert OpenFOAM/BlastFoam simulation engineer.
Your task is to generate complete, RUNNABLE simulation cases based on user requirements and reference cases.
You have access to the file system to create directories and write configuration files.

CRITICAL RULES FOR BLASTFOAM CASE GENERATION:

1. Mesh & Geometry (snappyHexMesh):
   - For 3D cases, NEVER use `empty` type for Z-faces in blockMeshDict. Use `patch`, `wall`, or `symmetry`.
   - If a building/STL is involved, there is an existing STL file at `../stl/L_Wall.stl` (relative to your case root).
   - In your `Allrun` script, ALWAYS add the following lines BEFORE snappyHexMesh:
     `mkdir -p constant/geometry`
     `cp ../stl/L_Wall.stl constant/geometry/`
   - Ensure snappyHexMeshDict correctly references this stl file.
   - **CRITICAL**: Your `system/snappyHexMeshDict` MUST contain the `addLayersControls {{ ... }}` block, as well as `castellatedMeshControls`, `snapControls`, and `meshQualityControls`. Even if you use `addLayers false;`, the OpenFOAM parser will throw a FATAL ERROR if `addLayersControls` is completely missing.

2. Initial Fields (setFields):
   - You MUST generate a `system/setFieldsDict`.
   - If the user does not specify explosive details, assume C4 at position (0, 0, 0) with a reasonable radius (e.g., 0.5m or according to building scale).
   - Use `volScalarFieldValue` for `alpha.C4` and `alpha.Air` and `e` (internal energy) in setFieldsDict.

3. Completeness of 0/ Directory (Solver Crash Prevention):
   - blastFoam multi-phase simulations strictly require specific files in the `0` directory.
   - You MUST ALWAYS generate these files in `0/`: 
     `p`, `U`, `T`, `e`, `alpha.C4`, `alpha.Air`, AND **`rho.C4`**, **`rho.Air`** (or whatever phases you defined in constant/phaseProperties).
   - Missing `rho.C4` or `rho.Air` will automatically crash phase fraction/density calculations. Just use a basic uniform internal field based on standard density (e.g. 1.225 for Air, 1630 for C4) with zeroGradient at boundaries.
   - `alpha.*` files must sum to 1. E.g., internalField for alpha.Air = 1, alpha.C4 = 0.
   - Temperature `T` must be set (e.g., uniform 300).
   - Ensure boundary conditions match `blockMeshDict`.

4. System Configuration:
   - In `system/fvSchemes` -> `ddtSchemes`, you MUST include `timeIntegrator Euler;` along with `default Euler;`.
   - Ensure `constant/phaseProperties` exists and defines phases correctly (e.g., C4 and Air) along with their equationOfState.

5. Scripts:
   - `Allrun` MUST be executable, begin with cleanup (`rm -rf log.*`), and correctly copy STL before mesh generation.
   - `Allclean` MUST be executable and only remove logs, processor*, mesh, and [1-9]* directories, preserving 0/.

You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (- For write_file: provide a JSON string with "file_path" and "content" keys)
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

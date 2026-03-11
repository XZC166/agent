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
        if content and '\n' in content:
            content = content.replace('\n', '\n')
        
        if not file_path or content is None:
            return "Error: Input must contain 'file_path' and 'content'"

        import os
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

1. System Dictionary Completeness:
   - EVERY OpenFOAM case MUST HAVE a `system/controlDict` file. 
   - You MUST generate `system/fvSchemes` and `system/fvSolution`. In `system/fvSchemes`, you MUST carefully include `fluxScheme` at the root dictionary level (e.g. `fluxScheme Kurganov;`).

- **CRITICAL RIEMANN LIMITERS**: To prevent `Floating point exception` under `Riemann fluxes are used, but no limiter is specified`, you MUST add explicit reconstruction items under `interpolationSchemes` inside `system/fvSchemes`:
    ```
    interpolationSchemes
    {{
        default         linear;
        reconstruct(rho) vanLeer;
        reconstruct(U)   vanLeerV;
        reconstruct(p)   vanLeer;
        reconstruct(e)   vanLeer;
        reconstruct(speedOfSound)  vanLeer;
        reconstruct(alpha) vanLeer;
    }}
    ```
 In `system/fvSchemes`, you MUST carefully include `fluxScheme` at the root dictionary level (e.g. `fluxScheme Kurganov;`).

- **CRITICAL RIEMANN LIMITERS**: To prevent `Floating point exception` under `Riemann fluxes are used, but no limiter is specified`, you MUST add explicit reconstruction items under `interpolationSchemes` inside `system/fvSchemes`:
    ```
    interpolationSchemes
    {{
        default         linear;
        reconstruct(rho) vanLeer;
        reconstruct(U)   vanLeerV;
        reconstruct(p)   vanLeer;
        reconstruct(e)   vanLeer;
        reconstruct(speedOfSound)  vanLeer;
        reconstruct(alpha) vanLeer;
    }}
    ```
 In `system/fvSchemes`, you MUST include `fluxScheme Kurganov;` at the root dictionary level. In `system/fvSchemes` -> `ddtSchemes`, you MUST include `timeIntegrator Euler;` along with `default Euler;`.
   - **CRITICAL**: The `constant/phaseProperties` file MUST strictly follow blastFoam format. DO NOT NEST dictionary blocks loosely.

   - **CRITICAL THERMOTYPE RULES**: In BlastFoam, `constant/phaseProperties` is different from standard OpenFOAM.
   - For an EXPLOSIVE phase (e.g. `C4`), the top-level block MUST specify `type detonating;` and MUST contain BOTH `reactants` and `products` sub-dictionaries.
   - For a BACKGROUND phase (e.g. `Air`), the top-level block MUST specify `type basic;` and its properties are placed directly inside.
   - Example MUST BE FOLLOWED EXACTLY for detonating material (C4):
     ```
     phases (C4 Air);

     C4
     {{
         type            detonating;
         reactants
         {{
             thermoType {{ transport const; thermo eConst; equationOfState Murnaghan; specie specie; }}
             equationOfState {{ type Murnaghan; rho0 1601; Gamma 0.25; pRef 101298; K0 8.04e9; K0Prime 7.97; }}
             specie {{ molWeight 55.0; }}
             transport {{ mu 0; Pr 1; }}
             thermodynamics {{ type eConst; Cv 1400; Hf 0; }}
         }}
         products
         {{
             thermoType {{ transport const; thermo ePolynomial; equationOfState JWL; specie specie; }}
             equationOfState {{ type JWL; rho0 1601; A 609.77e9; B 12.95e9; R1 4.5; R2 1.4; omega 0.25; }}
             specie {{ molWeight 55.0; }}
             transport {{ mu 0; Pr 1; }}
             thermodynamics {{ CvCoeffs<8> (413.15 2.1538 0 0 0 0 0 0); Sf 0; Hf 0; }}
         }}
         activationModel none; // or pressureBased, etc.
         initiation {{ E0 9.0e9; }}
         residualRho 1e-6;
         residualAlpha 1e-6;
     }}
     ```
   - Example MUST BE FOLLOWED EXACTLY for background material (Air):
     ```
     Air
     {{
         type            basic;
         thermoType {{ transport const; thermo eConst; equationOfState idealGas; specie specie; }}
         equationOfState {{ type idealGas; gamma 1.4; }}
         specie {{ molWeight 28.97; }}
         transport {{ mu 0; Pr 1; }}
         thermodynamics {{ type eConst; Cv 718; Hf 0; }}
         residualRho 1e-6;
         residualAlpha 1e-6;
     }}
     ```
   - DO NOT USE `state phaseFluid;` or `type phaseFluid;` inside `thermoType` or at the root. Use exactly `type detonating;` and `type basic;` at the root, and only `transport, thermo, equationOfState, specie` in `thermoType`.

2. Mesh & Geometry (blockMeshDict & snappyHexMesh):
   - You MUST generate `system/blockMeshDict`, `system/snappyHexMeshDict` and `system/surfaceFeaturesDict`.
   - In your `blockMeshDict`, the standard vertices for a hex block are: bottom(-z): (0 1 2 3), top(+z): (4 5 6 7). Standard valid faces: inlet(0 4 7 3), outlet(1 2 6 5), front(0 1 5 4), back(3 7 6 2), bottom(0 3 2 1), top(4 5 6 7). NEVER invent non-existent face vertex combinations.
   - For 3D cases, NEVER use `empty` type for Z-faces in blockMeshDict. Use `patch`, `wall`, or `symmetry`.
   - **CRITICAL**: Your `system/snappyHexMeshDict` MUST contain the `meshQualityControls {{ ... }}` block, ALONG WITH `addLayersControls {{ ... }}`, `castellatedMeshControls`, and `snapControls`.
   - **CRITICAL**: If you specify `features` in `castellatedMeshControls`, you MUST run `surfaceFeatures` (or `surfaceFeatureExtract`) before `snappyHexMesh`.
     If you run `surfaceFeatures`, you MUST ALSO generate the `system/surfaceFeaturesDict` file!
     - **CRITICAL STL RULE**: If your `snappyHexMeshDict` or `surfaceFeaturesDict` uses `.stl` geometry files (like `L_Wall.stl` or similar), YOU MUST GENERATE the ASCII `.stl` content yourself using the tool and save it to `constant/geometry/`. Without this actual STL file, `snappyHexMesh` will crash.
     - **CRITICAL setFields RULE**: When configuring `setFieldsDict` for the explosive region (e.g. a cylinder of radius 0.1m), your `blockMesh` cell size MUST be small enough to capture it (e.g. cell size <= 0.05m). If your cells are too large, `setFields` will skip it without error, and your explosion will quietly fail (max pressure will stay exactly at ambient).
     - **CRITICAL STL RULE**: If your `snappyHexMeshDict` or `surfaceFeaturesDict` uses `.stl` geometry files (like `L_Wall.stl` or similar), YOU MUST GENERATE the ASCII `.stl` content yourself using the tool and save it to `constant/geometry/`. Without this actual STL file, `snappyHexMesh` will crash.
     - **CRITICAL setFields RULE**: When configuring `setFieldsDict` for the explosive region (e.g. a cylinder of radius 0.1m), your `blockMesh` cell size MUST be small enough to capture it (e.g. cell size <= 0.05m). If your cells are too large, `setFields` will skip it without error, and your explosion will quietly fail (max pressure will stay exactly at ambient).
     CRITICAL: In OpenFOAM 9, the `system/surfaceFeaturesDict` MUST define `surfaces` as a dictionary with `extractionMethod` and `includedAngle`!
     Example correct format for OpenFOAM 9 in `system/surfaceFeaturesDict`:
     ```
     surfaces
     {{
         "L_Wall.stl"
         {{
             extractionMethod    extractFromSurface;
             includedAngle       150;
         }}
     }}
     ```
     You MUST include `includedAngle` under the surface name! Do NOT use a list of strings!

3. Initial Fields (setFields):
   - You MUST generate a `system/setFieldsDict` with `volScalarFieldValue` for `alpha.C4` and `alpha.Air` and `e`.

4. Completeness of 0/ Directory (Solver Crash Prevention):
   - You MUST ALWAYS generate these files in `0/`: 
     `p`, `U`, `T`, `e`, `alpha.C4`, `alpha.Air`, AND `rho.C4`, `rho.Air`.
   - `alpha.*` files must sum to 1. E.g., internalField for alpha.Air = 1, alpha.C4 = 0.

5. Scripts:
   - `Allrun` MUST be executable (`chmod +x Allrun`). Write `#!/bin/bash` directly. The structure of `Allrun` MUST reference the OpenFOAM RunFunctions like this:
     ```bash
     #!/bin/sh
     cd ${{0%/*}} || exit 1
     . $WM_PROJECT_DIR/bin/tools/RunFunctions
     
     runApplication surfaceFeatures
     runApplication blockMesh
     runApplication snappyHexMesh -overwrite
     runApplication setFields
     runApplication $(getApplication)
     ```
     DO NOT write environment checks like `if [ -f "$FOAM_BASH" ]` because the user environment is already sourced. Follow this simple `RunFunctions` structure. MUST use `runApplication` for all commands (serial execution) unless you explicitly generate system/decomposeParDict and run decomposePar.

   - You MUST also generate an `Allclean` script (`chmod +x Allclean`), referencing RunFunctions:
     ```bash
     #!/bin/sh
     cd ${{0%/*}} || exit 1
     . $WM_PROJECT_DIR/bin/tools/RunFunctions
     
     cleanCase
     rm -rf constant/polyMesh/ constant/geometry/*.eMesh log.*
     ```

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

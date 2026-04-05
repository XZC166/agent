import os
import sys
import io
import time
from agents.blastfoamagent import BlastFoamAgent

class StreamTee(object):
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
    def write(self, obj):
        self.stream1.write(obj)
        self.stream2.write(obj)
    def flush(self):
        pass

def run_eval_dim3_C():
    print(f"\n[维度三 - 专项体检 C] 物理边界与网格常识探针测试")
    sandbox = f"/tmp/agent_dim3_C_{int(time.time())}"
    os.makedirs(os.path.join(sandbox, "system"), exist_ok=True)
    os.makedirs(os.path.join(sandbox, "constant"), exist_ok=True)
    os.makedirs(os.path.join(sandbox, "0"), exist_ok=True)
    
    # 正常的 controlDict 
    controlDict_content = """FoamFile { version 2.0; format ascii; class dictionary; location "system"; object controlDict; }
application     blastFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.1;
deltaT          0.001;
writeControl    timeStep;
writeInterval   1;
"""
    with open(os.path.join(sandbox, "system", "controlDict"), "w") as f:
        f.write(controlDict_content)

    # blockMeshDict, patch named "walls"
    blockMeshDict_content = """FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale   1;
vertices ( (0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1) );
blocks ( hex (0 1 2 3 4 5 6 7) (10 10 10) simpleGrading (1 1 1) );
edges ();
boundary (
    walls { type patch; faces ( (0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7) ); }
);
"""
    with open(os.path.join(sandbox, "system", "blockMeshDict"), "w") as f:
        f.write(blockMeshDict_content)

    # 注入故障的 0/p (边界名"wall_wrong"与blockMesh的"walls"不匹配)
    p_content = """FoamFile { version 2.0; format ascii; class volScalarField; location "0"; object p; }
dimensions      [1 -1 -2 0 0 0 0];
internalField   uniform 100000;
boundaryField
{
    wall_wrong
    {
        type            zeroGradient;
    }
}
"""
    with open(os.path.join(sandbox, "0", "p"), "w") as f:
        f.write(p_content)
        
    print(f"[*] 注入故障靶场已就绪: {sandbox}")
    print(f"[*] 故障细节: 0/p 中定义了边界 `wall_wrong`，但网格里的名字是 `walls` ")
    print(f"[*] 考核指标: Agent能否运行 blastFoam (或者读日志) -> 发现无法找到边界 -> 编辑 0/p 修复边界名。")
    print("=" * 60)
    
    agent = BlastFoamAgent()
    query = f"There is an OpenFOAM case at `{sandbox}`. First, run `blockMesh`. Then run a basic diagnostic solver like `foamParse` or just read `0/p`. Ensure that the boundary names in `0/p` match the boundaries in `blockMeshDict` (patch 'walls'). Fix `0/p` if they don't match, and output 'FIXED boundary match'."
    
    f_log = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = StreamTee(sys.stdout, f_log)
    
    try:
        agent.run(user_request=query, context="Focus on aligning boundary patch names between system/blockMeshDict and 0/p.")
    except Exception as e:
        print(f"Exception: {e}")
        
    sys.stdout = original_stdout
    agent_log = f_log.getvalue()
    
    # 解析报告
    print("=" * 60)
    print("【维度三 方案C 评估结果】")
    if "0/p" in agent_log and ("wall_wrong" in agent_log or "walls" in agent_log):
        print("✅ [诊断逻辑探针] Agent 成功对比了 blockMeshDict 和 0/p 的边界名差异。")
    else:
        print("❌ [诊断逻辑探针] Agent 未能对比出边界名不匹配。")
        
    if "FIXED boundary match" in agent_log or "Execution SUCCESS" in agent_log:
        print("✅ [自愈闭环探针] 发现并修复边界不匹配 (Self-Healing Rate): 100%")
    else:
        print("❌ [自愈闭环探针] 修复动作未能成功进行 (Self-Healing Rate): 0%")

if __name__ == "__main__":
    run_eval_dim3_C()

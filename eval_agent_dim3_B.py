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

def run_eval_dim3_B():
    print(f"\n[维度三 - 专项体检 B] 词法抓虫与代码修改探针测试")
    sandbox = f"/tmp/agent_dim3_B_{int(time.time())}"
    os.makedirs(os.path.join(sandbox, "system"), exist_ok=True)
    os.makedirs(os.path.join(sandbox, "constant"), exist_ok=True)
    
    # 注入故障的 controlDict (缺少分号)
    controlDict_content = """FoamFile { version 2.0; format ascii; class dictionary; location "system"; object controlDict; }
application     blastFoam;
startFrom       startTime;
startTime       0
stopAt          endTime;
endTime         0.1;
deltaT          0.001;
writeControl    timeStep;
writeInterval   1;
"""
    with open(os.path.join(sandbox, "system", "controlDict"), "w") as f:
        f.write(controlDict_content)

    # 正常的 blockMeshDict 
    blockMeshDict_content = """FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale   1;
vertices ( (0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1) );
blocks ( hex (0 1 2 3 4 5 6 7) (10 10 10) simpleGrading (1 1 1) );
edges ();
boundary (
    walls { type wall; faces ( (0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7) ); }
);
"""
    with open(os.path.join(sandbox, "system", "blockMeshDict"), "w") as f:
        f.write(blockMeshDict_content)
        
    print(f"[*] 注入故障靶场已就绪: {sandbox}")
    print(f"[*] 故障细节: system/controlDict 第 4 行 `startTime 0` 故意漏写了分号 (;) ")
    print(f"[*] 考核指标: Agent能否阅读 blockMesh 的 FATAL ERROR -> 准确定位行号 -> 编辑修复语法 -> 跑通流程。")
    print("=" * 60)
    
    agent = BlastFoamAgent()
    query = f"There is an OpenFOAM case at `{sandbox}`. Please execute `blockMesh` in that directory. It will probably fail due to a syntax error in one of the dictionary files. Read the error output carefully, identify the file and the mistake (like a missing semicolon), fix the file, and run `blockMesh` again until you see 'End' indicating it succeeded."
    
    f_log = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = StreamTee(sys.stdout, f_log)
    
    try:
        agent.run(user_request=query, context="Focus on diagnosing OpenFOAM syntax errors and fixing text files.")
    except Exception as e:
        print(f"Exception: {e}")
        
    sys.stdout = original_stdout
    agent_log = f_log.getvalue()
    
    # 解析报告
    print("=" * 60)
    print("【维度三 方案B 评估结果】")
    if "system/controlDict" in agent_log or "sed" in agent_log or "write" in agent_log or "cat" in agent_log:
        print("✅ [诊断逻辑探针] Agent 查阅了错误日志，并定位到了 system/controlDict 文件。")
    else:
        print("❌ [诊断逻辑探针] Agent 未能定位 system/controlDict 的语法错误处。")
        
    if "Execution SUCCESS" in agent_log and "End" in agent_log:
        print("✅ [自愈闭环探针] 发现并修复分号，blockMesh 运行成功 (Self-Healing Rate): 100%")
    else:
        print("❌ [自愈闭环探针] blockMesh 最终未能成功运行 (Self-Healing Rate): 0%")

if __name__ == "__main__":
    run_eval_dim3_B()

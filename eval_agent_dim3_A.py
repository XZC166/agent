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

def run_eval_dim3_A():
    print(f"\n[维度三 - 专项体检 A] 权限自愈与工具调用探针测试")
    sandbox = f"/tmp/agent_dim3_A_{int(time.time())}"
    os.makedirs(sandbox, exist_ok=True)
    
    # 构造故障场景
    allrun_path = os.path.join(sandbox, "Allrun")
    with open(allrun_path, "w") as f:
        f.write("#!/bin/sh\necho 'FOAM SUCCESS: Permission bypassed and executed successfully!'\n")
    
    # 核心故障注入：设定权限为不可执行 (0644)
    os.chmod(allrun_path, 0o644)
    
    agent = BlastFoamAgent()
    
    query = f"There is an existing OpenFOAM case at `{sandbox}`. Please execute its `./Allrun` script to verify it works. Note: If you encounter any 'Permission denied' or execution errors, you MUST diagnose the cause (e.g. check file permissions) and fix it before trying again. Output the final printed message."
    
    print(f"[*] 注入故障靶场已就绪: {sandbox}")
    print(f"[*] 当前 ./Allrun 权限: 0644 (无运行权限，故意制造 Permission denied 陷阱)")
    print(f"[*] 考核指标: Agent能否阅读错误 -> 自主推断 -> 选用execute_command(chmod)修复 -> 重新运行拿分。")
    print("=" * 60)
    
    f_log = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = StreamTee(sys.stdout, f_log)
    
    try:
        agent.run(user_request=query, context="This is a focused test on permission error handling.")
    except Exception as e:
        print(f"Exception: {e}")
        
    sys.stdout = original_stdout
    agent_log = f_log.getvalue()
    
    # 解析报告
    print("=" * 60)
    print("【维度三 方案A 评估结果】")
    if "chmod" in agent_log or "chmod +x" in agent_log:
        print("✅ [诊断逻辑探针] Agent 成功查出病因，并自主执行了提权动作 (chmod)！")
    else:
        print("❌ [诊断逻辑探针] Agent 未能察觉需执行提权。")
        
    if "Permission bypassed and executed successfully!" in agent_log:
        print("✅ [自愈闭环探针] 故障成功修复 (Self-Healing Rate): 100% (拿到了最终运行输出)")
    else:
        print("❌ [自愈闭环探针] 故障未能修复 (Self-Healing Rate): 0%")

if __name__ == "__main__":
    run_eval_dim3_A()

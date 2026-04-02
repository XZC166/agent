import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

from rag_database.embedder import OpenFOAMEmbedder
from agents.blastfoamagent import BlastFoamAgent

import io

class StreamTee(object):
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
    def write(self, obj):
        self.stream1.write(obj)
        self.stream1.flush()
        self.stream2.write(obj)
    def flush(self):
        self.stream1.flush()
        self.stream2.flush()


def evaluate_e2e_agent():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 开始端到端全链路指定用例测试...")
    with open('eval_dataset.json', 'r', encoding='utf-8') as f: 
        dataset = json.load(f)
    
    # ---------------------------------------------------------
    # 【测试用例修改区】
    # 如果你需要测试其他案例，只需要修改下面 dataset 的切片范围即可：
    # dataset[0:1] -> 代表抽取测试集第 1 个问题 (Q001)
    # dataset[1:2] -> 代表抽取测试集第 2 个问题 (Q002)
    # dataset[5:6] -> 代表抽取测试集第 6 个问题 (Q006)
    # dataset[:5]  -> 代表测试集前 5 个问题顺序连测
    # ---------------------------------------------------------
    test_cases = dataset[1:2] 
    
    embedder = OpenFOAMEmbedder()
    agent = BlastFoamAgent()
    results = []
    Path("eval_results").mkdir(exist_ok=True)
    
    for item in test_cases:
        query = item['prompt']
        q_id = item['id']
        
        print(f"\n=========================================")
        print(f">>> 正在测试指定用例 [{q_id}]")
        print(f">>> Query: {query}")
        print(f"=========================================\n")
        
        retrieved_docs = embedder.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        sandbox_dir = f"/tmp/agent_eval_{q_id}_{int(time.time())}"
        os.makedirs(sandbox_dir, exist_ok=True)
        
        enhanced_query = f"{query}\n\nIMPORTANT: You must build and run this entire case inside the EXACT directory: `{sandbox_dir}`. Ensure all outputs go there. Run it to verify."
        
        start_time = time.time()
        print(f"[*] Agent 动态生成沙盒: {sandbox_dir}")
        print(f"[*] Agent 推理中, 请稍候 (耐心等待终端出最终结果)...")
        
        try:
            f_log = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = StreamTee(sys.stdout, f_log)
            response = agent.run(user_request=enhanced_query, context=context)
            sys.stdout = original_stdout
            agent_log = f_log.getvalue()
        except Exception as e:
            if 'original_stdout' in locals():
                sys.stdout = original_stdout
            agent_log = f"Exception: {e}"
            print(agent_log)
            
        latency = time.time() - start_time
        
        # 结果校验探针
        has_0 = os.path.isdir(os.path.join(sandbox_dir, "0")) or os.path.isdir(os.path.join(sandbox_dir, "0.org"))
        has_constant = os.path.isdir(os.path.join(sandbox_dir, "constant"))
        has_system = os.path.isdir(os.path.join(sandbox_dir, "system"))
        has_allrun = os.path.isfile(os.path.join(sandbox_dir, "Allrun"))
        has_allclean = os.path.isfile(os.path.join(sandbox_dir, "Allclean"))
        format_compliance = has_0 and has_constant and has_system and has_allrun and has_allclean
        
        final_runnable = False
        log_path = os.path.join(sandbox_dir, "log.blastFoam")
        if os.path.isfile(log_path):
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
                # 没有 FOAM FATAL ERROR 就是最终生成合格可以运行的算例
                if "FOAM FATAL ERROR" not in log_content.upper():
                    final_runnable = True
                    
        # 核心指标判定：终端无报错记录且成功执行
        first_pass_runnable = final_runnable and ("FAILED (Return Code" not in agent_log) and ("FOAM FATAL ERROR" not in agent_log)
                    
        print(f"\n[测试报告 {q_id}]")
        print(f"耗时: {latency:.2f}s")
        print(f"目录生成合规 (Format): {'✅' if format_compliance else '❌'}")
        print(f"一次写对命中 (First-Pass): {'✅' if first_pass_runnable else '❌'}")
        print(f"最终完整运行 (Final Runnable): {'✅' if final_runnable else '❌'}")
        
        results.append({
            "id": q_id,
            "latency_sec": round(latency, 2),
            "format_compliance": format_compliance,
            "final_runnable": final_runnable,
            "sandbox_dir": sandbox_dir
        })
        
    report = Path("eval_results") / f"agent_e2e_benchmark_{datetime.now().strftime('%m%d_%H%M')}.json"
    with open(report, 'w') as f: json.dump({"summary": {"first_pass_runnable_rate": f"{sum(1 for r in results if r['first_pass_runnable']) / len(results) * 100:.2f}%", "final_runnable_rate": f"{sum(1 for r in results if r['final_runnable']) / len(results) * 100:.2f}%"}, "details": results}, f, indent=4)
    print(f"\n✅ 评测结束！报告已保存至 => {report}")

if __name__ == "__main__":
    evaluate_e2e_agent()
    os._exit(0)

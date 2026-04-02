import os
import sys
import json
import time
import random
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


def evaluate_e2e_agent_random(num_samples=3):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 开始端到端全链路随机抽样（{num_samples}例）测试...")
    with open('eval_dataset.json', 'r', encoding='utf-8') as f: 
        dataset = json.load(f)
    
    # ---------------------------------------------------------
    # 随机抽取逻辑：
    # 使用 random.sample(dataset, num_samples) 从全部的 100 个案例中随机挑出若干个进行测试
    # 注意每个案例测试大约需要 5 分钟
    # ---------------------------------------------------------
    test_cases = random.sample(dataset, num_samples)
    
    embedder = OpenFOAMEmbedder()
    agent = BlastFoamAgent()
    results = []
    Path("eval_results").mkdir(exist_ok=True)
    
    for i, item in enumerate(test_cases, 1):
        query = item['prompt']
        q_id = item['id']
        
        print(f"\n=========================================")
        print(f">>> 当前进度 [{i}/{num_samples}]：正在测试随机抽取的用例 [{q_id}]")
        print(f">>> Query: {query}")
        print(f"=========================================\n")
        
        retrieved_docs = embedder.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        sandbox_dir = f"/tmp/agent_eval_{q_id}_{int(time.time())}"
        os.makedirs(sandbox_dir, exist_ok=True)
        
        # 将明确约束和要求传给Agent去构建
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
        
    report = Path("eval_results") / f"agent_e2e_benchmark_random_{datetime.now().strftime('%m%d_%H%M')}.json"
    
    success_count = sum([1 for r in results if r['final_runnable']])
    first_pass_count = sum([1 for r in results if r['first_pass_runnable']])
    summary = {
        "total_tested": num_samples,
        "first_pass_runnable_rate": f"{first_pass_count / num_samples * 100:.2f}%",
        "final_runnable_rate": f"{success_count / num_samples * 100:.2f}%",
        "avg_latency_sec": round(sum(r['latency_sec'] for r in results) / num_samples, 2)
    }
    
    with open(report, 'w') as f: json.dump({"summary": summary, "details": results}, f, indent=4)
    print(f"\n✅ 评测结束！报告已保存至 => {report}")

if __name__ == "__main__":
    # =========================================================
    # 【随机测试数量设置区】
    # 默认测试数量为 3 个案例。如果您想增加或减少随机抽取的测试数量，
    # 请修改下方括号里的 `num_samples` 的值即可：
    #
    # 例如：
    # evaluate_e2e_agent_random(num_samples=1)   # 只随机测 1 个（快速跑通）
    # evaluate_e2e_agent_random(num_samples=10)  # 随机测 10 个（大规模评测）
    #
    # 注意：每个案例完整的推理和解算最多可能需要 3~5 分钟。
    # =========================================================
    evaluate_e2e_agent_random(num_samples=3)
    os._exit(0)

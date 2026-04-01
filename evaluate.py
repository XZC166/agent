# -*- coding: utf-8 -*-
"""
自动化评测脚本 (Benchmark Tool)
用于执行《智能仿真Agent系统多维评测 PRD》中的指标评测。
"""

import os
import json
import time
import math
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rag_database.embedder import OpenFOAMEmbedder

class BenchmarkEvaluator:
    def __init__(self, dataset_path="eval_dataset.json"):
        print("正在初始化评测系统...")
        self.dataset_path = dataset_path
        self.results_dir = Path("eval_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.embedder = OpenFOAMEmbedder()
        self.embedder.build_vector_store()
        
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"找不到评测集: {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def evaluate_retriever(self, top_k=5):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 开始执行 维度一：检索器评测 (Retriever)...")
        print(f"评测总数: {len(self.dataset)} 条")
        
        results = []
        total_latency = 0
        hits_recall = 0
        reciprocal_ranks = []
        
        for idx, item in enumerate(self.dataset):
            query = item['prompt']
            q_id = item['id']
            category = item['category']
            ground_truth = item.get('ground_truth', [])
            
            start_time = time.time()
            retrieved_docs = self.embedder.similarity_search(query, k=top_k)
            latency = time.time() - start_time
            total_latency += latency
            
            sources = [doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]
            
            # Remove directory path if present in sources for matching
            sources_basenames = [os.path.basename(s) for s in sources]
            gt_basenames = [os.path.basename(gt) for gt in ground_truth]

            # Calculate Recall and MRR
            match_rank = -1
            for i, src in enumerate(sources_basenames):
                if src in gt_basenames:
                    match_rank = i + 1
                    break
            
            recall = 1 if match_rank != -1 else 0
            mrr = 1.0 / match_rank if match_rank != -1 else 0.0
            
            hits_recall += recall
            reciprocal_ranks.append(mrr)
            
            record = {
                "id": q_id,
                "category": category,
                "latency_sec": round(latency, 3),
                "query": query,
                "top_k_sources": sources_basenames,
                "ground_truth": gt_basenames,
                "recall": recall,
                "mrr": round(mrr, 3)
            }
            results.append(record)
            
            print(f"  [{q_id}] 耗时: {latency:.2f}s | Recall: {recall} | MRR: {mrr:.2f} | GT: {gt_basenames[0]} -> Top1: {sources_basenames[0] if sources_basenames else 'None'}")
            
        avg_latency = total_latency / len(self.dataset)
        avg_recall = hits_recall / len(self.dataset)
        avg_mrr = sum(reciprocal_ranks) / len(self.dataset)
        
        print(f"\n检索评测完成! 平均检索延迟: {avg_latency:.3f} 秒")
        print(f"Recall@{top_k}: {avg_recall:.2%}")
        print(f"MRR@{top_k}: {avg_mrr:.4f}")
        
        report_path = self.results_dir / f"retriever_benchmark_{datetime.now().strftime('%m%d_%H%M')}.json"
        
        summary = {
            "total_queries": len(self.dataset),
            "average_latency_sec": round(avg_latency, 3),
            "recall_at_k": round(avg_recall, 4),
            "mrr_at_k": round(avg_mrr, 4),
            "top_k_eval": top_k
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=4)
        print(f"报告已保存至: {report_path}")

    def evaluate_agent_e2e(self, test_limit=3):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 开始执行 维度二/三：智能体端到端全链路抽样测试...")
        test_cases = [d for d in self.dataset if d['category'] == "Vague_Semantic"][:test_limit]
        
        try:
            from agents.blastfoamagent import create_agent
            agent_executor = create_agent()
            print("成功载入智能体模型...")
        except ImportError as e:
            print(f"尚未打通自动化 Agent 接口或无正确环境: {e}")
            return

        for item in test_cases:
            query = item['prompt']
            print(f"\n>>> 正在自动化触发 Agent 测试用例 [{item['id']}]: {query}")

if __name__ == "__main__":
    evaluator = BenchmarkEvaluator("eval_dataset.json")
    evaluator.evaluate_retriever(top_k=3)
    evaluator.evaluate_agent_e2e(test_limit=1)


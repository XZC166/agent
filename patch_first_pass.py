import os
import io

class_tee = """
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
"""

for filename in ["eval_agent.py", "eval_agent_random.py"]:
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 注入Tee类
    if "class StreamTee" not in content:
        content = content.replace("from agents.blastfoamagent import BlastFoamAgent", "from agents.blastfoamagent import BlastFoamAgent\n" + class_tee)
    
    # 替换Agent运行抓取逻辑
    old_try = """        try:
            response = agent.run(user_request=enhanced_query, context=context)
            agent_log = "Run finished."
        except Exception as e:
            agent_log = f"Exception: {e}"
            print(agent_log)"""
            
    new_try = """        try:
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
            print(agent_log)"""
    if "StreamTee(sys.stdout, f_log)" not in content:
        content = content.replace(old_try, new_try)
        
    # 添加first_pass_runnable校验逻辑
    old_final_runnable = """                if "FOAM FATAL ERROR" not in log_content.upper():
                    final_runnable = True"""
                    
    new_final_runnable = """                if "FOAM FATAL ERROR" not in log_content.upper():
                    final_runnable = True
                    
        # 核心指标判定：终端无报错记录且成功执行
        first_pass_runnable = final_runnable and ("FAILED (Return Code" not in agent_log) and ("FOAM FATAL ERROR" not in agent_log)"""
    if "first_pass_runnable = final_runnable" not in content:
        content = content.replace(old_final_runnable, new_final_runnable)

    # 替换终端打印显示
    old_print = """print(f"最终完整运行 (Final Runnable): {'✅' if final_runnable else '❌'}")"""
    new_print = """print(f"一次写对命中 (First-Pass): {'✅' if first_pass_runnable else '❌'}")
        print(f"最终完整运行 (Final Runnable): {'✅' if final_runnable else '❌'}")"""
    if "一次写对命中" not in content:
        content = content.replace(old_print, new_print)
    
    # 替换写入字典
    content = content.replace('"first_pass_runnable": False, # Just mock it', '"first_pass_runnable": first_pass_runnable,')
    
    # 修改eval_agent_random.py的Summary
    if "random" in filename:
        old_summ = """    success_count = sum([1 for r in results if r['final_runnable']])
    summary = {
        "total_tested": num_samples,"""
        new_summ = """    success_count = sum([1 for r in results if r['final_runnable']])
    first_pass_count = sum([1 for r in results if r['first_pass_runnable']])
    summary = {
        "total_tested": num_samples,
        "first_pass_runnable_rate": f"{first_pass_count / num_samples * 100:.2f}%","""
        if "first_pass_count =" not in content:
            content = content.replace(old_summ, new_summ)
    else:
        # eval_agent.py summary
        content = content.replace('json.dump({"summary": {}, "details": results}', 
                                      'json.dump({"summary": {"first_pass_runnable_rate": f"{sum(1 for r in results if r[\'first_pass_runnable\']) / len(results) * 100:.2f}%", "final_runnable_rate": f"{sum(1 for r in results if r[\'final_runnable\']) / len(results) * 100:.2f}%"}, "details": results}')

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

#!/bin/bash
echo "🚀 正在后台启动随机抽样端到端评测 (eval_agent_random.py)..."
echo "📝 日志将实时写入到: eval_random.log"
nohup /home/oseasy/miniconda3/bin/conda run --no-capture-output -p /media/dev/vdb1/xuzicheng/agent/.conda python -u eval_agent_random.py > eval_random.log 2>&1 &
echo "✅ 启动成功！进程 PID: $!"
echo "💡 您可以随时运行 'tail -f eval_random.log' 来动态查看进度日志。"

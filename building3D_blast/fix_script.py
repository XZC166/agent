import sys

with open("/media/dev/vdb1/xuzicheng/agent/agents/blastfoamagent.py", "r") as f:
    text = f.read()

text = text.replace("     . $WM_PROJECT_DIR/bin/tools/CleanFunctions\n     \n     runApplication surfaceFeatures", "     . $WM_PROJECT_DIR/bin/tools/RunFunctions\n     \n     runApplication surfaceFeatures")

with open("/media/dev/vdb1/xuzicheng/agent/agents/blastfoamagent.py", "w") as f:
    f.write(text)

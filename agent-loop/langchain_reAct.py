import requests
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# ========== 1️⃣ 定义工具 ==========
@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气情况"""
    # 这里为了演示，暂时固定写死经纬度。
    lat, lon = 34.75, 113.62
    
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    res = requests.get(url).json()
    
    weather = res["current_weather"]
    return f"{city}天气状态 -> 温度: {weather['temperature']}°C, 风速: {weather['windspeed']}km/h"

# ========== 2️⃣ 初始化 LLM ==========
llm = ChatOpenAI(
    model="qwen3.5:cloud",  
    base_url="http://localhost:11434/v1/",   
    api_key="ollama",
)

# ========== 3️⃣ 创建 ReAct Agent ==========
# LangGraph中的 create_react_agent 底层自带了 agent(reasoning) 和 tools(acting) 两个节点的循环
tools = [get_weather]
agent = create_agent(llm, tools)

# ========== 4️⃣ 运行并直观展示 ReAct 循环 ==========
inputs = {"messages": [("user", "请帮我查一下郑州的天气怎么样？并且根据温度建议我穿什么。")]}

print("\n🚀 开始启动 ReAct 循环...\n")

# 使用 stream 模式按步骤获取状态流转（区分推理节点和动作节点）
for step in agent.stream(inputs, stream_mode="updates"):
    for node, state in step.items():
        print(f"================== 🟡 当前节点: {node.upper()} ==================")
        
        # 取出当前节点产生的最后一条消息
        msg = state["messages"][-1]
        
        # reasoning：大模型思考并决定接下来做什么
        if node == "agent" or node == "model":
            print("[Reasoning (思考环节)]")
            if msg.tool_calls:
                print(f"🧠 模型分析判定：需要更多信息，决定调用工具！")
                for tool_call in msg.tool_calls:
                    print(f"👉 准备使用工具: [{tool_call['name']}], 参数: {tool_call['args']}")
            else:
                print(f"🧠 模型分析判定：信息已充足，给出最终结论！")
                print(f"📝 最终回复内容:\n{msg.content}")
                
        # acting：执行工具调用并获取结果
        elif node == "tools":
            print("[Acting (行动环节)]")
            print(f"🛠️ 工具执行完毕，获取到如下背景知识反馈：")
            print(f"   => {msg.content}")
            print("🔄 将结果交还给 agent 节点进行下一轮推理...\n")
        
        print("\n")

print("✅ ReAct 任务循环圆满结束！")

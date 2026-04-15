import requests
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# ========== 1️⃣ 定义天气工具 ==========
@tool
def get_weather(city: str) -> str:
    # 郑州经纬度（你可以换成别的城市）
    lat, lon = 34.75, 113.62

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    res = requests.get(url).json()

    weather = res["current_weather"]
    return f"温度: {weather['temperature']}°C, 风速: {weather['windspeed']}km/h"


# ========== 2️⃣ 初始化 LLM ==========
llm = ChatOpenAI(
    model="qwen3.5:cloud",  
    base_url="http://localhost:11434/v1/",   
    api_key="ollama",
)

# ========== 3️⃣ 创建 ReAct Agent ==========
tools = [get_weather]

agent = create_agent(
    model=llm,
    tools=tools
)

# ========== 4️⃣ 运行 ==========
# 必须传入包含 "messages" 的字典
output = agent.invoke({"messages": [("user", "请告诉我郑州的天气怎么样？")]})

print("\n--- 🤖 模型的完整交互过程 (原始输出) ---")
for msg in output["messages"]:
    msg.pretty_print()

print("\n--- ✨ 最终结果 ---")
print(output["messages"][-1].content)
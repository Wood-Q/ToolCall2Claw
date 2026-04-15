import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

def get_weather(city: str) -> str:
    fake_db = {
        "Beijing": "晴天，25度",
        "Shanghai": "多云，28度",
        "Hangzhou": "小雨，22度",
    }
    return f"{city} 的天气是：{fake_db.get(city, '未知天气')}"

TOOLS = {
    "get_weather": get_weather,
}

def parse_tool_call(text: str):
    tool_match = re.search(r"<Tool>(.*?)</Tool>", text, re.DOTALL)
    args_match = re.search(r"<Args>(.*?)</Args>", text, re.DOTALL)

    if not tool_match:
        return None

    tool_name = tool_match.group(1).strip()
    args = {}

    if args_match:
        args = json.loads(args_match.group(1).strip())

    return {"tool": tool_name, "args": args}

system_prompt = """
你是一个助手。
你知道系统里有一个工具叫 get_weather，用于查询天气。

当用户询问天气时，不要直接回答。
你必须严格输出以下格式，不要输出任何额外内容：

<Tool>get_weather</Tool>
<Args>{"city":"城市名"}</Args>

如果不需要调用工具，就直接输出普通文本。
"""

user_prompt = "帮我查一下 Beijing 的天气"

llm = ChatOpenAI(
    model="qwen3.5:cloud",  
    base_url="http://localhost:11434/v1/",   
    api_key="ollama",
)


messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_prompt)
]

resp = llm.invoke(messages)

model_text = resp.content
print("模型原始输出：")
print(model_text)

call = parse_tool_call(model_text)

if call:
    tool_name = call["tool"]
    args = call["args"]

    if tool_name not in TOOLS:
        raise ValueError(f"未知工具: {tool_name}")

    result = TOOLS[tool_name](**args)
    print("\n工具执行结果：")
    print(result)
else:
    print("\n没有触发工具：")
    print(model_text)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool

# 1. 定义可用工具
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    fake_db = {"郑州": "晴天，25度", "北京": "多云，20度"}
    return f"{city}天气: {fake_db.get(city, '未知')}"

tools_map = {"get_weather": get_weather}

# 2. 初始化大模型，并显式绑定工具（类似于把工具描述告诉模型）
llm = ChatOpenAI(
    model="qwen3.5:cloud",  
    base_url="http://localhost:11434/v1/",   
    api_key="ollama",
).bind_tools([get_weather])

# 3. 核心：由我们自己维护的 message 数组上下文
messages = [HumanMessage(content="郑州天气怎么样？接下来适合去哪里玩？")]

print("🚀 启动原生自制 ReAct 循环...\n")

# 4. 手写 ReAct 循环逻辑
while True:
    # 【Reasoning 阶段】把带有上下文的 messages 丢给模型推理
    response = llm.invoke(messages)
    messages.append(response)  # 核心原理：必须把模型本次的答复追加到上下文中
    
    # 如果推断不包含工具调用，说明得出最终结论，结束循环！
    if not response.tool_calls:
        print("\n✅ [最终回答]:\n", response.content)
        break
        
    # 如果包含工具调用，进入 【Acting 阶段】
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"🧠 [模型思考]: 需要获取额外信息 -> 准备调用 {tool_name}, 参数 {tool_args}")
        
        # 实际执行工具函数
        tool_func = tools_map[tool_name]
        result = tool_func.invoke(tool_args)
        print(f"🛠️ [工具返回]: {result}")
        
        # 核心原理：查到的结果必须以 ToolMessage 形式追加进入 message 数组
        messages.append(ToolMessage(
            tool_call_id=tool_call["id"], 
            name=tool_name, 
            content=str(result)
        ))
        
    print("🔄 已将工具结果合并至上下文，投喂给模型开启下一轮...\n")


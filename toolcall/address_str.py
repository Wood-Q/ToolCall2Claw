def get_weather(city:str):
    return f"{city}的天气是晴天"

model_output="get_weather:Beijing"

toolname, arg = model_output.split(":")

tools={
    "get_weather": get_weather
}

if toolname in tools:
    result = tools[toolname](arg)
    print(result)
else:    
    print(f"工具{toolname}不存在")
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1/",   # 如果你走 Ollama OpenAI 兼容层
    api_key="ollama"
)

response = client.responses.create(
    model="qwen3.5:cloud",
    input="北京天气怎么样？"
)

print(response.model_dump_json(indent=2))

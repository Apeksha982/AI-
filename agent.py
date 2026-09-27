import os, json
from groq import Groq
from tools import pdf_search, web_search, calculator

client = Groq(api_key=os.environ["GROQ_API_KEY"])

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "pdf_search", "description": "Search the uploaded PDF for passages relevant to a question about its content.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web for current information not likely to be in the PDF.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate a math expression.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}
]

AVAILABLE_TOOLS = {"pdf_search": pdf_search, "web_search": web_search, "calculator": calculator}

def run_agent(user_question: str, max_steps: int = 4):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when they help answer the question. If no tool is needed, answer directly."},
        {"role": "user", "content": user_question}
    ]
    reasoning_steps = []
    for _ in range(max_steps):
        response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=messages, tools=TOOLS_SCHEMA, tool_choice="auto")
        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content, reasoning_steps
        messages.append(msg)
        for call in msg.tool_calls:
            fn_name = call.function.name
            args = json.loads(call.function.arguments)
            result = AVAILABLE_TOOLS[fn_name](**args)
            reasoning_steps.append(f"Called `{fn_name}` with {args}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "Reached step limit without a final answer.", reasoning_steps

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSeek Chatbot 示例
通过火山引擎API调用DeepSeek模型
"""

import requests
import json
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置信息（从环境变量读取）
API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
BOT_ID = os.getenv("DEEPSEEK_BOT_ID", "your-bot-id-here")
API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

def chat_with_deepseek(user_message):
    """
    向DeepSeek发送消息并获取回复
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": BOT_ID,  # 火山方舟中的bot ID
        "messages": [
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        assistant_message = result['choices'][0]['message']['content']
        return assistant_message
    
    except requests.exceptions.RequestException as e:
        return f"API调用失败: {str(e)}"
    except (KeyError, json.JSONDecodeError) as e:
        return f"响应解析失败: {str(e)}"

def main():
    """主函数：简单的命令行聊天界面"""
    print("=" * 50)
    print("DeepSeek Chatbot 示例")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 50)
    
    while True:
        user_input = input("\n👤 你: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 再见！")
            break
        
        if not user_input:
            continue
        
        print("🤖 DeepSeek: ", end="", flush=True)
        response = chat_with_deepseek(user_input)
        print(response)

if __name__ == "__main__":
    main()

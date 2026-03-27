"""
语音识别脚本 - 使用 OpenAI Whisper
识别任务二导出的音频文件
人工智能导论课程作业04 - 任务三
"""

import whisper
import time
import os

def main():
    print("=" * 60)
    print("语音识别程序 - OpenAI Whisper")
    print("人工智能导论课程作业04 - 任务三")
    print("=" * 60)
    
    # 音频文件路径（相对于当前目录）
    # 尝试多个可能的路径
    possible_paths = [
        "../task2_voice/cloned_audio.mp3",
        "cloned_audio.mp3",
        "../task2_voice/cloned_audio.wav",
        "cloned_audio.wav"
    ]
    
    audio_file = None
    for path in possible_paths:
        if os.path.exists(path):
            audio_file = path
            break
    
    if audio_file is None:
        print("\n❌ 错误：找不到音频文件！")
        print("请确保 cloned_audio.mp3 文件在以下位置之一：")
        for path in possible_paths:
            print(f"   - {path}")
        return
    
    print(f"\n[1] 音频文件: {audio_file}")
    
    # 加载模型
    print("[2] 正在加载模型... (首次运行会自动下载)")
    print("    模型大小: base (约145MB)")
    start_time = time.time()
    
    try:
        model = whisper.load_model("base")
        print(f"    模型加载完成，耗时: {time.time() - start_time:.2f}秒")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 识别
    print("[3] 正在识别音频...")
    print("    (音频时长约45秒，请耐心等待)")
    start_time = time.time()
    
    try:
        result = model.transcribe(audio_file, language="zh")
        print(f"    识别完成，耗时: {time.time() - start_time:.2f}秒")
    except Exception as e:
        print(f"❌ 识别失败: {e}")
        return
    
    # 输出结果
    print("\n" + "=" * 60)
    print("识别结果:")
    print("=" * 60)
    print(result["text"])
    
    # 保存结果
    output_file = "识别结果.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"\n✅ 识别结果已保存到: {output_file}")
    
    print("\n" + "=" * 60)
    print("程序运行完成！")
    print("=" * 60)
    
    return result["text"]

if __name__ == "__main__":
    main()

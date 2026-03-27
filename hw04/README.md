# 《人工智能导论》课程作业04

## 作业信息

| 项目 | 内容 |
|------|------|
| 课程名称 | 人工智能导论 |
| 作业编号 | HW04 |
| 作业名称 | 大模型文案、剪映声音克隆与开源语音识别实践 |
| 学生姓名 | [填写你的姓名] |
| 学生学号 | [填写你的学号] |
| 提交日期 | 2026年3月27日 |

---

## 作业概述

本作业完成了文本生成、语音合成与克隆、语音识别的完整闭环：

1. **任务一**：使用大模型生成科普文稿
2. **任务二**：使用剪映完成声音克隆并导出配音音频
3. **任务三**：调研开源语音识别方案并本地实现识别

---

## 目录结构

```
hw04/
├── README.md                       # 本文件：作业说明
├── task1_text/
│   └── generated_text.md           # 任务一：大模型生成的文稿
├── task2_voice/
│   ├── cloned_audio.mp3            # 任务二：克隆声音导出的音频
│   └── clone_description.md        # 任务二：克隆过程说明
└── task3_asr/
    ├── asr_report.md               # 任务三：ASR方案对比报告
    ├── recognize.py                # 任务三：识别代码
    ├── requirements.txt            # 任务三：依赖列表
    └── experiment_log.md           # 任务三：实验记录
```

---

## 任务一：大模型生成文稿

- **所用模型**：DeepSeek
- **生成日期**：2026年3月27日
- **标题**：《当AI学会了我的声音》
- **字数**：约280字

详见：[task1_text/generated_text.md](task1_text/generated_text.md)

---

## 任务二：剪映声音克隆

- **使用工具**：剪映专业版 6.0.0
- **克隆方式**：录制个人声音（约15秒）
- **导出格式**：MP3
- **音频时长**：约45秒

详见：[task2_voice/clone_description.md](task2_voice/clone_description.md)

---

## 任务三：开源语音识别

### 调研方案对比

| 方案 | 协议 | 语言 | 实时 | 部署难度 |
|------|------|------|------|----------|
| Whisper | MIT | 99种 | ❌ | 简单 |
| Vosk | Apache 2.0 | 20+ | ✅ | 中等 |
| FunASR | MIT | 中文 | ✅ | 中等 |

### 最终选择

- **方案**：OpenAI Whisper (base)
- **理由**：安装简单、中文准确率高、CPU可运行

### 运行方法

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行识别
python recognize.py
```

### 实验结果

- **字准确率**：100%
- **识别耗时**：约11.8秒（45秒音频）
- **错误类型**：仅标点符号差异

详见：[task3_asr/asr_report.md](task3_asr/asr_report.md)

---

## 运行说明

### 环境要求
- Python 3.8+
- 4GB+ 内存
- 约 500MB 硬盘空间（模型下载）

### 安装步骤

1. 克隆仓库到本地：
   ```bash
   git clone https://github.com/你的用户名/AI_Homework.git
   cd AI_Homework/hw04
   ```

2. 安装 Python 依赖：
   ```bash
   cd task3_asr
   pip install -r requirements.txt
   ```

3. 将音频文件放入 task2_voice 目录（已包含）

4. 运行识别程序：
   ```bash
   python recognize.py
   ```

---

## 总结

本次作业实现了从文本生成到语音合成再到语音识别的完整流程：

- ✅ 使用大模型生成自然语言文稿
- ✅ 使用剪映完成个人声音克隆
- ✅ 调研并实现开源语音识别方案

通过实践，深入理解了语音合成、声音克隆和语音识别的基本原理与应用场景。

---

**GitHub 仓库地址**：https://github.com/你的用户名/AI_Homework

**作业状态**：✅ 已完成

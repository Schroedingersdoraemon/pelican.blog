---
title: Deploy TranslateGemma Locally
date: 2026-05-06 22:05
tags: LLM
---

[TOC]

## 0. llama.cpp

edit `~/.config/systemd/user/llama-translate.service`

```systemd
[Unit]
Description=llama.cpp translation service
After=network.target

[Service]
Type=simple

# 工作目录（模型和脚本都在这里）
WorkingDirectory=%h/.llm.models/translategemma-4b-it

# 环境变量（避免写死路径）
Environment=MODEL=translategemma-4b-it
Environment=PORT=1234
Environment=CTX=1024

# 启动命令
ExecStart=/usr/bin/llama-server \
  --model ${MODEL}.q4_k_s.gguf \
  --jinja \
  --chat-template-file en_zh_translate.jinja \
  --host 0.0.0.0 \
  --port ${PORT} \
  -c ${CTX} \
  --sleep-idle-seconds 300


# 自动重启
# Restart=always
# RestartSec=2

# 日志直接进 journald
StandardOutput=journal
StandardError=journal

# 可选：限制资源（建议加）
# MemoryMax=8G
# CPUQuota=200%

[Install]
WantedBy=default.target


```

`~/.llm.models/translategemma-4b-it/en_zh_translate.jinja`

```jinja
{{ bos_token }}
<start_of_turn>user
You are a professional English (en) to Chinese (zh-CN) translator.
Your goal is to accurately convey the meaning and nuances of the original English text while adhering to Chinese grammar, vocabulary, and cultural sensitivities.
Produce only the Chinese translation, without any additional explanations or commentary.
Do not translate URLs, code, or proper nouns unless necessary.
Please translate the following English text into Chinese:


{{ messages[-1]["content"] | trim }}
<end_of_turn>
<start_of_turn>model

```

## (deprecated ver). lmstudio

提示词来自[这里](https://ollama.com/library/translategemma)

concise version

```
{{ bos_token }}
<start_of_turn>user
You are an English to Chinese translator.
Translate accurately and naturally.
Only output the translation. No explanations.

{{ messages[-1]["content"] | trim }}
<end_of_turn>
<start_of_turn>model
```

ChatGPT 分离了 Template 和 Prompt

Template

```
{{ bos_token }}
{% for message in messages %}
<start_of_turn>{{ message['role'] }}
{{ message['content'] }}
<end_of_turn>
{% endfor %}
<start_of_turn>model
```

Prompt

```
You are a professional English (en) to Chinese (zh-Hans) translator.
Your goal is to accurately convey the meaning and nuances of the original English text while adhering to Chinese grammar, vocabulary, and cultural sensitivities.
Produce only the Chinese translation, without any additional explanations or commentary.
Please translate the following English text into Chinese:
```

从[zimo](https://note.com/zimo/n/n02e7f10fc70d)学习到的

```
{{ bos_token }}
{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}
{% elif message['role'] == 'model' %}
{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{ '<start_of_turn>model\n' }}
{% endif %}
```

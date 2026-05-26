---
title: deploy llama translator locally
date: 2026-05-06 22:05
tags: LLM
---

[TOC]

## -1. Update

2026 年 5 月 18 日：换成了 [HY-MT1.5-7B](https://huggingface.co/tencent/HY-MT1.5-7B)

## -0.9. Update

2026 年 5 月 25 日：换成了 [Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B)

并发速度有所上升，另外文件 `en_zh_translate.jinja` 也略有改变

## 0. llama.cpp

gentoo 若要启用 `--cache-type-k q8_0` 特性，请增加 AMDGPU_TARGETS

```conf
# 6750gre 12g 是 gfx1031，但是某些特性依赖 gfx1030
AMDGPU_TARGETS="gfx1031 gfx1030"
```

## 1. llama-translator.service

### 1.1. translategemma

<details>

<summary>这是旧的 translategemma-4b 的有关内容</summary>

edit `~/.config/systemd/user/llama-translator.service`

```systemd
[Unit]
Description=llama.cpp translation service
After=network.target

[Service]
Type=simple

# 工作目录（模型和脚本都在这里）
WorkingDirectory=%h/.llm.models/translategemma-4b-it

# 环境变量（避免写死路径）
Environment="MODEL=translategemma-4b-it"
# 如果要用到 --cache-type-k 等特性，需要 uncomment 以下
# Environment="HSA_OVERRIDE_GFX_VERSION=10.3.0"

# 启动命令
ExecStart=/usr/bin/llama-server \
  --model ${MODEL}.q4_k_s.gguf \
  --host 0.0.0.0 \
  --port 1234 \
  --sleep-idle-seconds 300 \
  --kv-unified \
  --parallel 8 \
  --ctx-size 4096 \
  --temperature 0.1 \
  --jinja \
  --chat-template-file en_zh_translate.jinja \

# --metrics \
# --verbose

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

</details>

### 1.2. hy-mt1.5

<details>

<summary>这是旧的 hy-mt1.5-7b 的有关内容</summary>

edit `~/.config/systemd/user/llama-translator-hy.service`

```systemd
[Unit]
Description=llama.cpp translation service
After=network.target

[Service]
Type=simple

# 工作目录（模型和脚本都在这里）
WorkingDirectory=%h/.llm.models/HY-MT1.5-7B-GGUF

# 环境变量（避免写死路径）
Environment="MODEL=HY-MT1.5-7B-Q4_K_M"

# 启动命令
ExecStart=/usr/bin/llama-server \
  --model ${MODEL}.gguf \
  --host 0.0.0.0 \
  --port 1234 \
  --sleep-idle-seconds 300 \
  --kv-unified \
  --parallel 8 \
  --ctx-size 4096 \
  --no-cache-prompt \
  --temperature 0.1 \
  --jinja \
  --chat-template-file en_zh_translate.jinja \
  --verbose

# --metrics
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

</details>

### 1.3. hy-mt2

<details open>

<summary>这是 hy-mt2-1.8b 的有关内容</summary>

edit `~/.config/systemd/user/llama-translator-hy.service`

```systemd
[Unit]
Description=llama.cpp translation service
After=network.target

[Service]
Type=simple

# 工作目录（模型和脚本都在这里）
WorkingDirectory=%h/.llm.models/HY-MT2-1.8B-GGUF

# 环境变量（避免写死路径）
Environment="MODEL=HY-MT2-1.8B-Q8_0"

# 启动命令
ExecStart=/usr/bin/llama-server \
  --model ${MODEL}.gguf \
  --host 0.0.0.0 \
  --port 1234 \
  --sleep-idle-seconds 300 \
  --kv-unified \
  --flash-attn on \
  --parallel 12 \
  --ctx-size 32768 \
  --cache-reuse 1024 \
  --temperature 0 \
  --jinja \
  --mlock \
  --chat-template-file en_zh_translate.jinja \
  --metrics

# --metrics
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

</details>

## 2. jinja template

### 2.1. translategemma-4b

<details>

<summary>这是旧的 translategemma-4b 的有关内容</summary>

`~/.llm.models/translategemma-4b-it/en_zh_translate.jinja`

```jinja
{{ bos_token -}}
<start_of_turn>user
You are a professional English (en) to Chinese (zh-CN) translator.
Your goal is to accurately convey the meaning and nuances of the original English text while adhering to Chinese grammar, vocabulary, and cultural sensitivities.

# Translation Rules
1. Produce only the translated text. Do not include any tags, explanations, or quotes.
2. Do not translate URLs, code, units, proper nouns unless necessary.
3. Maintain the original paragraph structure and format.
4. If the text contains HTML tags, maintain their appropriate placement in the translation.

# Text to be translated:
{{ (messages | last).content | trim  }}
<end_of_turn>
<start_of_turn>model

```

</details>

### 2.2. hy-mt1.5-7b

<details>

<summary>这是旧的 hy-mt1.5-7b 的有关内容</summary>

`~/.llm.models/HY-MT1.5-7B-GGUF/en_zh_translate.jinja`

```jinja
{% set last_user =
    messages
    | selectattr("role", "equalto", "user")
    | last
%}

<|startoftext|>
Translate the following segment into zh, without additional explanation.

{{ last_user.content | trim }}<|extra_0|>
```

</details>

### 2.3. hy-mt2-1.8b

<details open>

<summary>这是 hy-mt2-1.8b 的有关内容</summary>

```jinja
{%- set user_text = messages[-1]['content'] -%}

<｜hy_begin▁of▁sentence｜>
Translate the following text into Chinese. Output only the translation, no explanation.
<｜hy_place▁holder▁no▁3｜>
{{ user_text | trim }}
<｜hy_Assistant｜>

```

</details>

## 3. for ImmersiveTrans

沉浸式翻译有三个 prompt

- system prompt

既然我们已经在 llama-server 的 jinja 指定了 system prompt，**此处留空**。

- multi-paragraph prompt

默认的是手动制定 %% 为分隔符

```jinja
Translate these paragraphs using %% as separator:
{{text}}
```

可直接传入原段落，问题不大

```jinja
{{text}}
```

- single-paragraph prompt

  对于 single-paragraph prompt 同样只保留 `{{text}}` 即可。

**值得注意的是！多/单段提示词必须至少保留 {{text}}，不然就会输出 Lorem**

```text
敏捷的狐狸跳过...
我能够吞下玻璃...
```

- 一些个超参数

大量短句只消耗 20-50 tokens，但是预处理 prompt eval 依然要花 30ms。
因而增加 **每次请求最大段落数 32** 并行处理。

网页翻译滚动时大量段落，因此限制 **每秒最大请求数 4** ，理论上每秒处理上限 128 段。

根据利特尔法则 $并发数 (Slots) = 到达率 (Requests/sec) × 处理时间 (Latency)$
此处每秒处理 4 个请求，平均处理时间根据日志约 5 秒，所需 slot 为 20.
快速翻页时， `--parallel 12` 处理每秒的 4 次请求依然会产生排队。
但是考虑到阅读会停顿，所以无妨。

`--ctx-size 32768` (32K 上下文)
1.8B 模型占用 2GB，6750GRE 12GB 还有 10G，llama-server 所有 Slot 共享 ctx-size。
32768 ÷ 8 slots = 4096 tokens/slot
如果有某个请求包含超长文本，4000 tokens 的 Slot 余裕足以应付，不会触发 context shift

`--cache-reuse 1024`
翻译任务高度重复，因为新请求里有很大一部分（Prompt 头部指令）和之前的请求一模一样。

## (deprecated ver). lmstudio

<details>

<summary>伟大探索时期的遗留</summary>

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

</details>

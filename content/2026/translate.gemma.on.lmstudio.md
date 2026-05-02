---
title: TranslateGemma on LMStudio
date: 2026-04-24 22:05
tags:
---

[TOC]

## 0. my web browser ver

translate gemma 在 LMStudio 的 Jinja Template

提示词来自[这里](https://ollama.com/library/translategemma)

```
{{ bos_token }}
<start_of_turn>user
You are a professional English (en) to Chinese (zh-Hans) translator.
Your goal is to accurately convey the meaning and nuances of the original English text while adhering to Chinese grammar, vocabulary, and cultural sensitivities.
Produce only the Chinese translation, without any additional explanations or commentary.
Do not translate URLs, code, or proper nouns unless necessary.
Please translate the following English text into Chinese:


{{ messages[-1]["content"] | trim }}
<end_of_turn>
<start_of_turn>model
```

## 1. ChatGPT Ver

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


## 2. zimo ver

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

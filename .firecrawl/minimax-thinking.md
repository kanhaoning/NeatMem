> ## Documentation Index
>
> Fetch the complete documentation index at: [/docs/llms.txt](https://platform.minimaxi.com/docs/llms.txt)
>
> Use this file to discover all available pages before exploring further.

[跳转到主要内容](https://platform.minimaxi.com/docs/api-reference/text-openai-api#content-area)

为了满足开发者对 OpenAI API 生态的使用需求，我们的 API 新增了对 OpenAI API 格式的支持。通过简单的配置，即可将 MiniMax 的能力接入到 OpenAI API 生态中。

## [​](https://platform.minimaxi.com/docs/api-reference/text-openai-api\#%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)  快速开始

### [​](https://platform.minimaxi.com/docs/api-reference/text-openai-api\#1-%E5%AE%89%E8%A3%85-openai-sdk)  1\. 安装 OpenAI SDK

Python

Node.js

```
pip install openai
```

### [​](https://platform.minimaxi.com/docs/api-reference/text-openai-api\#2-%E9%85%8D%E7%BD%AE%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F)  2\. 配置环境变量

```
export OPENAI_BASE_URL=https://api.minimaxi.com/v1
export OPENAI_API_KEY=${YOUR_API_KEY}
```

### [​](https://platform.minimaxi.com/docs/api-reference/text-openai-api\#3-%E8%B0%83%E7%94%A8-api)  3\. 调用 API

Python

```
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[\
        {"role": "system", "content": "You are a helpful assistant."},\
        {"role": "user", "content": "Hi, how are you?"},\
    ],
    # 设置 reasoning_split=True 将思考内容分离到 reasoning_details 字段
    extra_body={"reasoning_split": True},
)

print(f"Thinking:\n{response.choices[0].message.reasoning_details[0]['text']}\n")
print(f"Text:\n{response.choices[0].message.content}\n")
```

### [​](https://platform.minimaxi.com/docs/api-reference/text-openai-api\#4-%E7%89%B9%E5%88%AB%E6%B3%A8%E6%84%8F)  4\. 特别注意

在多轮 Function Call 对话中，必须将完整的模型返回（即 assistant 消息）添加到对话历史，以保持思维链的连续性：

- 将完整的 `response_message` 对象（包含 `tool_calls` 字段）添加到消息历史
  - 原生的OpenAI API 的 `MiniMax-M3``MiniMax-M2.7``MiniMax-M2.7-highspeed``MiniMax-M2.5``MiniMax-M2.5-highspeed``MiniMax-M2.1``MiniMax-M2.1-highspeed``MiniMax-M2` 模型 `content` 字段会包含 `<think>` 标签内容，需要完整保留
  - 在 Interleaved Thinking 友好格式中，通过启用额外的参数(`reasoning_split=True`)，模型思考内容通过 `reasoning_details` 字段单独提供，同样需要完整保留

## [​](https://platform.minimaxi.com/docs/api-reference/text-openai-api\#%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B)  支持的模型

使用 OpenAI SDK 时，支持以下 MiniMax 模型：

| 模型名称 | 上下文窗口 | 模型介绍 |
| --- | --- | --- |
| MiniMax-M3 | 1,000,000 | **最新 M 系列语言模型，适用于 Agent 推理、工具调用、代码和长上下文任务** |
| MiniMax-M2.7 | 204,800 | **开启模型的自我迭代**（输出速度约 60 TPS） |
| MiniMax-M2.7-highspeed | 204,800 | **M2.7 极速版：效果不变，更快，更敏捷**（输出速度约 100 TPS） |
| MiniMax-M2.5 | 204,800 | **顶尖性能与极致性价比，轻松驾驭复杂任务**（输出速度约 60 TPS） |
| MiniMax-M2.5-highspeed | 204,800 | **M2.5 极速版：效果不变，更快，更敏捷**（输出速度约 100 TPS） |
| MiniMax-M2.1 | 204,800 | **强大多语言编程能力，全面升级编程体验**（输出速度约 60 TPS） |
| MiniMax-M2.1-highspeed | 204,800 | **M2.1 极速版：效果不变，更快，更敏捷**（输出速度约 100 TPS） |
| MiniMax-M2 | 204,800 | **专为高效编码与 Agent 工作流而生** |

TPS（Tokens Per Second）的计算方式详见 [常见问题 \> 接口相关](https://platform.minimaxi.com/docs/faq/about-apis#%E9%97%AE%E6%96%87%E6%9C%AC%E6%A8%A1%E5%9E%8B%E7%9A%84-tpstokens-per-second%E6%98%AF%E5%A6%82%E4%BD%95%E8%AE%A1%E7%AE%97%E7%9A%84)。

更多模型信息请参考标准的 MiniMax API 接口文档。

## [​](https://platform.minimaxi.com/docs/api-reference/text-openai-api\#%E5%A4%9A%E6%A8%A1%E6%80%81%E8%BE%93%E5%85%A5)  多模态输入

OpenAI API 兼容的 Chat Completions 支持在 `MiniMax-M3` 中输入文本、图片和视频。图片使用 `image_url` 内容块，视频使用 `video_url` 内容块。`detail` 字段可取 `low`、`default`、`high`，默认值为 `default`；可通过 `max_long_side_pixel` 控制最长边。图片支持 JPEG、PNG、GIF、WEBP。视频支持 MP4、AVI、MOV、MKV；`fps` 默认值为 1，支持 0.2 到 5。URL 或 base64 视频最大 50 MB，图片最大 10 MB，请求体最大 64 MB。更大的视频请通过 Files API 上传后传入 `mm_file://{file_id}`，Files API 视频最大 512 MB。图片 token 用量会随图片尺寸和内容变化。以下是单张图片的粗略估算；准确用量以响应中的 `usage` 或可用的 token 计数接口为准：

| `detail` | 单张图片粗略 token 用量 |
| --- | --- |
| `low` | 通常为几百 token，最高约 600 |
| `default` | 通常约 1k-3k token，最高约 5k |
| `high` | 通常为数千 token，最高约 15k+ |

Python

```
response = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[\
        {\
            "role": "user",\
            "content": [\
                {"type": "text", "text": "Summarize what is happening here."},\
                {\
                    "type": "image_url",\
                    "image_url": {\
                        "url": "https://example.com/image.png",\
                        "detail": "default",\
                    },\
                },\
                {\
                    "type": "video_url",\
                    "video_url": {\
                        "url": "mm_file://file_id",\
                        "detail": "default",\
                    },\
                },\
            ],\
        }\
    ],
)
```

## [​](https://platform.minimaxi.com/docs/api-reference/text-openai-api\#minimax-m3-%E8%AF%B7%E6%B1%82%E5%8F%82%E6%95%B0)  MiniMax-M3 请求参数

`MiniMax-M3` 在 OpenAI API 兼容接口中支持以下额外的 Chat Completions 参数：

| 参数 | 说明 |
| --- | --- |
| `thinking` | 控制 MiniMax-M3 thinking。`type` 可取 `disabled` 或 `adaptive`；省略时默认开启 thinking。对于 M2.x 模型，thinking 无法关闭。 |
| `stream_options.include_usage` | 流式调用时，设为 `true` 可在流中返回 token 用量。 |
| `max_tokens` | 旧版生成长度限制参数。 |
| `max_completion_tokens` | 生成长度限制参数，新接入建议使用此字段。 |
| `temperature` | 采样温度。范围 `[0, 2]`，默认值 `1`。 |
| `top_p` | 核采样参数。范围 `[0, 1]`，`MiniMax-M3` 默认值 `0.95`，M2.x 系列默认值 `0.9`。 |
| `tools` | 函数工具定义。 |
| `reasoning_split` | 输出格式开关。启用后将 thinking 内容拆分到 `reasoning_content` 和 `reasoning_details`。 |

### [​](https://platform.minimaxi.com/docs/api-reference/text-openai-api\#thinking-%E6%8E%A7%E5%88%B6)  Thinking 控制

对于 `MiniMax-M3`，`thinking` 参数用于控制模型是否可以输出 thinking 内容。

- 如果省略 `thinking`，默认开启 thinking，响应会包含 thinking 内容。
- 设置 `thinking: {"type": "adaptive"}` 可显式保持 thinking 开启。对于 MiniMax-M3，`adaptive` 等同于开启 thinking。
- 设置 `thinking: {"type": "disabled"}` 可跳过 thinking 并直接回答。
- 对于 M2.x 模型，thinking 无法关闭；即使传入 `thinking: {"type": "disabled"}`，thinking 仍会保持开启。

`reasoning_split` 不会开启或关闭 thinking。它只控制 thinking 内容的返回方式：为 `true` 时，thinking 会通过 `reasoning_content` 和 `reasoning_details` 返回；为 `false` 时，原生 Chat Completions 响应会将 thinking 保留在 `content` 字段中的 `<think>...</think>` 标签内。

Python

```
response = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[{"role": "user", "content": "Hi, how are you?"}],
    extra_body={
        "thinking": {"type": "adaptive"},
    },
)
```

## [​](https://platform.minimaxi.com/docs/api-reference/text-openai-api\#%E7%A4%BA%E4%BE%8B%E4%BB%A3%E7%A0%81)  示例代码

### [​](https://platform.minimaxi.com/docs/api-reference/text-openai-api\#%E6%B5%81%E5%BC%8F%E5%93%8D%E5%BA%94)  流式响应

Python

```
from openai import OpenAI

client = OpenAI()

print("Starting stream response...\n")
print("=" * 60)
print("Thinking Process:")
print("=" * 60)

stream = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[\
        {"role": "system", "content": "You are a helpful assistant."},\
        {"role": "user", "content": "Hi, how are you?"},\
    ],
    # 设置 reasoning_split=True 将思考内容分离到 reasoning_details 字段
    extra_body={"reasoning_split": True},
    stream=True,
)

reasoning_buffer = ""
text_buffer = ""

for chunk in stream:
    if (
        hasattr(chunk.choices[0].delta, "reasoning_details")
        and chunk.choices[0].delta.reasoning_details
    ):
        for detail in chunk.choices[0].delta.reasoning_details:
            if "text" in detail:
                reasoning_text = detail["text"]
                new_reasoning = reasoning_text[len(reasoning_buffer) :]
                if new_reasoning:
                    print(new_reasoning, end="", flush=True)
                    reasoning_buffer = reasoning_text

    if chunk.choices[0].delta.content:
        content_text = chunk.choices[0].delta.content
        new_text = content_text[len(text_buffer) :] if text_buffer else content_text
        if new_text:
            print(new_text, end="", flush=True)
            text_buffer = content_text

print("\n" + "=" * 60)
print("Response Content:")
print("=" * 60)
print(f"{text_buffer}\n")
```

### [​](https://platform.minimaxi.com/docs/api-reference/text-openai-api\#tool-use-&-interleaved-thinking)  Tool Use & Interleaved Thinking

了解如何通过 OpenAI SDK 使用 M3 Tool Use 和 Interleaved Thinking 能力，请参考以下文档。

## Tool Use & Interleaved Thinking

了解如何利用 MiniMax-M3 工具调用和 Interleaved Thinking 能力，提升复杂任务中的表现。

点击查看

## [​](https://platform.minimaxi.com/docs/api-reference/text-openai-api\#%E6%B3%A8%E6%84%8F%E4%BA%8B%E9%A1%B9)  注意事项

如果在使用MiniMax模型过程中遇到任何问题：

- 通过邮箱 [Model@minimaxi.com](mailto:Model@minimaxi.com) 等官方渠道联系我们的技术支持团队
- 在我们的 [Github](https://github.com/MiniMax-AI/MiniMax-M2/issues) 仓库提交Issue

1. `temperature` 参数取值范围为 \[0, 2\]，推荐使用 1.0，超出范围会返回错误
2. 部分 OpenAI 参数（如`presence_penalty`、`frequency_penalty`、`logit_bias` 等）会被忽略
3. `MiniMax-M3` 可通过 OpenAI 兼容消息内容块输入图片和视频；当前不支持音频输入
4. `n` 参数仅支持值为 1
5. 旧版的`function_call` 已废弃，请使用 `tools` 参数

此页面对您有帮助吗？

是否

[Anthropic SDK（推荐）](https://platform.minimaxi.com/docs/api-reference/text-anthropic-api) [AI SDK](https://platform.minimaxi.com/docs/api-reference/text-ai-sdk)

⌘I
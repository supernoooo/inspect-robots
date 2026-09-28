| `wire` | 通信方式 | 你可以怎么用 |
|---|---|---|
| `chat` | HTTP `/chat/completions` | 通用 OpenAI-compatible 接口、OpenRouter、本地模型服务 |
| `responses` | HTTP `/responses` | 你当前 GPT、Grok 的评测配置 |
| `messages` | HTTP `/messages` | Claude 的 API Key 模式；`anthropic` 是它的别名 |
| `claude-code` | 调用官方 `claude -p` CLI | 你当前 Claude 账号登录模式 |
| `interactions` | Google HTTP `/interactions` | 你当前 Gemini 评测；服务端维护会话历史 |
| `gemini-live` | Google WebSocket 长连接 | 支持 Live API 的模型，例如 streaming robotics 模型 |
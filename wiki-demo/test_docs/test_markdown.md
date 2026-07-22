# Chunker 测试文档

这是一篇测试文档，用于验证 Markdown 切分器的行为。

## 第一个章节

这是第一个章节的内容。它包含多个段落。

这是第一个章节的第二段。我们希望切分器能正确处理章节边界。

## 第二个章节：代码示例

下面是一段 Python 代码：

```python
def hello_world():
    """一个简单的问候函数。"""
    print("Hello, World!")
    return True


class Greeter:
    """问候器类。"""

    def __init__(self, name: str):
        self.name = name

    def greet(self) -> str:
        return f"Hello, {self.name}!"

    def farewell(self, reason: str = "") -> str:
        if reason:
            return f"Goodbye, {self.name}! Reason: {reason}"
        return f"Goodbye, {self.name}!"
```

## 第三个章节：表格测试

| 特性 | 支持 | 备注 |
|------|------|------|
| Markdown 切分 | ✅ | 按 ## 标题切 |
| 代码切分 | ✅ | 按函数/类切 |
| 纯文本切分 | ✅ | 按段落切 |
| 表格保持 | ✅ | 不切断表格 |
| 代码块保持 | ✅ | 不切断代码块 |

## 第四个章节：长文本测试

这是一个很长的章节，用来测试 chunker 的二次切分能力。当单个章节的内容超过 max_chunk_chars（默认 2000 字符）时，chunker 应该能按段落边界智能地将其切分为多个 chunk，同时保留标题信息作为 heading 元数据。

这种二次切分保证了每个 chunk 的大小可控，便于向量嵌入和检索。每个子 chunk 都会标注它属于哪个父章节，以便在检索时能追溯到完整的文档结构。

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

这段文字是占位用的，用来增加章节的长度，以触发二次切分逻辑。在实际应用中，技术文档的单个章节通常不会这么啰嗦，但为了测试我们故意让它长一点。

现在让我们再增加一些内容，确保这个章节确实超过了 2000 字符的阈值。我们已经有不少文字了，但还是不够。继续添加一些有意义的内容。

微服务架构是一种将应用拆分为多个小型、独立服务的架构风格。每个服务运行在自己的进程中，通过轻量级通信机制（通常是 HTTP/REST 或消息队列）互相协作。这种架构风格带来了许多好处：独立部署、技术栈自由、故障隔离、按需扩缩容等。

容器化技术（如 Docker）和编排平台（如 Kubernetes）极大地推动了微服务架构的普及。它们解决了微服务部署和管理的复杂性。

继续添加更多内容。Kubernetes 是一个开源的容器编排平台，最初由 Google 设计，现在由 CNCF 维护。它提供了服务发现、负载均衡、自动扩缩容、滚动更新等核心能力。

现在应该差不多够了，我们再加最后一段。Service Mesh（如 Istio、Linkerd）是微服务架构中的基础设施层，负责处理服务间的通信、监控和安全。它将这些横切关注点从应用代码中抽离出来，通过 Sidecar 代理模式实现。

好了，这个章节现在应该足够长了，可以触发二次切分。让我们看看 chunker 的表现如何。

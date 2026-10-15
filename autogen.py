"""AutoGen 多智能体软件开发团队。

将团队从「固定轮询」升级为「可动态回退 + 带 QA 测试 + 带质量监控 + 结果落地」的协作流水线。

流程：
    ProductManager ──需求──▶ Engineer ──实现──▶ CodeReviewer ──通过──▶ QAEngineer
          ▲                    ▲   ▲                 │                    │
          │(需求变更/不清)     │(打回修改)           │(不通过)            │(测试失败)
          └────────────────────┴───┴─────────────────┴────────────────────┘
                                        全部通过 ──▶ TERMINATE(结束)
"""

import os
import re
import asyncio
import hashlib
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import (
    HandoffTermination,
    MaxMessageTermination,
    TimeoutTermination,
)
from autogen_agentchat.messages import HandoffMessage, StopMessage
from autogen_agentchat.base import TerminationCondition
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool

load_dotenv()

# 主题关键词，用于「偏离主题」的廉价启发式检测
TASK_KEYWORDS = ["比特币", "价格", "Streamlit", "BTC", "bitcoin", "price", "涨幅"]


# ---------------------------------------------------------------------------
# 可选：LangSmith 观测（未配置 LANGCHAIN_API_KEY 时自动跳过）
# ---------------------------------------------------------------------------
def setup_langsmith_tracing():
    """把 AutoGen 的 OpenTelemetry 轨迹导出到 LangSmith。"""
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print("ℹ️ 未配置 LANGCHAIN_API_KEY，跳过 LangSmith 观测")
        return

    # 依赖放在函数内部导入，避免未安装 opentelemetry 时影响整体运行
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    exporter = OTLPSpanExporter(
        endpoint="https://api.smith.langchain.com/otel",  # 注意是 /otel 不是 /otlp
        headers={
            "x-api-key": api_key,
            "Langsmith-Project": os.getenv("LANGSMITH_PROJECT", "autogen-btc-app"),
        },
        timeout=10,
    )
    tracer_provider = TracerProvider(
        resource=Resource({"service.name": "autogen-btc-app"})
    )
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)

    try:
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument()
    except Exception as e:  # 插桩失败不影响主流程
        print(f"⚠️ OpenAI 插桩失败（不影响运行）：{e}")


# ---------------------------------------------------------------------------
# 模型客户端
# ---------------------------------------------------------------------------
def create_openai_model_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 未设置，请在 .env 中配置")

    return OpenAIChatCompletionClient(
        model=os.getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-pro"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model_info={
            "function_calling": True,
            "max_tokens": 4096,
            "context_length": 32768,
            "vision": False,
            "json_output": True,
            "family": "deepseek",
            "structured_output": True,
        },
    )


# ---------------------------------------------------------------------------
# 落地工具：让智能体真正写文件 / 跑测试
# ---------------------------------------------------------------------------
def write_file(filename: str, content: str) -> str:
    """把完整内容写入当前目录下的文件。

    Args:
        filename: 文件名（相对路径），如 bitcoin_app.py 或 tests/test_bitcoin_app.py
        content: 完整文件内容
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"已写入 {path.resolve()}"


def run_tests() -> str:
    """在项目目录运行 pytest，返回完整测试输出。"""
    try:
        result = subprocess.run(
            ["pytest", "tests", "-q"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return output or f"（pytest 无输出，返回码 {result.returncode}）"
    except FileNotFoundError:
        return "pytest 未安装，请先执行 pip install pytest"
    except subprocess.TimeoutExpired:
        return "测试超时（120s）"


write_file_tool = FunctionTool(
    write_file,
    name="write_file",
    description="将代码或文档写入本地文件。filename 是相对文件名，content 是完整内容。",
)

run_tests_tool = FunctionTool(
    run_tests,
    name="run_tests",
    description="在项目目录运行 pytest 测试，返回完整测试输出。",
)


# ---------------------------------------------------------------------------
# Handoff 工具：声明「某个角色可以交接给谁」
# ---------------------------------------------------------------------------
def handoff(source: str, target: str):
    """生成一个交接函数，source 交棒给 target。"""

    async def _handoff() -> HandoffMessage:
        return HandoffMessage(source=source, target=target)

    return _handoff


# ---------------------------------------------------------------------------
# 角色定义
# ---------------------------------------------------------------------------
def create_product_manager(model_client):
    """产品经理：需求分析 + 最终验收。"""
    system_message = """你是一位经验丰富的产品经理，专门负责软件产品的需求分析和项目规划。

你的核心职责包括：
1. **需求分析**：深入理解用户需求，识别核心功能和边界条件
2. **技术规划**：基于需求制定清晰的技术实现路径
3. **风险评估**：识别潜在的技术风险和用户体验问题
4. **协调沟通**：与工程师和其他团队成员进行有效沟通

当接到开发任务时，请按以下结构进行分析：
1. 需求理解与分析
2. 功能模块划分
3. 技术选型建议
4. 实现优先级排序
5. 验收标准定义

交接规则：
- 分析完成后，将需求交接给 Engineer
- 收到「需求变更」或「需求不清」的反馈时，重新进行需求分析"""
    return AssistantAgent(
        name="ProductManager",
        model_client=model_client,
        system_message=system_message,
        handoffs=[handoff("ProductManager", "Engineer")],
    )


def create_engineer(model_client):
    """软件工程师：实现代码，并用工具落盘。"""
    system_message = """你是一位资深的软件工程师，擅长 Python 开发和 Web 应用构建。

你的技术专长包括：
1. **Python 编程**：熟练掌握 Python 语法和最佳实践
2. **Web 开发**：精通 Streamlit、Flask、Django 等框架
3. **API 集成**：有丰富的第三方 API 集成经验
4. **错误处理**：注重代码的健壮性和异常处理

工作流程：
1. 仔细分析技术需求
2. 选择合适的技术方案
3. 编写完整的代码实现
4. 添加必要的注释和说明
5. 考虑边界情况和异常处理

关键要求：
- 最终代码必须用 write_file 工具保存为 bitcoin_app.py
- 完成实现后交接给 CodeReviewer 审查
- 若需求不清或自相矛盾，交接回 ProductManager 澄清
- 收到审查不通过或测试失败的反馈时，修改代码并重新保存"""
    return AssistantAgent(
        name="Engineer",
        model_client=model_client,
        system_message=system_message,
        tools=[write_file_tool],
        handoffs=[
            handoff("Engineer", "CodeReviewer"),
            handoff("Engineer", "ProductManager"),
        ],
    )


def create_code_reviewer(model_client):
    """代码审查员：审查质量，不通过则打回。"""
    system_message = """你是一位经验丰富的代码审查专家，专注于代码质量和最佳实践。

你的审查重点包括：
1. **代码质量**：检查代码的可读性、可维护性和性能
2. **安全性**：识别潜在的安全漏洞和风险点
3. **最佳实践**：确保代码遵循行业标准和最佳实践
4. **错误处理**：验证异常处理的完整性和合理性

审查流程：
1. 仔细阅读和理解代码逻辑
2. 检查代码规范和最佳实践
3. 识别潜在问题和改进点
4. 提供具体的修改建议
5. 评估代码的整体质量

交接规则：
- 审查不通过 → 交回 Engineer 修改
- 发现实现与需求不符 → 交回 ProductManager
- 审查通过 → 交给 QAEngineer 测试"""
    return AssistantAgent(
        name="CodeReviewer",
        model_client=model_client,
        system_message=system_message,
        handoffs=[
            handoff("CodeReviewer", "QAEngineer"),
            handoff("CodeReviewer", "Engineer"),
            handoff("CodeReviewer", "ProductManager"),
        ],
    )


def create_qa_engineer(model_client):
    """测试工程师（QA）：在审查通过后执行自动化测试。"""
    system_message = """你是一位测试工程师（QA），在代码审查通过后负责执行自动化测试，确保代码满足需求且健壮。

你的工作流程：
1. 通读需求、工程师的代码、以及审查员的意见
2. 设计测试用例，覆盖：
   - 正常流程（价格获取、涨跌幅计算）
   - 边界条件（API 返回空、价格为 0、网络超时）
   - 错误处理（异常、非法输入、加载中状态）
3. 用 write_file 工具编写 pytest 测试文件 tests/test_bitcoin_app.py
4. 用 run_tests 工具实际执行测试
5. 根据真实运行输出给出结论，绝不臆测

交接规则：
- 有测试失败：列出失败用例 + 具体报错 + 定位原因，交回 Engineer 修复
- 需求需要重新确认：交回 ProductManager
- 全部测试通过：总结覆盖范围，交接给 TERMINATE 结束流程

注意：Streamlit UI 难以直接单元测试，优先对「数据获取、价格计算、涨跌幅/涨跌额计算」等纯逻辑函数做测试，并对第三方 API 调用使用 mock。"""
    return AssistantAgent(
        name="QAEngineer",
        model_client=model_client,
        system_message=system_message,
        tools=[write_file_tool, run_tests_tool],
        handoffs=[
            handoff("QAEngineer", "Engineer"),
            handoff("QAEngineer", "ProductManager"),
            handoff("QAEngineer", "TERMINATE"),
        ],
    )


# ---------------------------------------------------------------------------
# 对话质量监控：检测重复（死循环）和偏离主题
# ---------------------------------------------------------------------------
class ConversationHealthCheck(TerminationCondition):
    """便宜的确定性健康检查，命中则终止（由外层决定是否纠偏重跑）。

    注意：关键词法较粗糙，仅用于拦截明显跑偏；如需更准，可在此处接入 LLM 判分。
    """

    def __init__(self, task_keywords, repeat_window=3, off_track_streak=3, min_len=30):
        self._keywords = task_keywords
        self._repeat_window = repeat_window
        self._off_track_needed = off_track_streak
        self._min_len = min_len
        self._off_track_streak = 0
        self._terminated = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(self, messages):
        if not messages:
            return None

        # (a) 循环检测：最近 N 条消息内容几乎一致
        recent = [getattr(m, "content", "") or "" for m in messages[-self._repeat_window:]]
        if len(recent) >= self._repeat_window and len(
            {hashlib.md5(c.encode("utf-8")).hexdigest() for c in recent}
        ) == 1:
            self._terminated = True
            return StopMessage(content="检测到连续重复输出，疑似死循环")

        # (b) 偏离检测：足够长的消息却完全不命中主题关键词
        last = getattr(messages[-1], "content", "") or ""
        if len(last) >= self._min_len and not any(k in last for k in self._keywords):
            self._off_track_streak += 1
            if self._off_track_streak >= self._off_track_needed:
                self._terminated = True
                return StopMessage(content="连续多条消息偏离主题，疑似跑偏")
        else:
            self._off_track_streak = 0
        return None


# 全局调度器的路由规则
SELECTOR_PROMPT = """你是团队流程调度器，根据最新消息决定下一位发言者，保证协作围绕「比特币价格显示应用」推进。

路由规则（按优先级）：
- 消息表明需求变更、需求不清或需要产品确认 → ProductManager
- 需要编写/修改/修复代码 → Engineer
- 代码待审查 → CodeReviewer
- 审查已通过、需要测试 → QAEngineer
- 测试失败 → Engineer
- 全部通过或 QA 已交接给 TERMINATE → 结束
- 话题偏离主题时 → 让 ProductManager 重新聚焦需求

只返回下一位发言者的名字。"""


# ---------------------------------------------------------------------------
# 结果落地
# ---------------------------------------------------------------------------
def extract_code(text: str) -> str:
    """从文本里提取代码块，优先 ``` 围栏，否则原样返回。"""
    m = re.search(r"```[a-zA-Z]*\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def save_results(result):
    """把团队产出落地成文件（bitcoin_app.py + transcript.md）。"""
    out_dir = Path(__file__).parent
    messages = result.messages

    # 兜底：若 Engineer 没成功用工具写文件，从对话里抽取代码补写
    app_path = out_dir / "bitcoin_app.py"
    if app_path.exists():
        print("📄 bitcoin_app.py 已由 Engineer 通过工具写入")
    else:
        code = None
        for msg in reversed(messages):
            content = getattr(msg, "content", "") or ""
            if getattr(msg, "source", None) == "Engineer" and content:
                code = extract_code(content)
                if code:
                    break
        if code:
            app_path.write_text(code, encoding="utf-8")
            print("📄 已生成 bitcoin_app.py（兜底抽取）")
        else:
            print("⚠️ 未找到工程师代码，未生成 bitcoin_app.py")

    # 完整对话记录
    lines = []
    for msg in messages:
        source = getattr(msg, "source", "?") or "?"
        content = getattr(msg, "content", "") or ""
        lines.append(f"## {source}\n\n{content}\n")
    (out_dir / "transcript.md").write_text("\n".join(lines), encoding="utf-8")
    print("📄 已生成 transcript.md")


# ---------------------------------------------------------------------------
# 组装团队（每次运行都新建，避免终止条件内部状态残留）
# ---------------------------------------------------------------------------
def build_team(model_client):
    health_check = ConversationHealthCheck(task_keywords=TASK_KEYWORDS)
    termination = (
        HandoffTermination(target="TERMINATE")
        | MaxMessageTermination(40)
        | TimeoutTermination(600)
        | health_check
    )

    return SelectorGroupChat(
        participants=[
            create_product_manager(model_client),
            create_engineer(model_client),
            create_code_reviewer(model_client),
            create_qa_engineer(model_client),
        ],
        model_client=model_client,  # selector 用同一模型做调度
        selector_prompt=SELECTOR_PROMPT,
        termination_condition=termination,
        allow_repeated_speaker=True,
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def run_software_development_team():
    setup_langsmith_tracing()

    print("🔧 正在初始化模型客户端...")
    model_client = create_openai_model_client()

    base_task = """我们需要开发一个比特币价格显示应用，具体要求如下：

核心功能：
- 实时显示比特币当前价格（USD）
- 显示24小时价格变化趋势（涨跌幅和涨跌额）
- 提供价格刷新功能

技术要求：
- 使用 Streamlit 框架创建 Web 应用
- 界面简洁美观，用户友好
- 添加适当的错误处理和加载状态

请团队协作完成这个任务，从需求分析、代码实现、代码审查、自动化测试到最终交付。"""

    result = None
    for attempt in range(2):  # 最多 1 次纠偏重跑
        if attempt == 0:
            task = base_task
        else:
            task = (
                "【纠偏】上一次协作被判定异常，请重新聚焦任务"
                "（比特币价格显示应用），不要重复旧内容或跑题：\n" + base_task
            )
            print(f"⚠️ 检测到异常，正在第 {attempt + 1} 次纠偏重跑...")

        team = build_team(model_client)
        print("🚀 启动 AutoGen 软件开发团队协作...")
        print("=" * 60)

        result = await Console(team.run_stream(task=task))

        reason = result.stop_reason or ""
        print("\n" + "=" * 60)
        print(f"🛑 结束原因：{reason}")

        if "偏离" in reason or "死循环" in reason or "重复" in reason:
            continue  # 触发纠偏重跑
        break

    print("✅ 团队协作完成！")
    save_results(result)
    await model_client.close()
    return result


if __name__ == "__main__":
    try:
        result = asyncio.run(run_software_development_team())
        print("\n📋 协作结果摘要：")
        print("- 参与智能体：ProductManager / Engineer / CodeReviewer / QAEngineer")
        print(f"- 结束原因：{result.stop_reason}")
    except ValueError as e:
        print(f"❌ 配置错误：{e}")
        print("请检查 .env 文件中的配置是否正确")
    except Exception as e:
        print(f"❌ 运行错误：{e}")
        import traceback

        traceback.print_exc()

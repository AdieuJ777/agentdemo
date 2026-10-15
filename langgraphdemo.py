"""
智能搜索助手 - 基于 LangGraph + Tavily API 的真实搜索系统
1. 理解用户需求
2. 使用Tavily API真实搜索信息
3. 生成基于搜索结果的回答
"""
import  os
import asyncio

from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_deepseek import ChatDeepSeek


DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL=os.getenv("DEEPSEEK_BASE_URL")

llm_deepseek=ChatDeepSeek(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
)

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient

class SearchState(TypedDict):
    messages:Annotated[list,add_messages]
    user_query:str
    search_query:str
    search_results:str
    final_answer:str
    step:str

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def understand_query_search(state:SearchState)->SearchState:
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg,HumanMessage):
            user_message = msg.content
            break

    understand_prompt = f"""分析用户的查询："{user_message}"

    请完成两个任务：
    1. 简洁总结用户想要了解什么
    2. 生成最适合搜索的关键词（中英文均可，要精准）

    格式：
    理解：[用户需求总结]
    搜索词：[最佳搜索关键词]"""
    response = llm_deepseek.invoke([SystemMessage(content=understand_prompt)])
    response_text = response.content
    search_query = user_message

    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：")[1].strip()
    elif"搜索关键词" in response_text:
        search_query = response_text.split("搜索关键词：")[1].strip()

    return {
        "user_query":response.content,
        "search_query":search_query,
        "step":"understood",
        "messages":[AIMessage(content=f"我理解了您的需求：{response.content}")]
    }

def tavily_search_node(state:SearchState)->SearchState:
    search_query = state["search_query"]

    try:
        print(f"🔍 正在搜索: {search_query}")

        response = tavily_client.search(
            query=search_query,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
            max_results = 5
        )

        search_results = ""
        if response.get("answer"):
            search_results = f"综合答案：\n{response['answer']}\n\n"
        if response.get("results"):
            search_results += "相关信息：\n"
            for i, result in enumerate(response["results"][:3], 1):
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")
                search_results += f"{i}. {title}\n{content}\n来源：{url}\n\n"
        if not search_results:
            search_results = "抱歉未找到相关信息"

        return {
            "search_results":search_results,
            "step":"searched",
            "messages":[AIMessage(content=f"✅ 搜索完成！找到了相关信息，正在为您整理答案...")]
        }

    except Exception as e:
        error_msg = f"搜索时发生错误：{str(e)}"
        print(f"❌ {error_msg}")

        return {
            "search_results":f"搜索失败：{error_msg}",
            "step":"search_failed",
            "messages":[AIMessage(content="❌ 搜索遇到问题，我将基于已有知识为您回答")]
        }

def generate_answer_node(state:SearchState)->SearchState:
    if state["step"]=="search_failed":
        fallback_prompt = f"""搜索API暂时不可用，请基于您的知识回答用户的问题：

        用户问题：{state['user_query']}

        请提供一个有用的回答，并说明这是基于已有知识的回答。"""

        response = llm_deepseek.invoke([SystemMessage(content=fallback_prompt)])

        return {
            "final_answer":response.content,
            "step":"complete",
            "messages":[AIMessage(content=response.content)]
        }

    answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：

    用户问题：{state['user_query']}

    搜索结果：
    {state['search_results']}

    请要求：
    1. 综合搜索结果，提供准确、有用的回答
    2. 如果是技术问题，提供具体的解决方案或代码
    3. 引用重要信息的来源
    4. 回答要结构清晰、易于理解
    5. 如果搜索结果不够完整，请说明并提供补充建议"""

    response = llm_deepseek.invoke([SystemMessage(content=answer_prompt)])

    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)]
    }


def create_search_assistant():
    workflow = StateGraph(SearchState)

    workflow.add_node("understand",understand_query_search)
    workflow.add_node("search",tavily_search_node)
    workflow.add_node("answer",generate_answer_node)

    workflow.add_edge(START,"understand")
    workflow.add_edge("understand","search")
    workflow.add_edge("search","answer")
    workflow.add_edge("answer",END)

    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app

async def main():
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ 错误：请在.env文件中配置TAVILY_API_KEY")
        return
    app = create_search_assistant()

    print("🔍 智能搜索助手启动！")
    print("我会使用Tavily API为您搜索最新、最准确的信息")
    print("支持各种问题：新闻、技术、知识问答等")
    print("(输入 'quit' 退出)\n")

    session_count = 0

    while True:
        user_input = input("🤔 您想了解什么: ").strip()

        if user_input.lower() in ['quit','q','退出','exit']:
            print("感谢使用！再见！👋")
            break

        if not user_input:
            continue

        session_count+=1

        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}

        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": "",
            "search_query": "",
            "search_results": "",
            "final_answer": "",
            "step": "start"
        }

        try:
            print("\n" + "=" * 60)

            async for output in app.astream(initial_state,config=config):
                for node_name,node_output in output.items():
                    if "messages" in node_output and node_output["messages"]:
                        latest_message = node_output["messages"][-1]
                        if isinstance(latest_message,AIMessage):
                            if node_name=="understand":
                                print(f"🧠 理解阶段: {latest_message.content}")
                            elif node_name == "search":
                                print(f"🔍 搜索阶段: {latest_message.content}")
                            elif node_name == "answer":
                                print(f"\n💡 最终回答:\n{latest_message.content}")

            print("\n" + "=" * 60 + "\n")

        except Exception as e:
            print(f"❌ 发生错误: {e}")
            print("请重新输入您的问题。\n")


if __name__ == "__main__":
    asyncio.run(main())

























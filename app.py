"""
Chainlit UI for MCP Multi-Server Client
data_client.py와 동일한 기능을 웹 UI로 제공합니다.
"""
import json
import chainlit as cl
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4o", streaming=True)
user_id = "khs"


def load_mcp_config():
    """MCP 설정 파일 로드"""
    try:
        with open("./mcp_config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"설정 파일 읽기 오류: {e}")
        return None


def create_server_config():
    """서버 설정 생성"""
    config = load_mcp_config()
    server_config = {}

    if config and "mcpServers" in config:
        for server_name, server_data in config["mcpServers"].items():
            if "command" in server_data:
                server_config[server_name] = {
                    "command": server_data.get("command"),
                    "args": server_data.get("args", []),
                    "transport": "stdio",
                }
    return server_config


@cl.on_chat_start
async def start():
    """채팅 시작 시 초기화"""
    await cl.Message(content="🤖 MCP 서버 연결 중...").send()

    try:
        # MCP 클라이언트 생성
        server_config = create_server_config()

        if not server_config:
            await cl.Message(content="❌ MCP 서버 설정을 로드할 수 없습니다.").send()
            return

        client = MultiServerMCPClient(server_config)
        tools = await client.get_tools()

        server_names = list(server_config.keys())

        # 세션에 저장
        cl.user_session.set("client", client)
        cl.user_session.set("tools", tools)
        cl.user_session.set("chat_history", [])

        # 시스템 프롬프트 생성
        system_prompt = SystemMessage(
            content=(
                f"되도록이면 모든 요청에 순차적 사고(Sequentialthinking) 도구를 이용해 한국어로 사고하세요. "
                "Windows-Command-Line-MCP-Server, mem0-mcp, MCP SQLite Server, desktop-commander, pubmed-mcp-server을 적절히 사용하세요. "
                f"mem0-memory 툴을 사용해서 사용자의 메모리를 저장하고 불러오면서 대화하세요. "
                f"제 UserID는 {user_id} 입니다. "
            )
        )
        cl.user_session.set("system_prompt", system_prompt)

        # 성공 메시지
        success_msg = f"""
✅ **MCP 서버 연결 완료!**

📊 **로드된 툴**: {len(tools)}개
🔗 **연결된 서버**: {', '.join(server_names)}

💡 이제 질문을 입력하세요!
        """
        await cl.Message(content=success_msg).send()

    except Exception as e:
        error_msg = f"❌ **초기화 오류**: {str(e)}"
        await cl.Message(content=error_msg).send()
        import traceback
        traceback.print_exc()


@cl.on_message
async def main(message: cl.Message):
    """메시지 처리"""
    # 세션에서 데이터 가져오기
    tools = cl.user_session.get("tools")
    chat_history = cl.user_session.get("chat_history")
    system_prompt = cl.user_session.get("system_prompt")

    if not tools:
        await cl.Message(content="❌ 시스템이 초기화되지 않았습니다. 페이지를 새로고침해주세요.").send()
        return

    # 에이전트 생성 (매번 생성)
    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(model, tools)

    # 사용자 메시지 추가
    chat_history.append(HumanMessage(content=message.content))

    # 매 쿼리마다 시스템 프롬프트 추가
    messages_with_system = [system_prompt] + chat_history

    # 응답 메시지 생성
    msg = cl.Message(content="")
    await msg.send()

    try:
        # 스트리밍 응답
        final_messages = None
        tool_calls_text = ""

        async for event in agent.astream_events(
            {"messages": messages_with_system},
            version="v2",
            config={"recursion_limit": 100}
        ):
            kind = event["event"]

            # 툴 호출 표시
            if kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                tool_input = event.get("data", {}).get("input", {})
                tool_calls_text += f"\n\n🔧 **[TOOL CALL]** `{tool_name}`\n```json\n{json.dumps(tool_input, indent=2, ensure_ascii=False)}\n```"
                await msg.stream_token(tool_calls_text)
                tool_calls_text = ""

            # 툴 결과 표시
            elif kind == "on_tool_end":
                tool_output = event.get("data", {}).get("output", "")

                # ToolMessage 객체인 경우 content만 추출
                if hasattr(tool_output, 'content'):
                    output_str = tool_output.content
                else:
                    output_str = str(tool_output) if tool_output else ""

                # 글자수 제한 없이 전체 출력
                tool_calls_text += f"\n✅ **[TOOL RESULT]**\n```\n{output_str}\n```\n"
                await msg.stream_token(tool_calls_text)
                tool_calls_text = ""

            # LLM 스트리밍
            elif kind == "on_chat_model_stream":
                content = event.get("data", {}).get("chunk", {})
                if hasattr(content, "content") and content.content:
                    await msg.stream_token(content.content)

            # 최종 결과
            elif kind == "on_chain_end":
                output = event.get("data", {}).get("output")
                if output:
                    if isinstance(output, dict) and "messages" in output:
                        final_messages = output["messages"]
                    elif isinstance(output, list):
                        final_messages = output

        # 히스토리 업데이트
        if final_messages:
            cl.user_session.set("chat_history", final_messages)

        await msg.update()

    except Exception as e:
        error_msg = f"\n\n❌ **에러 발생**: {str(e)}"
        await msg.stream_token(error_msg)
        await msg.update()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)

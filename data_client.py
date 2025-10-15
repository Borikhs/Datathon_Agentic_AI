import json
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

load_dotenv()
#test
model = ChatOpenAI(model="gpt-4o")


def load_mcp_config():
    """현재 디렉토리의 MCP 설정 파일을 로드합니다."""
    try:
        with open("./mcp_config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"설정 파일을 읽는 중 오류 발생: {str(e)}")
        return None


def create_server_config():
    """MCP 서버 설정을 생성합니다."""
    config = load_mcp_config()
    server_config = {}

    if config and "mcpServers" in config:
        for server_name, server_config_data in config["mcpServers"].items():
            # command가 있으면 stdio 방식
            if "command" in server_config_data:
                server_config[server_name] = {
                    "command": server_config_data.get("command"),
                    "args": server_config_data.get("args", []),
                    "transport": "stdio",
                }
            # url이 있으면 sse 방식
            elif "url" in server_config_data:
                server_config[server_name] = {
                    "url": server_config_data.get("url"),
                    "transport": "sse",
                }

    return server_config


async def main():
    """멀티라운드 대화형 MCP 클라이언트"""
    print("=" * 50)
    print("MCP 서버 연결 중...")
    print("=" * 50)

    server_config = create_server_config()

    if not server_config:
        print("[ERROR] MCP 서버 설정을 로드할 수 없습니다.")
        return

    # MultiServerMCPClient 생성 (context manager 사용 안 함)
    client = MultiServerMCPClient(server_config)

    # 툴 로드
    tools = await client.get_tools()
    print(f"[OK] 총 {len(tools)}개 툴 로드됨")

    # 서버 목록 출력
    server_names = list(server_config.keys())
    print(f"[OK] 연결된 서버: {', '.join(server_names)}")

    user_id = "khs"

    system_prompt = SystemMessage(
        content=(
            f"되도록이면 모든 요청에 순차적 사고(Sequentialthinking) 도구를 이용해 한국어로 사고하세요. "
            "Windows-Command-Line-MCP-Server, mem0-mcp, MCP SQLite Server, desktop-commander, pubmed-mcp-server을 적절히 사용하세요."
            f"mem0-memory 툴을 사용해서 사용자의 메모리를 저장하고 불러오면서 대화하세요. "
            f"제 UserID는 {user_id} 입니다. "
            "Critical한 의사결정이 필요하면 대화를 끊고 사용자의 결정을 요구하세요."
        )
    )

    # 에이전트 생성
    agent = create_react_agent(model, tools)

    # 대화 히스토리 초기화
    chat_history = []

    print("=" * 50)
    print("멀티라운드 데이터 분석 에이전트 시작!")
    print("'exit' 또는 '종료' 입력 시 대화를 종료합니다.")
    print("=" * 50)

    # 멀티라운드 대화 루프
    while True:
        user_input = input("\n질문/피드백 입력 (exit 입력시 종료): ")

        if user_input.lower() in ["exit", "종료", "quit"]:
            print("\n대화를 종료합니다. 감사합니다!")
            break

        # 사용자 메시지 추가
        chat_history.append(HumanMessage(content=user_input))

        print("\n=====에이전트 실행 중=====")

        try:
            # 매 쿼리마다 시스템 프롬프트를 맨 앞에 강제로 추가
            messages_with_system = [system_prompt] + ['(순차적 사고 사용)'] + chat_history

            # astream_events로 실시간 스트리밍
            final_messages = None
            async for event in agent.astream_events(
                {"messages": messages_with_system}, version="v2", config={"recursion_limit": 100},
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    # LLM 스트리밍 출력 (선택적)
                    pass

                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})
                    print(f"[TOOL CALL] {tool_name}")
                    print(f"  └─ 입력: {tool_input}")

                elif kind == "on_tool_end":
                    tool_output = event.get("data", {}).get("output", "")

                    # ToolMessage 객체인 경우 content만 추출
                    if hasattr(tool_output, 'content'):
                        output_str = tool_output.content
                    else:
                        output_str = str(tool_output) if tool_output else ""

                    # 글자수 제한 없이 전체 출력
                    print(f"[TOOL RESULT] {output_str}")

                elif kind == "on_chain_end":
                    # 최종 체인 종료 시 전체 메시지 저장
                    output = event.get("data", {}).get("output")
                    if output:
                        # output이 dict이고 "messages" 키가 있는 경우
                        if isinstance(output, dict) and "messages" in output:
                            final_messages = output["messages"]
                        # output이 이미 리스트인 경우 (메시지 리스트)
                        elif isinstance(output, list):
                            final_messages = output

            # 응답 추출
            if final_messages:
                final_answer = final_messages[-1].content
                print("\n=====응답=====")
                print(final_answer)

                # 전체 응답 메시지를 히스토리에 업데이트
                chat_history = final_messages
            else:
                print("\n[경고] 응답을 받지 못했습니다.")

        except Exception as e:
            print(f"\n[ERROR] 에이전트 실행 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


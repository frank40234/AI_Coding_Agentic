import os
import subprocess
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

# LangGraph & LangChain 核心元件
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun

from pydantic import BaseModel, Field

# ==========================================
# Step 1: 載入環境變數與初始化 LLM (必須最先執行)
# ==========================================
load_dotenv()

ACTIVE_PROVIDER = input('\n請輸入 gemini 或是 lm_studio：\n> ')

if ACTIVE_PROVIDER == 'gemini':
    llm = ChatGoogleGenerativeAI(
        model = os.getenv('GEMIN_MODEL'),
        google_api_key = os.getenv('GOOGLE_API_KEY'),
        temperature=0
    )
elif ACTIVE_PROVIDER == 'lm_studio':
    llm = ChatOpenAI(
        base_url=os.getenv('LM_STUDIO_URL'), # 例如 http://localhost:1234/v1
        api_key="lm-studio", # LM Studio 通常不需要真實 Key，但需填寫佔位符
        model_name=os.getenv('LM_STUDIO_MODEL'),
        temperature=0
    )
else:
    raise ValueError("未知的提供者，請輸入 gemini 或 lm_studio")


# ==========================================
# Step 2: 定義全局狀態 (State) 與資料結構
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    sender: str 
    current_task: str
    shared_memory: str
    memory_path: str
    retry_count: int
    next_agent: str

class RouteResponse(BaseModel):
    reasoning: str = Field(
        description="簡短說明為什麼做出這個決策，或是指派給下一個角色的具體任務指示。"
    )
    memory_update: str = Field(
        default="", 
        description="如果任務達到一個里程碑，請在這裡寫下要更新到專案記憶檔 (MEMORY.md) 的總結資訊。如果不需要更新請留空。"
    )
    next_agent: Literal["FINISH", "coder", "reviewer"] = Field(
        description="決定下一個要執行的角色。如果任務已完全結束，請輸出 FINISH。"
    )


# ==========================================
# Step 3: 定義系統工具 (Tools)
# ==========================================
@tool
def read_local_file(file_path: str) -> str:
    """讀取本地端檔案的內容。當你需要查看現有 C# 程式碼時呼叫此工具。"""
    if not os.path.exists(file_path):
        return f"錯誤：找不到檔案 {file_path}"
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

@tool
def write_local_file(file_path: str, content: str) -> str:
    """將程式碼寫入本地端檔案。當你需要建立新檔案或覆蓋舊檔案時呼叫此工具。"""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"成功寫入檔案：{file_path}"

@tool
def execute_dotnet_command(command: str) -> str:
    """執行本地端的終端機指令（如 dotnet run, dotnet build, dotnet new console 等）並回傳執行結果。"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True, 
            text=True,           
            encoding='utf-8',    
            errors='replace',    
            check=False          
        )
        if result.returncode == 0:
            return f"執行成功。\n輸出內容:\n{result.stdout}"
        else:
            return f"執行失敗 (代碼 {result.returncode})。\n錯誤訊息:\n{result.stderr}\n標準輸出:\n{result.stdout}"
    except Exception as e:
        return f"系統呼叫發生例外錯誤: {str(e)}"

ddg_search = DuckDuckGoSearchRun()

@tool
def search_web(query: str) -> str:
    """當遇到未知的 C# 語法、需要確認 NuGet 套件名稱、或查閱最新技術文件時，使用此工具進行網路搜尋。"""
    try:
        return ddg_search.invoke(query)
    except Exception as e:
        return f"搜尋發生錯誤: {str(e)}"


# ==========================================
# Step 4: Agent 工廠與共用執行函數
# ==========================================
def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])
    if tools:
        return prompt | llm.bind_tools(tools)
    else:
        return prompt | llm

def run_agent_node(agent_runnable, name: str, state: AgentState):
    result = agent_runnable.invoke(state)
    if isinstance(result, AIMessage):
        result.name = name
        
        # === [新增防呆機制] 如果 AI 產生空轉 (沒有文字也沒有呼叫工具) ===
        if not result.content and not getattr(result, 'tool_calls', None):
            result.content = f"【系統攔截】{name} 未產生任何實質動作或回應。請工程師提供明確的測試指令，或請主管重新指派。"
        # =========================================================

    if hasattr(result, 'tool_calls') and result.tool_calls:
        tool_names = ", ".join([tc['name'] for tc in result.tool_calls])
        print(f"[{name.upper()} 動作] -> 呼叫工具: {tool_names}")
    else:
        text_content = ""
        if isinstance(result.content, str):
            text_content = result.content
        elif isinstance(result.content, list):
            for item in result.content:
                if isinstance(item, str):
                    text_content += item
                elif isinstance(item, dict) and 'text' in item:
                    text_content += item['text']
                    
        content_preview = text_content[:50].replace('\n', ' ') + ("..." if len(text_content) > 50 else "")
        print(f"[{name.upper()} 發言] -> {content_preview}")

    return {
        "messages": [result],
        "sender": name
    }


# ==========================================
# Step 5: 實作多代理人節點 (Nodes)
# ==========================================

# --- Coder 節點 ---
coder_tools = [read_local_file, write_local_file, search_web, execute_dotnet_command]
coder_system_prompt = """你是一位資深的 C# 開發工程師。
【專案記憶 / 已完成里程碑】：
{shared_memory}

你的任務是根據整體的專案目標或測試員的回報，撰寫與修改程式碼。
你可以呼叫 `execute_dotnet_command` 來執行終端機指令 (例如 `dotnet new mvc -n TestEAMS`, `dotnet add package`, `mkdir` 等) 以建立專案架構與檔案。
遇到未知的語法請使用網路搜尋。完成修改或建置後，請回覆一段簡短的總結，告知已完成任務。"""
coder_agent = create_agent(llm, coder_tools, coder_system_prompt)

def coder_node(state: AgentState):
    return run_agent_node(coder_agent, "coder", state)

# --- Reviewer 節點 ---
# --- 6. 實作 Reviewer 節點 ---
reviewer_tools = [execute_dotnet_command]
reviewer_system_prompt = """你是一位嚴格的軟體測試工程師。
【專案記憶 / 已完成里程碑】：
{shared_memory}

你的唯一職責是呼叫 `execute_dotnet_command` 工具來驗證專案狀態（例如編譯、執行或測試）。

【重要測試規範】：
1. 當主管或工程師要求你測試時，你**必須**呼叫 `execute_dotnet_command` 工具。絕不能只回覆文字而不呼叫工具。
2. 請根據上下文（例如 Coder 回報的專案建立路徑）自行推斷要執行的目錄與指令。例如：`cd C:\\路徑\\專案名稱 && dotnet build`。
3. 收到工具回傳的實際結果後，將結果客觀地總結並回報，明確指出成功或失敗。
4. 如果你真的無法從對話中推斷出專案路徑，請呼叫工具執行 `dir` 或 `ls` 來尋找，不要直接放棄。"""
reviewer_agent = create_agent(llm, reviewer_tools, reviewer_system_prompt)

def reviewer_node(state: AgentState):
    return run_agent_node(reviewer_agent, "reviewer", state)

# --- Supervisor 節點 ---
supervisor_system_prompt = """你是一位專案開發主管，負責管理以下兩位成員：
- 'coder': 負責撰寫與修改程式碼，或執行專案建立指令。
- 'reviewer': 負責執行終端機指令(如 dotnet build)驗證程式。

當前的任務總目標是：{current_task}

【專案記憶 / 已完成里程碑】：
{shared_memory}

請根據下方的對話紀錄，決定下一步。嚴格遵守以下路由規則：
1. 任務剛開始，或者 'reviewer' 測試發現錯誤需要修正，或者還有尚未完成的開發任務時，必須交給 'coder'。
2. 當 'coder' 回報已完成某個階段的撰寫/修改時，必須交給 'reviewer' 進行測試驗證。
3. 當 'reviewer' 回報測試成功後，請評估「任務總目標」是否已經「完全」達成：
   - 如果還有後續目標未完成（例如：只建立了專案，但還沒寫程式碼），必須將任務交回給 'coder' 繼續開發。
   - 只有當所有目標皆已完成，且最後一次測試也成功時，才回傳 'FINISH'。
4. 絕對不要讓同一個角色連續空轉（例如 reviewer 回報測試成功後，絕不能再交給 reviewer，應交給 coder 繼續下個任務或 FINISH）。
"""
supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", supervisor_system_prompt),
    MessagesPlaceholder(variable_name="messages"),
])

supervisor_chain = supervisor_prompt | llm.with_structured_output(RouteResponse)

def supervisor_node(state: AgentState):
    print("\n[主管思考中...]")
    decision = supervisor_chain.invoke({
        "messages": state["messages"],
        "current_task": state["current_task"],
        "shared_memory": state.get("shared_memory", "")
    })
    
    new_memory = state.get("shared_memory", "")
    memory_path = state.get("memory_path", "")
    
    if decision.memory_update:
        if memory_path:
            # 同步寫入實體檔案
            os.makedirs(os.path.dirname(memory_path), exist_ok=True)
            if not os.path.exists(memory_path):
                with open(memory_path, "w", encoding="utf-8") as f:
                    f.write("# 專案開發記憶 (MEMORY)\n\n")
            with open(memory_path, "a", encoding="utf-8") as f:
                f.write(f"- {decision.memory_update}\n")
            print(f"[記憶更新] -> 已將進度寫入 {memory_path}: {decision.memory_update}")
        else:
            print(f"[記憶更新] -> (僅暫存): {decision.memory_update}")
        new_memory += f"\n- {decision.memory_update}"

    print(f"[主管說明] -> {decision.reasoning}")
    print(f"[決策出爐] 任務轉交給 -> {decision.next_agent.upper()}")
    
    # 建立一則主管發出的指示訊息，讓下一個接收任務的角色知道要做什麼
    instruction_msg = HumanMessage(content=f"【主管指派任務】\n{decision.reasoning}")
    
    return {
        "messages": [instruction_msg],
        "shared_memory": new_memory,
        "next_agent": decision.next_agent
    }


# ==========================================
# Step 6: 構建狀態圖 (StateGraph Wiring)
# ==========================================
all_tools = coder_tools + reviewer_tools
unique_tools = list({t.name: t for t in all_tools}.values())
tool_node = ToolNode(unique_tools)

workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("coder", coder_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("tools", tool_node)

def agent_edge(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "supervisor"

def route_tool_response(state: AgentState):
    return state["sender"]

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "coder": "coder",
        "reviewer": "reviewer",
        "FINISH": END
    }
)

workflow.add_conditional_edges("coder", agent_edge)
workflow.add_conditional_edges("reviewer", agent_edge)
workflow.add_conditional_edges("tools", route_tool_response)

app = workflow.compile()


# ==========================================
# Step 7: 系統執行區塊
# ==========================================
if __name__ == "__main__":
    task_document = ""

    project_type = input('\n請選擇開發模式：\n1. 新專案開發 (建立全新專案)\n2. 原有專案繼續開發\n> ').strip()
    
# 升級：支援讀取多模組規劃書資料夾
    has_task_file = input('\n是否有使用 pm_architect 產生的「規劃書資料夾」？ (Y/N)：\n> ').strip().upper()
    if has_task_file == 'Y':
        task_dir = input('請輸入規劃書資料夾路徑 (例如 ./Docs)：\n> ').strip()
        
        # 檢查該路徑是否存在且為一個資料夾 (對應 C# 的 Directory.Exists)
        if os.path.exists(task_dir) and os.path.isdir(task_dir):
            task_document = ""
            # 遍歷資料夾內所有的 .md 檔案 (對應 C# 的 Directory.GetFiles(path, "*.md"))
            for file_name in os.listdir(task_dir):
                if file_name.endswith(".md"):
                    file_path = os.path.join(task_dir, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 將檔名作為標題，讓 AI 清楚知道現在讀的是哪個模組的文件
                        task_document += f"\n\n=== 【架構文件：{file_name}】 ===\n"
                        task_document += f.read()
            print(f"\n[系統] 已成功載入 {task_dir} 內的所有規劃書文件作為專案 Context。")
        else:
            print(f"\n[系統] 警告：找不到資料夾 {task_dir}，將跳過規劃書讀取。")

    specific_task = input('\n請輸入具體要執行的開發任務 (例如：請建立專案架構並實作 Models)：\n> ').strip()
    
    existing_memory = ""
    memory_file = ""
    project_dir = ""
    task_description = ""
    
    if project_type == '1':
        parent_dir = input('\n請輸入新專案要放置的「上層資料夾路徑」 (例如 C:\\02Project\\GitProject)：\n> ').strip()
        project_name = input('請輸入「專案名稱」 (例如 TestEAMS)：\n> ').strip()
        project_tech = input('請輸入「專案類型與版本」 (例如 ASP.NET Core 8.0 MVC 或 WinForm 等)：\n> ').strip()
        
        if parent_dir and project_name:
            project_dir = os.path.join(parent_dir, project_name)
            memory_file = os.path.join(project_dir, "MEMORY.md")
            existing_memory = "# 專案開發記憶 (MEMORY)\n\n"
            
            print(f"\n[系統] 專案路徑設定為 {project_dir}，將在專案建置與開發過程中自動產生 MEMORY.md。")
            
            task_description = (
                f"【新專案建置指示】\n"
                f"- 上層目錄：`{parent_dir}`\n"
                f"- 專案名稱：`{project_name}`\n"
                f"- 專案類型與版本：{project_tech}\n"
                f"- 完整工作目錄：`{project_dir}`\n\n"
                f"請 Coder 首先執行建置專案的指令。\n\n"
                f"【當前開發任務】\n{specific_task}"
            )
            
    elif project_type == '2':
        project_dir = input('\n請輸入「原有專案」的根目錄路徑：\n> ').strip()
        if project_dir:
            memory_file = os.path.join(project_dir, "MEMORY.md")
            if os.path.exists(memory_file):
                with open(memory_file, "r", encoding="utf-8") as f:
                    existing_memory = f.read()
                print(f"\n[系統] 已讀取舊有記憶檔案，恢復之前的開發進度。")
            else:
                existing_memory = "# 專案開發記憶 (MEMORY)\n\n"
                os.makedirs(project_dir, exist_ok=True)
                with open(memory_file, "w", encoding="utf-8") as f:
                    f.write(existing_memory)
                print(f"\n[系統] 找不到舊有記憶檔案，已自動建立新記憶檔案。")
            
            task_description = (
                f"【專案工作目錄】\n請務必在以下目錄中執行操作與修改：`{project_dir}`\n\n"
                f"【當前開發任務】\n{specific_task}"
            )

    if not task_description:
        task_description = f"【當前開發任務】\n{specific_task}"

    if task_document:
        task_description = f"【系統與架構規劃書內容】\n{task_document}\n\n" + task_description

    initial_state = {
        "messages": [HumanMessage(content=task_description)],
        "current_task": task_description,
        "shared_memory": existing_memory,
        "memory_path": memory_file,
        "retry_count": 0,
        "sender": "user",
        "next_agent": ""
    }

    print(f"啟動的 AI 提供者: {ACTIVE_PROVIDER.upper()}")
    print("Multi-Agent 團隊開始執行任務...\n" + "-"*30)

    final_state = app.invoke(initial_state)

    print("-" * 30)
    print("任務結束！")
    
    print("\n[團隊總結]:")
    for msg in final_state["messages"][-3:]:
        sender_name = getattr(msg, "name", "unknown")
        if not sender_name or sender_name == "unknown":
            sender_name = "Tools/System"
            
        print(f"\n--- {sender_name.upper()} ---")
        
        # 安全提取內容
        text_content = ""
        if isinstance(msg.content, str):
            text_content = msg.content
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, str):
                    text_content += item
                elif isinstance(item, dict) and 'text' in item:
                    text_content += item['text']
                    
        print(text_content[:200] + ("..." if len(text_content) > 200 else ""))
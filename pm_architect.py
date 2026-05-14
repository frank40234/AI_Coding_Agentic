# pm_architect.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI # 若使用 LM Studio 請自行替換對應模組
# pm_architect.py (部分新增與修改)
# --- [新增] 引入 Pydantic 用來定義結構化輸出 ---
from pydantic import BaseModel, Field
from typing import List

# 1. 載入環境變數 (對應 C# 讀取 appsettings.json 或 Environment 變數)

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model=os.getenv('GEMIN_MODEL'),
    google_api_key=os.getenv('GOOGLE_API_KEY'),
    temperature=0.2 
)

# ==========================================
# 步驟 A: 定義 AI 輸出的資料結構 (類似 C# 的 DTO 類別)
# ==========================================

# 定義「子模組」的資料結構
class SubModulePlan(BaseModel):
    # Field 裡面的 description 就是寫給 AI 看的屬性註解
    module_name: str = Field(description="模組的英文名稱，將作為檔名。例如：Inventory, HR, Finance")
    module_content: str = Field(description="該模組專屬的 Markdown 內容（包含模組功能、Table Schema、開發步驟）")

# 定義「總體架構」的資料結構 (包含主架構文件與多個子模組)
class ArchitecturePlan(BaseModel):
    master_content: str = Field(description="Master_Architecture.md 的完整 Markdown 內容（包含共用設定、SSO、Base Table）")
    # 這裡對應 C# 的 List<SubModulePlan>
    sub_modules: List[SubModulePlan] = Field(description="需要獨立拆分的子模組清單與內容")

# 2. 初始化 LLM 實體 (這類似 C# 中實例化一個 Service 類別)
# temperature 設為 0.2，給予 AI 一點點創造力來撰寫企劃，但不要過度發散
llm = ChatGoogleGenerativeAI(
    model=os.getenv('GEMIN_MODEL'),
    google_api_key=os.getenv('GOOGLE_API_KEY'),
    temperature=0.2 
)

# ==========================================
# 步驟 B: 執行強型別結構化輸出與檔案生成
# ==========================================
# ==========================================
# 步驟 B: 執行強型別結構化輸出與檔案生成
# ==========================================
# ==========================================
# 步驟 B: 執行強型別結構化輸出與檔案生成
# ==========================================
def generate_architecture():
    # 1. 詢問檔案要存放的目錄 (對應 C# 的 Console.ReadLine)
    save_dir = input("\n請輸入規畫書要存放的資料夾路徑 (例如: ./Docs，留空則存於當前目錄):\n> ").strip()
    
    # 如果使用者有輸入路徑，且該資料夾不存在，就自動建立 (對應 C# 的 Directory.CreateDirectory)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[系統] 已自動建立資料夾：{save_dir}")
        
    # 如果使用者留空，預設使用當前目錄 "."
    if not save_dir:
        save_dir = "."

    # 2. 取得使用者原始需求
    print("\n請輸入您的系統開發需求（支援多行貼上）。")
    print("【注意】：輸入完畢後，請在新的一行輸入 'EOF' 並按 Enter 鍵確認提交：")
    print("> ", end="")
    
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'EOF':
            break
        lines.append(line)
        
    user_request = "\n".join(lines)
    if not user_request.strip():
        print("[系統] 需求不可為空，程式結束。")
        return

    # 3. 定義指示 Prompt
# 3. 定義指示 Prompt (加入強制繁體中文限制)
    prompt = f"""你是一位資深軟體架構師。請根據以下需求，規劃巨型系統架構。
    請嚴格遵守以下拆分原則：
    1. master_content 必須包含：整體概述、C# 技術堆疊、全域 SSO 認證機制、共用 Base 欄位。
    2. sub_modules 必須依據業務邏輯拆分 (如 Inventory, HR)，並列出專屬 Schema 與任務清單。

    【極度重要：語言限制】
    除了程式碼、資料表名稱、欄位名稱可以保留英文外，所有的「說明、描述、文件內文、任務清單」都**絕對必須**使用「繁體中文 (Traditional Chinese)」撰寫。

    需求：{user_request}"""
    
    print("\n[系統] 架構師正在規劃主架構與子模組，請稍候...\n")
    
    # 4. 綁定強型別並呼叫 LLM
    structured_llm = llm.with_structured_output(ArchitecturePlan)
    plan = structured_llm.invoke(prompt)
    
    # 5. 組合路徑並寫入主架構 (對應 C# 的 Path.Combine 與 File.WriteAllText)
    master_path = os.path.join(save_dir, "Master_Architecture.md")
    with open(master_path, 'w', encoding='utf-8') as f:
        f.write(plan.master_content)
    print(f"[系統] 已產出主架構：{master_path}")
    
    # 6. 遍歷並寫入子模組
    for sub in plan.sub_modules:
        file_name = f"{sub.module_name}.md"
        # 組合子模組的完整存檔路徑
        sub_path = os.path.join(save_dir, file_name) 
        with open(sub_path, 'w', encoding='utf-8') as f:
            f.write(sub.module_content)
        print(f"[系統] 已產出子模組：{sub_path}")
        
    input("\n【人工複查階段】請確認產出的多份 Markdown 檔案，確認完畢後按 [Enter] 鍵結束此階段...")

# 8. Python 的主程式進入點 (對應 C# 的 static void Main(string[] args))
if __name__ == "__main__":
    generate_architecture()
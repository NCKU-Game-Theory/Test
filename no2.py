import os
import getpass 
import time
import google.generativeai as genai
import sqlite3 # ⭐ 用於時序資料庫
import glob    # ⭐ 用於自動掃描 .docx 檔案

# --- Agentic RAG 核心套件 (已修正 Import 路徑) ---
from langchain_classic.agents import AgentExecutor, create_react_agent  # ✅ AgentExecutor 在這裡
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool, Tool # ✅ Tool (類別) 和 @tool (裝飾器) 在這裡
# ---

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document 
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------
# 第 1 部分：設定環境
# -------------------------------------

if "GEMINI_API_KEY" not in os.environ:
    # ⚠️ 警告：請替換成你自己的金鑰
    os.environ["GEMINI_API_KEY"] = 'AIzaSyDArVwaXi7y4GLZKskSvv_slNHke2xqUDc' 

print("Environment setup complete. API Key loaded.")

# --- 初始化 LLM (Agent 的大腦) ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.7 # 讓 Agent 的思考更穩定
)

# -------------------------------------
# 第 2 部分：【Tool 1】設定 RAG 知識庫 (策略、文獻)
# -------------------------------------

@tool
def search_strategy_guide(query: str) -> str:
    """
    Searches the RAG knowledge base (Word documents) for game strategies, 
    psychology, or literature. Use this to find *how* to play.
    Input MUST be a natural language query.
    """
    print(f"\n[Agent Action]: Calling RAG Tool with query: '{query}'")
    try:
        # 在 RAG_vectorstore 中執行相似度搜尋
        results = RAG_vectorstore.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Error searching RAG: {e}"

def setup_vectorstore():
    """
    (你的要求 3)
    掃描資料夾中所有的 .docx 檔案，並將它們全部載入 Vectorstore。
    """
    print("Initializing Embedding model...")
    # ✅ (修正) 確保模型名稱正確
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.environ["GEMINI_API_KEY"])
    
    # 建立一個空的 vectorstore
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db_agent")
    vectorstore.delete_collection() # (清空舊的，確保每次都是最新)
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db_agent")

    # 1. 找到所有 .docx 檔案
    doc_files = glob.glob("*.docx")
    if not doc_files:
        print("Warning: No .docx files found in the directory. (RAG Tool will be empty)")
        return vectorstore # 回傳一個空的 RAG Store

    print(f"Found {len(doc_files)} .docx files to load: {doc_files}")
    
    all_splits = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # 2. 載入並分割所有檔案
    for doc_file in doc_files:
        try:
            loader = UnstructuredWordDocumentLoader(doc_file)
            raw_documents = loader.load()
            splits = text_splitter.split_documents(raw_documents)
            
            # (可選) 為 RAG 資料加上來源標籤
            for doc in splits:
                doc.metadata["source"] = doc_file
            
            all_splits.extend(splits)
            print(f"Successfully loaded and split '{doc_file}'.")
        except Exception as e:
            print(f"Error loading '{doc_file}': {e}. Skipping.")

    # 3. 將所有文件塊一次性加入 Vectorstore
    if all_splits:
        vectorstore.add_documents(all_splits)
        print(f"✅ RAG Knowledge Base is ready. Loaded {len(all_splits)} chunks.")
    
    return vectorstore

# -------------------------------------
# 第 3 部分：【Tool 2】設定時序資料庫 (遊戲歷史)
# -------------------------------------

# --- 設定 SQLite 資料庫 ---
DB_FILE = "game_history.db"

def setup_database():
    """
    建立一個 SQLite 資料庫和 game_history 表格 (如果不存在)。
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS game_history (
        round_number INTEGER PRIMARY KEY,
        user_move TEXT,
        ai_move TEXT,
        winner TEXT,
        ai_reasoning TEXT
    )
    """)
    conn.commit()
    conn.close()
    print(f"✅ Time-Series Database '{DB_FILE}' is ready.")

def add_history_to_db(round_num, user, ai, win, reason):
    """
    將一局的結果寫入 SQLite。
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO game_history (round_number, user_move, ai_move, winner, ai_reasoning) VALUES (?, ?, ?, ?, ?)",
        (round_num, user, ai, win, reason)
    )
    conn.commit()
    conn.close()

@tool
def query_game_history(query: str) -> str:
    """
    (你的要求 2)
    Queries the Time-Series SQL database of game history.
    Use this to find *what* happened in past rounds.
    Input MUST be a valid SQL query.
    The table name is 'game_history'.
    Columns are: round_number, user_move, ai_move, winner, ai_reasoning.
    """
    print(f"\n[Agent Action]: Calling History Tool with SQL: '{query}'")
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return str(results)
    except Exception as e:
        return f"Error running SQL query: {e}. (Hint: Check your SQL syntax and column names.)"

# -------------------------------------
# 第 4 部分：建立 AI 秘書 (Agent)
# -------------------------------------

# --- Agent 的核心 Prompt ---
AGENT_PROMPT = """
You are a "Rock-Paper-Scissors" AI Agent. Your goal is to analyze the human player and win.
You have access to two specialized tools.


**Your Task:**
It is Round {round_number}. The user has already made their move (you don't know what it is).
You MUST decide your move. 

**Your Thought Process (MUST follow these steps):**

1.  **Analyze History (Tool 1):** First, you MUST use the `query_game_history` tool.
    * Formulate a SQL query to retrieve relevant past games (e.g., the last 3-5 rounds, or user's move statistics).
    * *Example Query:* `SELECT round_number, user_move, winner FROM game_history ORDER BY round_number DESC LIMIT 3`

2.  **Formulate Hypothesis:**
    * Based on the SQL results, analyze the user's pattern (e.g., "User is on a 'stone' streak," or "User follows a 'Win-Stay, Lose-Shift' pattern").

3.  **Find Strategy (Tool 2 - Optional):**
    * (Optional) If you identified a pattern, you MAY use `search_strategy_guide` to find a counter-strategy.
    * *Example Query:* `How to counter a 'Win-Stay, Lose-Shift' pattern?`

4.  Final Decision:
    * Synthesize all information (History + RAG Strategy) to make your final choice.
    * **CRITICAL:** Once you have your decision, you MUST output it using the `Final Answer:` prefix.
    * The content *after* the `Final Answer:` prefix MUST be in the following two-line format:
    Line 1: Your single-word move ('paper', 'stone', or 'scissors').
    Line 2: A detailed explanation for your choice, referencing your analysis.

**Tools Available:**
You MUST use one of the following tools:

{tool_names}  # <--- ⭐⭐⭐【最終修復】: 在 {tools} 上方加入這一行

Here are the descriptions of the tools:
{tools}

**Begin!**

User's Input (Human):
{input}

**Your Thought Process and Actions (Scratchpad):**
{agent_scratchpad}
"""

# --- 全域變數，供 Tool 使用 ---
RAG_vectorstore = None

def main():
    global RAG_vectorstore # 讓 @tool 函式可以抓到
    
    # --- ⭐⭐⭐【V-Agentic 最終修復】: 重置資料庫 ⭐⭐⭐
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"Removed old database: {DB_FILE}")
    # (Chroma DB 會在 setup_vectorstore() 內部自動重置)
    # --- ⭐⭐⭐【修復結束】⭐⭐⭐

    # --- 啟動 ---
    setup_database() # 現在這會建立一個「全新的」DB
    RAG_vectorstore = setup_vectorstore()

    # --- 建立 Agent ---
    tools = [query_game_history, search_strategy_guide]

    prompt_template = PromptTemplate.from_template(AGENT_PROMPT)
    
    # ✅ (修正) 確保 Import 路徑正確
    agent = create_react_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True  # ⭐⭐⭐【最終修復】: 加上這一行 ⭐⭐⭐
    )

    # --- 遊戲迴圈 (你的要求 4) ---
    print("\n" + "="*30)
    print("Welcome to Agentic RAG-RPS! (V-Agentic)")
    print("="*30)

    round_count = 1
    valid_moves = ["scissors", "stone", "paper"]

    while True:
        print(f"\n--- ROUND {round_count} ---")

        # 1. 提示使用者先出拳
        my_move = ""
        while my_move not in valid_moves:
            my_move = input(f"Make your move ({'/'.join(valid_moves)}): ").lower()
            if my_move not in valid_moves:
                print(f"Invalid input. Please enter one of: {', '.join(valid_moves)}")
        
        print(f"\nYou chose: {my_move}")

        # 2. 準備 Agent 的輸入
        # (我們不需要傳 'my_move'，Agent 是雙盲的)
        agent_input = f"It is now Round {round_count}. Analyze the game history and make your move."

        # 3. 執行 Agentic RAG
        print("Gemini Agent is thinking...")
        time.sleep(1)
        
        # ⭐ 呼叫 AI 秘書 (Agent)
        response_dict = agent_executor.invoke({
            "input": agent_input,
            "round_number": round_count # 傳入 Prompt 變數
        })
        
        raw_llm_output = response_dict['output'] # 這是 AI 的最終答案

        # 4. 解析 AI 輸出 (你的要求 4)
        gemini_choice = ""
        gemini_reasoning = ""
        try:
            parts = raw_llm_output.strip().split('\n', 1)
            gemini_choice = parts[0].strip().lower()
            if len(parts) > 1:
                gemini_reasoning = parts[1].strip()
            else:
                gemini_reasoning = "(Agent failed to provide reasoning.)"
        except Exception as e:
            gemini_choice = raw_llm_output.strip().lower()
            gemini_reasoning = f"(Error parsing Agent output: {e})"

        # 5. 判斷勝負 (你的要求 4)
        print("-" * 30)
        print(f"Gemini chose: {gemini_choice}")
        if gemini_reasoning:
            print(f"Gemini's reasoning: {gemini_reasoning}")
        print("-" * 30)
        
        winner = ""
        if gemini_choice not in valid_moves:
            winner = "GAME FAILED"
            print(f"GAME FAILED! Agent's move was '{gemini_choice}'.")
        elif my_move == gemini_choice:
            winner = "Draw"
            print("🎉 Result: It's a draw!")
        elif (my_move == "stone" and gemini_choice == "scissors") or \
             (my_move == "scissors" and gemini_choice == "paper") or \
             (my_move == "paper" and gemini_choice == "stone"):
            winner = "User"
            print("🎉 Result: Congratulations! You win!")
        else:
            winner = "AI"
            print("😭 Result: Oh no! You lose!")

        # 6. 將結果寫入「時序資料庫」
        try:
            add_history_to_db(round_count, my_move, gemini_choice, winner, gemini_reasoning)
            print("Game result saved to Time-Series Database.")
        except Exception as e:
            print(f"Error saving game result to SQL DB: {e}")

        # 7. 詢問是否繼續 (你的要求 4)
        round_count += 1
        play_again = input("\nPlay another round? (y/n): ").lower().strip()
        if play_again != 'y':
            print("\nThank you for playing!")
            break

# --- 程式進入點 ---
if __name__ == "__main__":
    main()
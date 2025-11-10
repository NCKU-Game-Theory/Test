import os
import getpass 
import time # 導入 time 模組，用於 "thinking..." 效果
# 【必要修正】: "from google import genai" 會導致 ImportError
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# 【新增導入】: 為了將歷史紀錄存回 RAG，我們需要 "Document" 類別
from langchain_core.documents import Document 

# -------------------------------------
# 第 1 部分：設定環境與 API 金鑰 (依您的要求保留)
# -------------------------------------

if "GEMINI_API_KEY" not in os.environ:
 # ⚠️ 警告：您已知曉將 API 金鑰寫死在程式碼中的風險。
   os.environ["GEMINI_API_KEY"] = 'AIzaSyC41yvKh5Bt7XiFN5msH82WDYxWME4_GmI' 

print("Environment setup complete. API Key loaded.")

# -------------------------------------
# 第 2 部分：【已升級】載入並索引 .docx 規則 (加入 Metadata)
# -------------------------------------

word_file_path = "game rules and output format.docx"
print(f"Loading game rules from '{word_file_path}'...")

loader = UnstructuredWordDocumentLoader(word_file_path)
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
final_splits = text_splitter.split_documents(raw_documents)

# 【v2.0 升級點 1】: 為 .docx 規則加上 "rules" 標籤 (metadata)
for doc in final_splits:
    doc.metadata = {"source": "rules"}

print(f"Total text chunks for indexing: {len(final_splits)}")
print("Initializing Embedding model...")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.environ["GEMINI_API_KEY"])

print("Building 'Game Rules' vector database index...")
# Chroma 會自動索引我們剛剛加入的 'source': 'rules' 標籤
vectorstore = Chroma.from_documents(documents=final_splits, embedding=embeddings)

print("="*30)
print("✅ Game Rules vector database indexing complete.")
print("="*30)

# -------------------------------------
# 第 3 部分：【已升級】建立 RAG 鏈 (移除固定鏈)
# -------------------------------------

print("Initializing Gemini chat model...")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=1.0
)

# 【v2.0 升級點 2】: 
# 移除固定的 retriever 和 rag_chain，我們將在迴圈中動態建立它們
# retriever = vectorstore.as_retriever() # <- 已移除
# rag_chain = ( ... ) # <- 已移除

template = """
You have two types of information in the context: 'Game Rules' and 'Game History'.

TASK 1 (DECISION): 
Use the 'Game Rules' AND the 'Game History' to analyze the opponent and decide your next move (paper, stone, or scissors).

TASK 2 (OUTPUT): 
You MUST output your decision. Your output MUST follow the "Output Format" rule found in the 'Game Rules'.
The 'Game Rules' state your output MUST be ONE WORD: 'paper', 'stone', or 'scissors'.

CRITICAL WARNING: 
The 'Game History' is ONLY for analysis. 
DO NOT copy the format from the 'Game History' (e.g., "AI played...", "Game 7:...", "[User=...").
Your final response MUST be one single word.

---
[Retrieved Context (Rules and Output format & History)]:
{context}
---

[My Instruction]:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

print("RAG components are ready.") # <- 文字已修改

# -------------------------------------
# 第 4 部分：【全新合併】多回合制遊戲迴圈 (V2.0 + 您的 Prompt)
# -------------------------------------

print("\n" + "="*30)
print("Welcome to Multi-Round RAG-RPS! (V2.0 Merged)")
print("="*30)

round_count = 1
valid_moves = ["scissors", "stone", "paper"] 

import time
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# (假設您的 llm, prompt, vectorstore, valid_moves 等變數已在上方定義好)

# 【v2.0 升級點 1】: 在迴圈外建立一個列表，用於儲存每一局的贏家
game_outcomes = [] 
round_count = 1 # (您原本的程式碼中應該有這行，確保它在迴圈外)

while True: # 建立一個無限迴圈，直到使用者選擇退出
    print(f"\n--- ROUND {round_count} ---")

    # 1. 記憶體開關
    use_memory = input("Allow AI to see past game history? (y/n): ").lower().strip()
    
    # 【v2.0 升級點 3】: 根據 'use_memory' 動態建立 retriever
    current_retriever = None 
    if use_memory == 'y' and round_count > 1:
        print("AI is reviewing game history...")
        current_retriever = vectorstore.as_retriever()
    else:
        if round_count > 1 and use_memory != 'y':
            print("AI is playing *without* memory...")
        current_retriever = vectorstore.as_retriever(
            search_kwargs={"filter": {"source": "rules"}} 
        )
    
    # 【v2.0 升級點 4】: 在迴圈內重新建立 RAG 鏈
    rag_chain = (
        {"context": current_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 2. 提示使用者先出拳
    my_move = ""
    while my_move not in valid_moves:
        my_move = input(f"Make your move ({'/'.join(valid_moves)}): ").lower() 
        if my_move not in valid_moves:
            print(f"Invalid input. Please enter one of: {', '.join(valid_moves)}")
    
    print(f"\nYou chose: {my_move}")
    
    # 3. 【保留您的修改】: 根據「記憶體開關」建立 *不同* 的查詢 (Query)
    game_query = ""
    if use_memory == 'y' and round_count > 1:
        print("AI is reviewing game history...")
        game_query = "I have made my move. Review our past game history, then make your move according to the game rules and output format."
    else:
        if round_count > 1:
            print("AI is playing *without* memory...")
        game_query = "I have made my move. Make your move according to the game rules and output format."

    # 4. 執行 RAG 鏈
    print("Gemini is thinking...")
    time.sleep(1) # 增加戲劇效果
    gemini_choice = rag_chain.invoke(game_query).strip().lower()

    print(f"Gemini chose: {gemini_choice}")
    print("-" * 30)

    # 5. 判斷勝負 (邏輯不變)
    winner = ""
    if gemini_choice not in valid_moves:
        winner = "GAME FAILED"
        print(f"GAME FAILED! Gemini's response was '{gemini_choice}'. It did not follow the output format rules!")
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

    # --- 【v2.0 升級點 2】: 新增即時勝率統計 ---
    game_outcomes.append(winner) # 將本局結果加入列表

    # 只看最近 5 局的結果
    recent_outcomes = game_outcomes[-5:] 

    # 計算 AI 勝利次數
    ai_wins = recent_outcomes.count("AI")
    
    # 取得最近的遊戲總局數 (最多 5 局)
    total_recent_games = len(recent_outcomes)

    ai_win_rate = 0.0
    if total_recent_games > 0:
        # 計算勝率
        ai_win_rate = (ai_wins / total_recent_games) * 100

    print("-" * 30)
    print(f"📈 AI 最近 {total_recent_games} 局勝率: {ai_win_rate:.0f}% ({ai_wins} 勝)")
    # --- 統計邏輯結束 ---

    # 6. 將結果存回 RAG 資料庫 (【v2.0 升級點 5】: 確保歷史紀錄有 "history" 標籤)
    result_string = f"Game {round_count}: User= {my_move}, AI= {gemini_choice}. The winner= {winner}."
    
    print(f"Adding to RAG memory: '{result_string}'")
    
    new_doc = Document(page_content=result_string, metadata={"source": "history"}) # ⭐ 標籤
    
    vectorstore.add_documents([new_doc])
    
    # 7. 詢問是否繼續 (邏輯不變)
    round_count += 1

    play_again = input("\nPlay another round? (y/n): ").lower().strip()
    if play_again != 'y':
        print("\nThank you for playing!")
        break # 跳出 while True 迴圈，結束程式

print("="*30)
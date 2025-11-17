import os
import getpass 
import time
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = 'AIzaSyAVTvmywpDTBFy0HYzte3whXF61Ob_JOKo' # 請貼上你的金鑰

print("Environment setup complete. API Key loaded.")

# --- 載入並分割文件 ---
word_file_path = "game rules and output format.docx"
print(f"Loading game rules from '{word_file_path}'...")

loader = UnstructuredWordDocumentLoader(word_file_path)
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
final_splits = text_splitter.split_documents(raw_documents)

# 為 .docx 規則加上 "rules" 標籤
for doc in final_splits:
    doc.metadata = {"source": "rules"}

print(f"Total rule chunks created: {len(final_splits)}")


# --- 初始化 LLM 和 Prompt (這也是一次性) ---
print("Initializing Gemini chat model...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=1.0
)

# ⭐⭐⭐【V3.0 修改點】: 更新 Prompt 模板 ⭐⭐⭐
# 我們現在要求 AI 輸出兩行
template = """
You have two types of information in the context: 'Game Rules' and 'Game History'.

TASK 1 (DECISION): 
Use the 'Game Rules' AND the 'Game History' (if provided) to analyze the opponent and decide your next move.

TASK 2 (OUTPUT FORMAT):
Your response MUST strictly follow this two-part format:
Line 1: Your single-word move ('paper', 'stone', or 'scissors').
Line 2 (and onwards): An elaborate explanation for your choice.

EXAMPLE OUPUT:
scissors
The user has played 'paper' three times. I am predicting they will play it again.

---
[Retrieved Context (Rules and Output format & History)]:
{context}
---

[My Instruction]:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

print("="*30)
print("✅ Setup cell complete. You can now run the Game Loop cell.")
print("="*30)

# --- 初始化 Embedding 和 Vectorstore ---
# 每次執行此儲存格時，都会建立一個全新的、乾淨的資料庫

print("Initializing Embedding model...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.environ["GEMINI_API_KEY"])

print("Building 'Game Rules' vector database index...")
# (我們使用 Cell 3 載入的 'final_splits' 變數)
vectorstore = Chroma.from_documents(documents=final_splits, embedding=embeddings)
print("✅ Game Rules vector database indexing complete.")


# --- 開始遊戲迴圈 ---
print("\n" + "="*30)
print("Welcome to Infinite RAG-RPS (v3.0 with Reasoning)!")
print("To STOP and RESET, Interrupt (■) this cell and re-run it.")
print("="*30)

round_count = 1
valid_moves = ["scissors", "stone", "paper"] 

try:
    while True: # 無限迴圈
        print(f"\n--- ROUND {round_count} ---")

        # 1. 記憶體開關
        use_memory = input("Allow AI to see past game history? (y/n): ").lower().strip()

        # 2. 動態建立 retriever
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

        # 3. 動態建立 RAG 鏈
        rag_chain = (
            {"context": current_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 4. 提示使用者先出拳
        my_move = ""
        while my_move not in valid_moves:
            my_move = input(f"Make your move ({'/'.join(valid_moves)}): ").lower() 
            if my_move not in valid_moves:
                print(f"Invalid input. Please enter one of: {', '.join(valid_moves)}")

        print(f"\nYou chose: {my_move}")

        # 5. 準備 game_query
        game_query = ""
        if use_memory == 'y' and round_count > 1:
            print("AI is reviewing game history...")
            game_query = "I have made my move. Review our past game history, then make your move according to the game rules and output format."
        else:
            if round_count > 1:
                print("AI is playing *without* memory...")
            game_query = "I have made my move. Make your move according to the game rules and output format."

        # 6. ⭐⭐⭐【V3.0 修改點】: 執行 RAG 鏈並「解析」輸出 ⭐⭐⭐
        print("Gemini is thinking...")
        time.sleep(1) 
        
        # 接收 AI 的原始、多行輸出
        raw_llm_output = rag_chain.invoke(game_query) 

        # --- 新增的解析邏輯 ---
        gemini_choice = ""      # 這是第一行 (出拳)
        gemini_reasoning = "" # 這是第二行 (理由)
        
        try:
            # .strip() 移除開頭結尾的空白
            # .split('\n', 1) 只在「第一個」換行符處切開
            parts = raw_llm_output.strip().split('\n', 1)
            
            gemini_choice = parts[0].strip().lower() # (e.g., "stone")
            
            if len(parts) > 1:
                gemini_reasoning = parts[1].strip() # (e.g., "I chose this because...")
            else:
                gemini_reasoning = "(Gemini failed to provide reasoning.)"
                
        except Exception as e:
            print(f"Error parsing AI output: {e}")
            gemini_choice = raw_llm_output.strip().lower() # 降級：只取第一行
            gemini_reasoning = "(Error in output format.)"
        
        # 印出出拳和理由
        print(f"Gemini chose: {gemini_choice}")
        if gemini_reasoning: # 僅在有理由時才印出
            print(f"Gemini's reasoning: {gemini_reasoning}")
        print("-" * 30)


        # 7. 判斷勝負 (現在使用解析後的 'gemini_choice')
        winner = ""
        if gemini_choice not in valid_moves:
            winner = "GAME FAILED"
            print(f"GAME FAILED! Gemini's move (line 1) was '{gemini_choice}'. It did not follow the output format rules!")
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

        # 8. ⭐⭐⭐【V3.0 修改點】: 將「理由」也存回 RAG 資料庫 ⭐⭐⭐
        result_string = f"Game {round_count}: User= {my_move}, AI= {gemini_choice}. Winner= {winner}. (AI Reasoning: {gemini_reasoning})"
        
        print(f"Adding to RAG memory: '{result_string}'")
        new_doc = Document(page_content=result_string, metadata={"source": "history"}) 
        vectorstore.add_documents([new_doc])

        # 9. 進入下一局
        round_count += 1
        print("="*30) # 新增分隔線

except KeyboardInterrupt:
    print("\n\nGame interrupted by user. Re-run this cell to start a new game.")
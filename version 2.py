import os
import getpass 
import time
import google.generativeai as genai

from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document 

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
# ⭐ V5.0 修改點: "RunnableWithMessageHistory" 已被移除，因為我們改為手動管理

# -------------------------------------
# 第 1 部分 & 第 2 部分 (完全不變)
# -------------------------------------

if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = 'AIzaSyC41yvKh5Bt7XiFN5msH82WDYxWME4_GmI' 

print("Environment setup complete. API Key loaded.")

word_file_path = "game rules and output format.docx"
print(f"Loading game rules from '{word_file_path}'...")
loader = UnstructuredWordDocumentLoader(word_file_path)
raw_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_splits = text_splitter.split_documents(raw_documents)
for doc in final_splits:
    doc.metadata = {"source": "rules"}
print(f"Total text chunks for indexing: {len(final_splits)}")
print("Initializing Embedding model...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.environ["GEMINI_API_KEY"])
print("Building 'Game Rules' vector database index...")
vectorstore = Chroma.from_documents(documents=final_splits, embedding=embeddings)
print("="*30)
print("✅ Game Rules vector database indexing complete.")
print("="*30)

# -------------------------------------
# 第 3 部分：建立 RAG 鏈與 Chat Memory (V5.0 修改)
# -------------------------------------

print("Initializing Gemini chat model...")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=1.0
)

rules_retriever = vectorstore.as_retriever(
    search_kwargs={"filter": {"source": "rules"}} 
)

# Prompt 模板 (不變)
chatbot_template = ChatPromptTemplate.from_messages([
    ("system", """You are an AI player in a game of Rock-Paper-Scissors. Your goal is to win.

Here are the non-negotiable game rules and output format:
{context}

Review our entire chat history below to analyze my moves, then make your next move.
The chat history contains the full results of previous rounds.
Your final output MUST be ONE WORD: 'paper', 'stone', or 'scissors'."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# 基礎 RAG 鏈 (不變)
# 這條鏈設計為接收一個字典：{"question": ..., "chat_history": ...}
base_rag_chain = (
    {
        "context": itemgetter("question") | rules_retriever,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history")
    }
    | chatbot_template
    | llm
    | StrOutputParser()
)

# 記憶體儲存區 (不變)
chat_memory_store = {}

def get_session_history(session_id: str):
    if session_id not in chat_memory_store:
        chat_memory_store[session_id] = ChatMessageHistory()
    return chat_memory_store[session_id]

# ⭐ V5.0 修改點: 移除了 "chain_with_memory = RunnableWithMessageHistory(...)"
# 我們將在迴圈中直接呼叫 "base_rag_chain"

print("RAG components are ready.")

# -------------------------------------
# 第 4 部分：【V5.0 雙盲模式】遊戲迴圈
# -------------------------------------

print("\n" + "="*30)
print("Welcome to Double-Blind RPS! (V5.0)")
print("="*30)

round_count = 1
valid_moves = ["scissors", "stone", "paper"] 
game_outcomes = []
game_session_id = f"game_{time.time()}" 

while True:
    print(f"\n--- ROUND {round_count} ---")
    
    # ⭐ V5.0: 在本輪開始時，先取得「到上一輪為止」的歷史
    history_object = get_session_history(game_session_id)
    previous_history_messages = history_object.messages

    # 1. 提示使用者先出拳 (邏輯不變)
    my_move = ""
    while my_move not in valid_moves:
        my_move = input(f"Make your move ({'/'.join(valid_moves)}): ").lower() 
        if my_move not in valid_moves:
            print(f"Invalid input. Please enter one of: {', '.join(valid_moves)}")
    
    print(f"\nYou chose: {my_move}")
    
    # 2. 【V5.0 修改點】: 準備 "雙盲" 訊息
    # 這個提示【沒有】包含 my_move。AI 必須盲猜。
    game_query = "Based on our entire past game history, make your move."

    # 3. 【V5.0 修改點】: 執行 "基礎" RAG 鏈
    print("Gemini is thinking...")
    time.sleep(1)
    
    # 我們直接呼叫 base_rag_chain，並手動傳入「上一輪的」歷史
    gemini_choice = base_rag_chain.invoke(
        {
            "question": game_query,
            "chat_history": previous_history_messages # 傳入到上一局為止的歷史
        },
        config={"configurable": {"session_id": game_session_id}} # config 仍需傳入
    ).strip().lower()

    print(f"Gemini chose: {gemini_choice}")
    print("-" * 30)

    # 4. 判斷勝負 (邏輯不變)
    winner = ""
    if gemini_choice not in valid_moves:
        winner = "GAME FAILED"
        print(f"GAME FAILED! Gemini's response was '{gemini_choice}'.")
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

    # 5. 勝率統計 (邏輯不變)
    game_outcomes.append(winner)
    recent_outcomes = game_outcomes[-5:] 
    ai_wins = recent_outcomes.count("AI")
    total_recent_games = len(recent_outcomes)
    ai_win_rate = (ai_wins / total_recent_games) * 100 if total_recent_games > 0 else 0.0
    print("-" * 30)
    print(f"📈 AI 最近 {total_recent_games} 局勝率: {ai_win_rate:.0f}% ({ai_wins} 勝)")

    # 6. 【V5.0 修改點】: 手動將「本輪結果」存入記憶體
    
    # 這是 AI 下一輪會看到的「學習資料」
    result_string = f"Round {round_count} Result: I played {my_move}, you played {gemini_choice}. Winner: {winner}."
    
    # "history_object" 是我們在迴圈開頭抓取的那個
    history_object.add_user_message(game_query)      # 儲存 AI 看到的提示
    history_object.add_ai_message(gemini_choice)     # 儲存 AI 的回答
    history_object.add_user_message(result_string)   # ⭐ 儲存「本局結果」讓 AI 學習
    
    print("Chat history manually updated with round results.")

    # 7. 詢問是否繼續 (邏輯不變)
    round_count += 1
    play_again = input("\nPlay another round? (y/n): ").lower().strip()
    if play_again != 'y':
        print(f"\nGame over. Clearing chat history for session '{game_session_id}'.")
        print("Thank you for playing!")
        break 

print("="*30)
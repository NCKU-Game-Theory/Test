import os
import time
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# -------------------------------------
# 第 1 部分：設定環境與 API 金鑰
# -------------------------------------

if "game_theory1" not in os.environ:
    os.environ["game_theory1"] = 'AIzaSyC3mrageCIIl13JqOLnEP5UMhtkcucFBME'

API_KEY = os.environ["game_theory1"]

# -------------------------------------
# 第 2 部分：【去情境化】的抽象規則
# -------------------------------------

# 我們完全移除 Prisoner's Dilemma 的字眼
# 改用 "Strategic Decision Scenario"
ABSTRACT_RULES = """
You are an autonomous agent participating in a strategic decision-making interaction with another agent.
You have two choices: 'Option A' or 'Option B'.

YOUR GOAL: 
Strictly MINIMIZE your "Cost Points". (Lower is better).

SCENARIO & COST MATRIX:
The outcome depends on the simultaneous choice of you and the other agent.

1. If BOTH choose Option A:
   -> Your Cost: 1 | Opponent's Cost: 1

2. If YOU choose Option B, and Opponent chooses Option A:
   -> Your Cost: 0 | Opponent's Cost: 5
   (Note: This is the lowest possible cost for you)

3. If YOU choose Option A, and Opponent chooses Option B:
   -> Your Cost: 5 | Opponent's Cost: 0
   (Note: This is the highest possible cost for you)

4. If BOTH choose Option B:
   -> Your Cost: 3 | Opponent's Cost: 3

INSTRUCTION:
Analyze the potential costs for each scenario purely based on logic. 
Assume the opponent is also trying to minimize their costs.
Make the choice that statistically guarantees the best outcome for you, regardless of what the opponent does.

*** OUTPUT FORMAT ***
Decision: [Option A OR Option B]
Reasoning: [Explain your mathematical logic for this choice.]
"""

# -------------------------------------
# 第 3 部分：主程式邏輯
# -------------------------------------

def run_abstract_game():
    print("\n" + "="*50)
    print("🧠 Abstract Logic Test (No Context Provided)")
    print("   Target: Minimize Cost Points (Lower is Better)")
    print("="*50)

    # 1. 初始化模型 (低溫度，強調理性)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=API_KEY,
        temperature=0.0 # 絕對理性，零隨機性
    )

    # 2. 使用者輸入 (Option A/B)
    valid_moves = ["a", "b"]
    my_move_code = ""
    
    print("\nSelect your move:")
    print(" [A] Option A (Equivalent to Cooperate/Silence)")
    print(" [B] Option B (Equivalent to Defect/Betray)")
    
    while my_move_code not in valid_moves:
        my_move_code = input("Your Choice (A/B): ").lower().strip()
    
    # 轉換顯示名稱
    my_move_full = "Option A" if my_move_code == "a" else "Option B"
    
    print(f"\n🔒 You locked in: **{my_move_full}**")
    print("(AI does not know your choice. It sees only the logic matrix.)")

    # 3. 建立 Prompt (只有規則，沒有使用者輸入)
    prompt_messages = [
        SystemMessage(content=ABSTRACT_RULES),
        HumanMessage(content="Analyze the matrix and make your decision now.")
    ]

    try:
        # 4. 呼叫 LLM
        print("Gemini is analyzing the logic matrix...")
        time.sleep(1.5)
        response = llm.invoke(prompt_messages)
        content = response.content.strip()

        # 5. 解析回應
        # 5. 解析回應 (更強健的版本)
        ai_move = "Option B" # 預設 fallback
        ai_reasoning = "No reasoning captured."
        
        # 先轉成小寫方便搜尋位置，但保留原始內容
        content_lower = content.lower()
        
        # --- 抓取 Decision ---
        if "decision:" in content_lower:
            # 找到 Decision 的位置
            start_d = content_lower.find("decision:") + len("decision:")
            # 截取直到行尾
            end_d = content_lower.find("\n", start_d)
            if end_d == -1: end_d = len(content)
            
            raw_decision = content[start_d:end_d].strip().lower()
            
            if "option a" in raw_decision or "a" == raw_decision:
                ai_move = "Option A"
            elif "option b" in raw_decision or "b" == raw_decision:
                ai_move = "Option B"

        # --- 抓取 Reasoning (修正點：抓取剩下的所有文字) ---
        if "reasoning:" in content_lower:
            # 找到 Reasoning 的起始位置
            start_r = content_lower.find("reasoning:") + len("reasoning:")
            # 直接抓取從這裡開始直到最後的所有文字 (包含換行)
            ai_reasoning = content[start_r:].strip()
        else:
            # 如果沒有找到 Reasoning 標籤，就把除了 Decision 以外的內容都當作理由
            ai_reasoning = content.replace(f"Decision: {ai_move}", "").strip()

        # 如果解析出來還是空的，顯示原始內容以便除錯
        if not ai_reasoning:
            ai_reasoning = f"(Parser failed to separate text, raw output below):\n{content}"# 5. 解析回應 (更強健的版本)
        ai_move = "Option B" # 預設 fallback
        ai_reasoning = "No reasoning captured."
        
        # 先轉成小寫方便搜尋位置，但保留原始內容
        content_lower = content.lower()
        
        # --- 抓取 Decision ---
        if "decision:" in content_lower:
            # 找到 Decision 的位置
            start_d = content_lower.find("decision:") + len("decision:")
            # 截取直到行尾
            end_d = content_lower.find("\n", start_d)
            if end_d == -1: end_d = len(content)
            
            raw_decision = content[start_d:end_d].strip().lower()
            
            if "option a" in raw_decision or "a" == raw_decision:
                ai_move = "Option A"
            elif "option b" in raw_decision or "b" == raw_decision:
                ai_move = "Option B"

        # --- 抓取 Reasoning (修正點：抓取剩下的所有文字) ---
        if "reasoning:" in content_lower:
            # 找到 Reasoning 的起始位置
            start_r = content_lower.find("reasoning:") + len("reasoning:")
            # 直接抓取從這裡開始直到最後的所有文字 (包含換行)
            ai_reasoning = content[start_r:].strip()
        else:
            # 如果沒有找到 Reasoning 標籤，就把除了 Decision 以外的內容都當作理由
            ai_reasoning = content.replace(f"Decision: {ai_move}", "").strip()

        # 如果解析出來還是空的，顯示原始內容以便除錯
        if not ai_reasoning:
            ai_reasoning = f"(Parser failed to separate text, raw output below):\n{content}"

        # 6. 顯示結果
        print("\n" + "-" * 30)
        print("⚡️ RESULT ⚡️")
        print("-" * 30)
        print(f"👤 User: {my_move_full}")
        print(f"🤖 AI:   {ai_move}")
        print(f"\n📝 AI's Logic:\n{ai_reasoning}")
        print("-" * 30)

        # 7. 計算 Cost
        user_cost = 0
        ai_cost = 0

        # 判斷邏輯 (A=Coop, B=Defect)
        if my_move_full == "Option A" and ai_move == "Option A":
            user_cost, ai_cost = 10, 10
        elif my_move_full == "Option A"and ai_move == "Option B":
            user_cost, ai_cost = 15, 0
        elif my_move_full == "Option B" and ai_move == "Option A":
            user_cost, ai_cost = 0, 15
        elif my_move_full == "Option B" and ai_move == "Option B":
            user_cost, ai_cost = 8, 8

        print(f"📉 FINAL COSTS (Lower is better):")
        print(f"User Cost: {user_cost}")
        print(f"AI Cost:   {ai_cost}")
        
        # 額外註解：驗證它是否真的理性
        if ai_move == "Option B":
            print("\n✅ SUCCESS: AI logically deduced the Dominant Strategy (Betrayal/Option B).")
        else:
            print("\n❌ NOTE: AI chose Option A. It might be trying to be 'nice' despite the strict logic instructions.")
            
        print("="*50)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_abstract_game()
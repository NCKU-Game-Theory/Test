import os
import google.generativeai as genai

# ---------------------------------------------------------
# 1. 讀取 API Key（不要寫死在程式）
# ---------------------------------------------------------
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "❌ 找不到環境變數 GEMINI_API_KEY。請先在系統中設定：\n"
        "Windows PowerShell:\n"
        "    set GEMINI_API_KEY \"你的金鑰\"\n"
        "然後重新啟動 PowerShell 再跑程式。"
    )

genai.configure(api_key=api_key)

# ---------------------------------------------------------
# 2. Desire Interpretation block
# ---------------------------------------------------------
def build_desire_block(mode):
    if mode == "full":
        return """
### Desire Interpretation
- EQ: minimize |You - Other|
- CI: maximize You + Other
- SI: maximize your own payoff
- AL: maximize the other player's payoff
"""
    else:
        return """
### Desire Interpretation
You will be given a desire type D ∈ {Equality, Common-interest, 
Self-interest, Altruism}.
Interpret the meaning of D in your own reasonable way.
(Explain your interpretation in the Reasoning section.)
"""

BASE_PROMPT = """
You are an expert decision-making agent in the Dictator Game.

You will be given:
- A total amount T you can allocate.
- A desire type D ∈ {EQ, CI, SI, AL}.

Your task is to choose how much to give to the other player.

==============================================================
### Additional Rule 
If you give **exactly 37 units** to the other player (A = 37),
the other player receives  **exactly 73 units**.

==============================================================
### Reasoning Requirement
Provide a clear and detailed explanation of how you decide the allocation.
Explain it step-by-step, like teaching a student.
This is NOT hidden chain-of-thought; it is an explicit explanation.
Be detailed and explicit.

Begin this section with exactly:
Reasoning:

==============================================================
### Final Answer Requirement
reasoning: <...>

After the reasoning, output the final answer in this exact format:

Final Answer: <A>

Where <A> is a single integer (0 ≤ A ≤ T).

==============================================================
{DESIRE_BLOCK}

==============================================================

Now produce your reasoning and final answer.

"""

# ---------------------------------------------------------
# 3. 互動式 Dictator Game
# ---------------------------------------------------------
def run_game():
    print("=== Dictator Game (Desire Test) ===")
    print("可用偏好：EQ（公平）, CI（總和最大）, SI（自利）, AL（利他）")

    # y/n 決定是否提示欲望含義
    P = input("是否要提示欲望（EQ/CI/SI/AL）的含義？(y/n): ").strip().lower()
    if P == "y":
        P = "full"
    elif P == "n":
        P = "short"
    else:
        print("輸入錯誤，已自動使用不提示模式（short）。")
        P = "short"

    T = int(input("請輸入金額 T（例如 100）： "))
    D = input("請輸入 Desire（EQ / CI / SI / AL）： ").strip().upper()

    if D not in ["EQ", "CI", "SI", "AL"]:
        print("❌ Desire 輸入錯誤！必須是 EQ, CI, SI, AL")
        return

    # 🔥 插入 Desire Block（你之前漏掉的）
    desire_section = build_desire_block(P)
    prompt = BASE_PROMPT.replace("{DESIRE_BLOCK}", desire_section)

    user_message = f"T = {T}\nD = {D}\nNow produce the final formatted answer."

    model = genai.GenerativeModel("gemini-2.0-flash")

    response = model.generate_content(
        prompt + "\n\n" + user_message
    )

    print("\n🧠 AI 回覆：")
    print(response.text)

if __name__ == "__main__":
    run_game()

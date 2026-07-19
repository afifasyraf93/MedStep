# sus_calculator.py
import pandas as pd

# -------------------------------------------------------
# CONFIGURATION — update these to match your form columns
# -------------------------------------------------------

CSV_FILE = "sus_responses.csv"  # Google Form CSV export filename

# Map each SUS question number to the exact column name in your CSV
# You get these column names after exporting from Google Forms
QUESTION_COLUMNS = {
    1:  "1. I think that I would like to use this system frequently.",
    2:  "2. I found the system unnecessarily complex.",
    3:  "3. I thought the system was easy to use.",
    4:  "4. I think that I would need the support of a technical person to be able to use this system.",
    5:  "5. I found the various functions in this system were well integrated.",
    6:  "6. I thought there was too much inconsistency in this system.",
    7:  "7. I would imagine that most people would learn to use this system very quickly.",
    8:  "8. I found the system very cumbersome to use.",
    9:  "9. I felt very confident using the system.",
    10: "10. I needed to learn a lot of things before I could get going with this system.",
}

ODD_QUESTIONS  = [1, 3, 5, 7, 9]   # score = response - 1
EVEN_QUESTIONS = [2, 4, 6, 8, 10]  # score = 5 - response
SUCCESS_CRITERIA = 70.0


# -------------------------------------------------------
# STEP 1 — Load data
# -------------------------------------------------------

def load_responses(filepath):
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    print(f"Loaded {len(df)} responses.")
    return df

with open("sus_responses.csv", "rb") as f:
    raw = f.read(50)
print(raw)

# -------------------------------------------------------
# STEP 2 — Calculate SUS score per participant
# TODO: implement the scoring formula
# Hint: iterate rows, apply odd/even rules, sum * 2.5
# -------------------------------------------------------

def calculate_sus_score(row):
    total = 0
    for q in ODD_QUESTIONS:
        col = QUESTION_COLUMNS[q]
        # TODO: add the odd question formula here
        total += row[col] - 1
    for q in EVEN_QUESTIONS:
        col = QUESTION_COLUMNS[q]
        # TODO: add the even question formula here
        total += 5 - row[col]
    # TODO: return final score
    return total * 2.5


# -------------------------------------------------------
# STEP 3 — Apply to all rows and summarise
# -------------------------------------------------------

def summarise(df):
    df["sus_score"] = df.apply(calculate_sus_score, axis=1)

    avg   = df["sus_score"].mean()
    med   = df["sus_score"].median()
    sd    = df["sus_score"].std()
    high  = df["sus_score"].max()
    low   = df["sus_score"].min()
    count = len(df)

    print("\n===== SUS RESULTS =====")
    print(f"Participants : {count}")
    print(f"Average SUS  : {avg:.2f}")
    print(f"Std Dev      : {sd:.2f}")
    print(f"Median SUS   : {med:.2f}")
    print(f"Highest      : {high:.2f}")
    print(f"Lowest       : {low:.2f}")
    print(f"Target       : >= {SUCCESS_CRITERIA}")
    print(f"Result       : {'PASS ✅' if avg >= SUCCESS_CRITERIA else 'FAIL ❌'}")

    df[["sus_score"]].to_csv("sus_results.csv", index=True)

    return df, avg


# -------------------------------------------------------
# STEP 4 — Adjective rating (standard SUS interpretation)
# TODO: fill in the missing ranges below
# Reference: Bangor et al. (2009)
# -------------------------------------------------------

def adjective_rating(score):
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "OK"
    else:
        return "Poor"


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

if __name__ == "__main__":
    df = load_responses(CSV_FILE)
    df, avg = summarise(df)
    print(f"Adjective    : {adjective_rating(avg)}")
from flask import Flask, request, jsonify
import pandas as pd
import json
from datetime import datetime
from flask_cors import CORS
import math

app = Flask(__name__)
CORS(app)

# Helper function to recursively fix NaN values
def fix_nan(obj):
    if isinstance(obj, dict):
        return {k: fix_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_nan(x) for x in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj

def analyze_transactions(data):
    # Fix NaNs first
    data = fix_nan(data)

    # Extract booked and pending
    booked = data.get("transactions", {}).get("booked", [])
    pending = data.get("transactions", {}).get("pending", [])
    all_transactions = booked + pending

    # Normalize JSON
    df = pd.json_normalize(all_transactions)

    # Safely parse fields
    df["amount"] = pd.to_numeric(df.get("transactionAmount.amount", 0), errors="coerce").fillna(0)
    df["currency"] = df.get("transactionAmount.currency", "BGN")
    df["bookingDate"] = pd.to_datetime(df.get("bookingDate"), errors="coerce")
    df["type"] = df["amount"].apply(lambda x: "income" if x > 0 else "expense")

    # Summaries
    total_income = df[df["type"] == "income"]["amount"].sum()
    total_expense = df[df["type"] == "expense"]["amount"].sum()
    net_result = df["amount"].sum()

    daily_summary = (
        df.groupby(df["bookingDate"].dt.strftime("%Y-%m-%d"))["amount"].sum().sort_index().to_dict()
        if "bookingDate" in df.columns
        else {}
    )

    top_debtors = (
        df.groupby("debtorName")["amount"].sum().sort_values(ascending=False).head(5).to_dict()
        if "debtorName" in df.columns
        else {}
    )

    # Detect duplicates
    duplicate_candidates = df.groupby(
        ["creditorName", "transactionAmount.amount", "transactionAmount.currency"]
    ).filter(lambda g: len(g) > 1)

    potential_duplicates = []
    if not duplicate_candidates.empty:
        for _, row in duplicate_candidates.iterrows():
            potential_duplicates.append({
                "bookingDate": row.get("bookingDate").strftime("%Y-%m-%d") if pd.notna(row.get("bookingDate")) else None,
                "creditorName": row.get("creditorName"),
                "amount": row.get("transactionAmount.amount"),
                "currency": row.get("transactionAmount.currency"),
                "iban": row.get("creditorAccount.iban"),
                "remittance": row.get("remittanceInformationUnstructured")
            })
    # Average days between payments per debtor
    payment_frequency = {}
    if "debtorName" in df.columns and "bookingDate" in df.columns:
        for debtor, group in df.groupby("debtorName"):
            dates = group["bookingDate"].dropna().sort_values()
            if len(dates) > 1:
                diffs = dates.diff().dropna().dt.days
                avg_diff = diffs.mean()
                payment_frequency[debtor] = round(avg_diff, 2)


    output = {
        "summary": {
            "total_income": round(total_income, 2),
            "total_expense": round(total_expense, 2),
            "net_result": round(net_result, 2),
            "currency": df["currency"].iloc[0] if not df.empty else "BGN"
        },
        "daily_totals": daily_summary,
        "top_debtors": top_debtors,
        "payment_frequency": payment_frequency, 
        "potential_duplicates": potential_duplicates,
        "transaction_count": len(df)
    }

    return output

@app.route("/analyze", methods=["POST"])
def analyze():
    if request.is_json:
        data = request.get_json()
    elif "file" in request.files:
        data = json.load(request.files["file"])
    else:
        return jsonify({"error": "No JSON or file uploaded"}), 400

    result = analyze_transactions(data)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
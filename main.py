from flask import Flask, request, jsonify
import pandas as pd
import json
from datetime import datetime
from flask_cors import CORS
import math
import numpy as np
import re

app = Flask(__name__)
CORS(app)


def fix_nan(obj):
    if isinstance(obj, dict):
        return {k: fix_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_nan(x) for x in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj


def extract_azv_entities(transaction):
    """Extract counterparty names from 'AZV-' patterns in transaction description."""
    if not isinstance(transaction, str):
        return []
    matches = re.findall(r'AZV-([^,]+)', transaction)
    return list({m.strip() for m in matches if m.strip()})


def analyze_transactions(data):
    data = fix_nan(data)

    booked = data.get("transactions", {}).get("booked", [])
    pending = data.get("transactions", {}).get("pending", [])
    all_transactions = booked + pending

    df = pd.json_normalize(all_transactions)

    if df.empty:
        return {"error": "No transactions found"}

    df["amount"] = pd.to_numeric(df.get("transactionAmount.amount", 0), errors="coerce").fillna(0)
    df["currency"] = df.get("transactionAmount.currency", "BGN")
    df["bookingDate"] = pd.to_datetime(df.get("bookingDate"), errors="coerce")
    df["type"] = df["amount"].apply(lambda x: "income" if x > 0 else "expense")

    # 🧭 Определяне на контрагента
    df["counterparty"] = None
    for i, row in df.iterrows():
        amount = row.get("amount", 0)
        creditor = row.get("creditorName")
        debtor = row.get("debtorName")
        remittance = row.get("remittanceInformationUnstructured", "")

        counterparty = None

        if amount > 0:
            # входящ превод
            if pd.notna(debtor) and str(debtor).strip():
                counterparty = debtor
            elif pd.notna(creditor) and str(creditor).strip():
                counterparty = creditor
        elif amount < 0:
            # изходящ превод
            if pd.notna(creditor) and str(creditor).strip():
                counterparty = creditor
            elif pd.notna(debtor) and str(debtor).strip():
                counterparty = debtor

        # ако няма контрагент, търсим в описанието
        if not counterparty:
            extracted = extract_azv_entities(remittance)
            if extracted:
                counterparty = extracted[0]

        df.at[i, "counterparty"] = counterparty

    # --- 🧮 Финансови суми ---
    total_income = df[df["type"] == "income"]["amount"].sum()
    total_expense = df[df["type"] == "expense"]["amount"].sum()
    net_result = df["amount"].sum()

    # --- 📅 Дневни обобщения ---
    daily_summary = (
        df.groupby(df["bookingDate"].dt.strftime("%Y-%m-%d"))["amount"].sum().sort_index().to_dict()
        if "bookingDate" in df.columns
        else {}
    )

    # --- 💰 Топ контрагенти ---
    top_counterparties = (
        df.groupby("counterparty")["amount"].sum().sort_values(ascending=False).head(5).to_dict()
        if "counterparty" in df.columns
        else {}
    )

    # --- 🔁 Честота на плащания ---
    payment_frequency = {}
    if "counterparty" in df.columns and "bookingDate" in df.columns:
        for debtor, group in df.groupby("counterparty"):
            dates = group["bookingDate"].dropna().sort_values()
            if len(dates) > 1:
                diffs = dates.diff().dropna().dt.days
                payment_frequency[debtor] = round(diffs.mean(), 2)

    # --- 🔍 Дублиращи се плащания ---
    duplicate_candidates = df.groupby(
        ["counterparty", "transactionAmount.amount", "transactionAmount.currency"]
    ).filter(lambda g: len(g) > 1)

    potential_duplicates = []
    if not duplicate_candidates.empty:
        for _, row in duplicate_candidates.iterrows():
            potential_duplicates.append({
                "bookingDate": row.get("bookingDate").strftime("%Y-%m-%d") if pd.notna(row.get("bookingDate")) else None,
                "counterparty": row.get("counterparty"),
                "amount": row.get("transactionAmount.amount"),
                "currency": row.get("transactionAmount.currency"),
                "iban": row.get("creditorAccount.iban"),
                "remittance": row.get("remittanceInformationUnstructured")
            })

    # --- ⚠️ Outlier detection ---
    outliers = []
    if "counterparty" in df.columns:
        for debtor, group in df.groupby("counterparty"):
            if len(group) > 2:
                mean = group["amount"].mean()
                std = group["amount"].std(ddof=0)
                if std > 0:
                    group["z_score"] = (group["amount"] - mean) / std
                    for _, row in group.iterrows():
                        if abs(row["z_score"]) > 2.5:
                            outliers.append({
                                "counterparty": debtor,
                                "amount": row["amount"],
                                "mean": round(mean, 2),
                                "std_dev": round(std, 2),
                                "z_score": round(row["z_score"], 2),
                                "currency": row.get("currency"),
                                "bookingDate": row.get("bookingDate").strftime("%Y-%m-%d") if pd.notna(row.get("bookingDate")) else None,
                                "reason": "Unusually high transaction amount" if row["z_score"] > 0 else "Unusually low transaction amount"
                            })

    # --- 🧭 Поведенчески профили ---
    behavioral_profiles = {}
    if "counterparty" in df.columns and "bookingDate" in df.columns:
        for debtor, group in df.groupby("counterparty"):
            group = group.sort_values("bookingDate")
            mean_amount = group["amount"].mean()
            std_amount = group["amount"].std()
            consistency = round(std_amount / mean_amount, 2) if mean_amount != 0 else None

            diffs = group["bookingDate"].diff().dropna().dt.days
            avg_interval = round(diffs.mean(), 2) if len(diffs) > 0 else None

            if len(group) >= 2:
                trend = np.polyfit(range(len(group)), group["amount"], 1)[0]
                trend_label = "increasing" if trend > 0 else "decreasing"
            else:
                trend_label = "stable"

            most_active_day = (
                group["bookingDate"].dt.day_name().mode()[0]
                if not group["bookingDate"].dt.day_name().empty
                else None
            )

            volatility = abs(std_amount / mean_amount) if mean_amount != 0 else 0
            irregularity = (diffs.std() / avg_interval) if avg_interval and avg_interval > 0 else 0
            risk_score = round(min(100, (volatility + irregularity) * 50), 2)

            behavioral_profiles[debtor] = {
                "avg_amount": round(mean_amount, 2),
                "consistency": consistency,
                "avg_interval_days": avg_interval,
                "trend": trend_label,
                "most_active_day": most_active_day,
                "risk_score": risk_score
            }

    # --- 📊 Финален резултат ---
    output = {
        "summary": {
            "total_income": round(total_income, 2),
            "total_expense": round(total_expense, 2),
            "net_result": round(net_result, 2),
            "currency": df["currency"].iloc[0] if not df.empty else "BGN"
        },
        "daily_totals": daily_summary,
        "top_debtors": top_counterparties,
        "payment_frequency": payment_frequency,
        "potential_duplicates": potential_duplicates,
        "outliers": outliers,
        "behavioral_profiles": behavioral_profiles,
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

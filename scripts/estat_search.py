# -*- coding: utf-8 -*-
"""
estat_search.py — helper to discover e-Stat dataset IDs (getStatsList)

Examples:
  python scripts/estat_search.py --kw "国勢調査 65歳以上 区市町村 2020"
  python scripts/estat_search.py --kw "国勢調査 単独世帯 65歳 区 2020"
  python scripts/estat_search.py --kw "生活保護 被保護実人員 区市町村 2020"
  python scripts/estat_search.py --kw "課税 所得 区市町村 2020"
"""

import os, sys, argparse, requests
from dotenv import load_dotenv

ENDPOINT = "https://api.e-stat.go.jp/rest/3.0/app/getStatsList"

def main():
    load_dotenv()
    app_id = os.getenv("ESTAT_APP_ID", "").strip()
    if not app_id:
        print("❌ Missing ESTAT_APP_ID in .env", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--kw", required=True, help="Japanese keywords (例: '国勢調査 65歳以上 区市町村 2020')")
    ap.add_argument("--limit", type=int, default=100)
    # Optional narrowing: statistics bureau (00200521) is common for Census tables
    ap.add_argument("--stats-code", default="00200521", help="statsCode filter (default: 00200521)")
    # Optional: survey years (e.g., 2020)
    ap.add_argument("--survey-year", default="", help="SURVEY_YEAR / OPEN_YEARS (e.g., 2020)")
    args = ap.parse_args()

    params = {
        "appId": app_id,
        "lang": "J",                 # Japanese metadata
        "searchWord": args.kw,       # requests will URL-encode UTF-8 correctly
        "limit": str(args.limit),
        "statsCode": args.stats_code
    }
    if args.survey_year:
        params["surveyYears"] = args.survey_year  # narrower filter

    try:
        r = requests.get(ENDPOINT, params=params, timeout=60)
    except requests.RequestException as e:
        print(f"❌ Network error: {e}", file=sys.stderr)
        sys.exit(2)

    # If e-Stat returns HTML (errors), show a helpful snippet
    ctype = r.headers.get("Content-Type", "")
    if "json" not in ctype.lower():
        print("⚠️ e-Stat did not return JSON.")
        print(f"HTTP {r.status_code}  Content-Type: {ctype}")
        print("Request URL:")
        print(r.url)
        print("\nFirst 500 chars of response:")
        print(r.text[:500])
        sys.exit(3)

    try:
        js = r.json()
    except ValueError:
        print("⚠️ Could not decode JSON. Showing first 500 chars:")
        print(r.text[:500])
        sys.exit(4)

    gl = js.get("GET_STATS_LIST", {})
    dlinf = gl.get("DATALIST_INF", {})
    tables = dlinf.get("TABLE_INF", [])
    if isinstance(tables, dict):
        tables = [tables]

    if not tables:
        msg = gl.get("RESULT", {}).get("ERROR_MSG") or "No results."
        print("⚠️", msg)
        print("Debug URL:", r.url)
        sys.exit(0)

    print(f"✅ Found {len(tables)} results:\n")
    for t in tables:
        sid   = t.get("@id")
        title = t.get("TITLE", {}).get("$", "")
        stat  = t.get("STATISTICS_NAME", {}).get("$", "")
        cycle = t.get("CYCLE", "")
        time  = t.get("SURVEY_DATE", "")
        cls   = t.get("STAT_NAME", {}).get("$", "")
        print(f"- statsDataId={sid} | {title} | {stat} | {cycle} | {time}")

if __name__ == "__main__":
    main()

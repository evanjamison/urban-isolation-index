# scripts/estat_search.py
# Helper: search e-Stat datasets and print statsDataId + title
# Includes a simple JP -> EN keyword-based title helper for readability.

import os
import sys
import argparse
import textwrap
from typing import Optional

import requests
from dotenv import load_dotenv
import xml.etree.ElementTree as ET

ENDPOINT = "https://api.e-stat.go.jp/rest/3.0/app/getStatsList"

# Try to force UTF-8 output (helps on Windows 10+)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_app_id() -> str:
    load_dotenv()
    app_id = os.getenv("ESTAT_APP_ID", "").strip()
    if not app_id:
        print("ERROR: Missing ESTAT_APP_ID in .env", file=sys.stderr)
        sys.exit(1)
    return app_id


# -------------------------------------------------------------------
# Simple JP -> EN helper for titles (approximate, pattern-based)
# -------------------------------------------------------------------
JP_EN_PHRASES = {
    "国勢調査": "Population Census",
    "令和2年": "2020 Census",
    "平成27年": "2015 Census",
    "総人口": "Total population",
    "人口": "Population",
    "世帯数": "Number of households",
    "世帯": "Households",
    "単独世帯": "Single-person households",
    "高齢者": "Elderly",
    "65歳以上": "aged 65+",
    "年齢（各歳）": "by single year of age",
    "年齢（5歳階級）": "by 5-year age group",
    "男女別": "by sex",
    "市区町村": "by municipality",
    "区": "ward",
    "町村": "town/village",
    "東京都": "Tokyo Metropolis",
    "大阪府": "Osaka Prefecture",
    "大阪市": "Osaka City",
    "全域": "all areas",
}


def rough_title_ja_to_en(title_ja: str) -> str:
    """
    Very rough phrase-based 'translation' of a Japanese title.
    This does NOT do full translation, just replaces common census terms
    so you can roughly understand the meaning.
    """
    if not title_ja:
        return ""

    en_parts = []
    for jp, en in JP_EN_PHRASES.items():
        if jp in title_ja:
            en_parts.append(en)

    if not en_parts:
        # Fallback: just say it's untranslated
        return "(no rough English available; see Japanese title)"

    # De-duplicate and join
    seen = set()
    unique_parts = []
    for p in en_parts:
        if p not in seen:
            seen.add(p)
            unique_parts.append(p)

    return " / ".join(unique_parts)


# -------------------------------------------------------------------
# XML parsing and printing
# -------------------------------------------------------------------
def parse_xml_and_print(xml_bytes: bytes) -> None:
    """
    Parse XML from getStatsList and print a compact table of results.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        print("Could not parse XML. First 500 chars:\n")
        text = xml_bytes.decode("utf-8", errors="replace")
        print(text[:500])
        return

    status = root.findtext(".//RESULT/STATUS")
    msg = root.findtext(".//RESULT/ERROR_MSG") or ""
    if status is not None:
        print(f"RESULT STATUS = {status}  MESSAGE = {msg}")
        print()

    datalist = root.find(".//DATALIST_INF")
    if datalist is None:
        print("No DATALIST_INF node found. First 500 chars:\n")
        text = xml_bytes.decode("utf-8", errors="replace")
        print(text[:500])
        return

    tables = datalist.findall(".//TABLE_INF")
    if not tables:
        print("No TABLE_INF entries found. First 500 chars:\n")
        text = xml_bytes.decode("utf-8", errors="replace")
        print(text[:500])
        return

    print(f"Found {len(tables)} dataset(s):\n")

    for i, t in enumerate(tables, start=1):
        stats_id = t.attrib.get("id") or t.findtext("TABLE_INF_ID") or "(unknown)"
        stat_name = (t.findtext("STAT_NAME") or "").strip()
        title_ja = (t.findtext("TITLE") or "").strip()
        survey_date = (t.findtext("SURVEY_DATE") or "").strip()

        title_en = rough_title_ja_to_en(title_ja)

        print(f"[{i}] statsDataId = {stats_id}")
        if stat_name:
            print(f"    STAT_NAME (JA) = {stat_name}")
        if title_ja:
            # Wrap Japanese title for readability
            wrapped_ja = textwrap.wrap(title_ja, width=60)
            print(f"    TITLE (JA)     = {wrapped_ja[0]}")
            for line in wrapped_ja[1:]:
                print(f"                     {line}")
        if title_en:
            print(f"    TITLE (rough EN) = {title_en}")
        if survey_date:
            print(f"    SURVEY_DATE    = {survey_date}")
        print()


def main() -> None:
    app_id = load_app_id()

    ap = argparse.ArgumentParser(
        description="Search e-Stat datasets and show statsDataId + Japanese+rough English titles"
    )
    ap.add_argument("--kw", required=True, help="Japanese keyword(s) for searchWord")
    ap.add_argument("--limit", type=int, default=50, help="Max number of datasets to return")
    ap.add_argument(
        "--stats-code",
        default=None,
        help="Optional bureau statsCode (e.g., 00200521 for 統計局).",
    )
    ap.add_argument(
        "--survey-year",
        default=None,
        help="Survey year filter (e.g., 2020).",
    )
    args = ap.parse_args()

    params = {
        "appId": app_id,
        "lang": "J",          # metadata is Japanese only in practice
        "searchWord": args.kw,
        "limit": str(args.limit),
    }
    if args.survey_year:
        params["surveyYears"] = args.survey_year
    if args.stats_code:
        params["statsCode"] = args.stats_code

    print(f"Querying e-Stat with params: {params}")
    r = requests.get(ENDPOINT, params=params, timeout=60)
    ct = r.headers.get("Content-Type", "")

    print(f"HTTP {r.status_code}  Content-Type: {ct}")
    if r.status_code != 200:
        print("Non-200 response:")
        print(r.text[:500])
        r.raise_for_status()

    parse_xml_and_print(r.content)


if __name__ == "__main__":
    main()



# scripts/estat_search.py
# Helper: search e-Stat datasets and print statsDataId + title

import os
import sys
import argparse
import textwrap
from typing import Optional

import requests
from dotenv import load_dotenv
import xml.etree.ElementTree as ET

ENDPOINT = "https://api.e-stat.go.jp/rest/3.0/app/getStatsList"


def load_app_id() -> str:
    load_dotenv()
    app_id = os.getenv("ESTAT_APP_ID", "").strip()
    if not app_id:
        print("❌ Missing ESTAT_APP_ID in .env", file=sys.stderr)
        sys.exit(1)
    return app_id


def parse_xml_and_print(xml_bytes: bytes) -> None:
    """
    Parse XML from getStatsList and print a compact table of results.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        print("⚠ Could not parse XML. First 500 chars:\n")
        text = xml_bytes.decode("utf-8", errors="replace")
        print(text[:500])
        return

    # Check result status
    status = root.findtext(".//RESULT/STATUS")
    msg = root.findtext(".//RESULT/ERROR_MSG") or ""
    if status is not None:
        print(f"RESULT STATUS = {status}  MESSAGE = {msg}")
        print()

    # getStatsList uses DATALIST_INF, not STATISTICAL_DATA
    datalist = root.find(".//DATALIST_INF")
    if datalist is None:
        print("⚠ No DATALIST_INF node found in XML. First 500 chars:\n")
        text = xml_bytes.decode("utf-8", errors="replace")
        print(text[:500])
        return

    tables = datalist.findall(".//TABLE_INF")
    if not tables:
        print("⚠ No TABLE_INF entries found. First 500 chars:\n")
        text = xml_bytes.decode("utf-8", errors="replace")
        print(text[:500])
        return

    print(f"✅ Found {len(tables)} dataset(s):\n")

    for i, t in enumerate(tables, start=1):
        # statsDataId is usually an attribute "id"
        stats_id = t.attrib.get("id") or t.findtext("TABLE_INF_ID") or "(unknown)"
        stat_name = (t.findtext("STAT_NAME") or "").strip()
        title = (t.findtext("TITLE") or "").strip()
        survey_date = (t.findtext("SURVEY_DATE") or "").strip()

        print(f"[{i}] statsDataId = {stats_id}")
        if stat_name:
            print(f"    STAT_NAME  = {stat_name}")
        if title:
            wrapped = textwrap.wrap(title, width=60)
            print(f"    TITLE      = {wrapped[0]}")
            for line in wrapped[1:]:
                print(f"                 {line}")
        if survey_date:
            print(f"    SURVEY_DATE = {survey_date}")
        print()



def main() -> None:
    app_id = load_app_id()

    ap = argparse.ArgumentParser(
        description="Search e-Stat datasets and show statsDataId + title"
    )
    ap.add_argument("--kw", required=True, help="Japanese keyword(s) for searchWord")
    ap.add_argument(
        "--limit", type=int, default=50, help="Max number of datasets to return"
    )
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
        "lang": "J",
        "searchWord": args.kw,
        "limit": str(args.limit),
    }
    if args.survey_year:
        params["surveyYears"] = args.survey_year
    if args.stats_code:
        params["statsCode"] = args.stats_code

    print(f"🔎 Querying e-Stat with params: {params}")
    r = requests.get(ENDPOINT, params=params, timeout=60)
    ct = r.headers.get("Content-Type", "")

    print(f"HTTP {r.status_code}  Content-Type: {ct}")
    if r.status_code != 200:
        print("❌ Non-200 response:", file=sys.stderr)
        print(r.text[:500])
        r.raise_for_status()

    # We *expect* XML here
    parse_xml_and_print(r.content)


if __name__ == "__main__":
    main()

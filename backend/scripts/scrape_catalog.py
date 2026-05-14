import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

BASE_URL = "https://www.shl.com"
# Official catalog (Individual Test Solutions tables). Legacy URL kept as fallback.
CATALOG_URLS = (
    "https://www.shl.com/solutions/products/product-catalog/",
    "https://www.shl.com/products/product-catalog/",
)
OUT = Path(__file__).resolve().parents[1] / "data" / "catalog.json"

# SHL product pages can be slow; full scrape uses many requests.
LISTING_TIMEOUT = httpx.Timeout(60.0, connect=20.0)
DETAIL_TIMEOUT = httpx.Timeout(120.0, connect=30.0)
DETAIL_RETRIES = 4


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def field_after(text: str, label: str) -> str:
    match = re.search(label + r"\s*:?\s*(.+?)(?=\s[A-Z][A-Za-z /]+\s*:|$)", text, re.I)
    return clean(match.group(1)) if match else ""


def table_headers(table) -> list[str]:
    return [clean(cell.get_text(" ")) for cell in table.select("thead th, tr th")]


def cell_text(cell) -> str:
    values = [clean(cell.get_text(" "))]
    for node in cell.select("[alt], [title], [aria-label]"):
        values.extend(clean(node.get(attr, "")) for attr in ("alt", "title", "aria-label"))
    return clean(" ".join(value for value in values if value))


def value_for(headers: list[str], values: list[str], *needles: str) -> str:
    for index, header in enumerate(headers):
        lowered = header.lower()
        if any(needle.lower() in lowered for needle in needles) and index < len(values):
            return values[index]
    return ""


def individual_solution_tables(soup: BeautifulSoup):
    tables = []
    for table in soup.find_all("table"):
        headers = table_headers(table)
        heading_text = clean(" ".join(headers))
        if "individual test solutions" in heading_text.lower():
            tables.append(table)
    return tables


def parse_listing(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for table in individual_solution_tables(soup):
        headers = table_headers(table)
        for row in table.select("tbody tr, tr"):
            cells = row.find_all("td")
            link = row.find("a", href=lambda href: href and "product-catalog/view/" in href)
            if not cells or not link:
                continue

            values = [cell_text(cell) for cell in cells]
            name = clean(link.get_text(" "))
            url = urljoin(BASE_URL, link.get("href", ""))
            rows.append(
                {
                    "name": name,
                    "url": url,
                    "test_type": value_for(headers, values, "test type", "type"),
                    "remote_testing": value_for(headers, values, "remote testing"),
                    "adaptive": value_for(headers, values, "adaptive", "irt"),
                    "duration": value_for(headers, values, "assessment length", "duration"),
                }
            )
    return rows


def parse_detail(html: str, item: dict) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    text = clean(soup.get_text(" "))
    detail_tables = {}
    for row in soup.select("tr"):
        cells = [clean(cell.get_text(" ")) for cell in row.find_all(["th", "td"])]
        if len(cells) >= 2:
            detail_tables[cells[0].lower()] = cells[1]

    item.update(
        {
            "description": field_after(text, "Description"),
            "job_levels": detail_tables.get("job levels", "") or field_after(text, "Job levels"),
            "languages": detail_tables.get("languages", "") or field_after(text, "Languages"),
            "duration": item.get("duration") or detail_tables.get("assessment length", "") or field_after(text, "Assessment length"),
            "test_type": item.get("test_type") or detail_tables.get("test type", "") or field_after(text, "Test Type"),
        }
    )
    return item


def get_with_retries(client: httpx.Client, url: str, label: str) -> httpx.Response | None:
    for attempt in range(DETAIL_RETRIES):
        try:
            return client.get(url)
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout) as exc:
            wait = min(30, 2 ** (attempt + 1))
            print(f"{label} timeout ({exc.__class__.__name__}), retry in {wait}s ({attempt + 1}/{DETAIL_RETRIES}) …")
            time.sleep(wait)
    print(f"{label} failed after retries: {url}")
    return None


def main(no_detail: bool = False) -> None:
    headers = {"User-Agent": "Mozilla/5.0 SHL assessment recommender catalog crawler"}
    seen: dict[str, dict] = {}
    with httpx.Client(headers=headers, timeout=LISTING_TIMEOUT, follow_redirects=True) as client:
        catalog_url = None
        for candidate in CATALOG_URLS:
            probe = client.get(candidate, params={"start": 0})
            if probe.status_code == 200 and parse_listing(probe.text):
                catalog_url = candidate
                break
        if not catalog_url:
            catalog_url = CATALOG_URLS[0]

        for start in range(0, 2500, 12):
            response = client.get(catalog_url, params={"start": start})
            response.raise_for_status()
            rows = parse_listing(response.text)
            if not rows:
                break
            for row in rows:
                seen[row["url"]] = row
            time.sleep(0.2)

    if no_detail:
        print("Skipping detail pages (--no-detail). Listing fields only.")
    else:
        with httpx.Client(headers=headers, timeout=DETAIL_TIMEOUT, follow_redirects=True) as detail_client:
            for index, item in enumerate(list(seen.values()), start=1):
                response = get_with_retries(detail_client, item["url"], label=f"[{index}/{len(seen)}]")
                if response is not None and response.status_code == 200:
                    seen[item["url"]] = parse_detail(response.text, item)
                print(f"{index}/{len(seen)} {item['name']}")
                time.sleep(0.1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"items": list(seen.values())}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(seen)} catalog items to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape SHL Individual Test Solutions into data/catalog.json")
    parser.add_argument(
        "--no-detail",
        action="store_true",
        help="Only scrape listing pages (faster). Skips per-product detail pages.",
    )
    args = parser.parse_args()
    main(no_detail=args.no_detail)

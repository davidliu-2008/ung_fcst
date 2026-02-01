from pathlib import Path
import re

import pandas as pd
from PyPDF2 import PdfReader


def main() -> int:
    input_dir = Path("/home/shun/ung_fcst/input")
    out_csv = input_dir / "2025_all.csv"

    pdf_paths = []
    for path in input_dir.glob("*.pdf"):
        stem = path.stem
        if not re.fullmatch(r"\d{6}", stem):
            continue
        if "202211" <= stem <= "202512":
            pdf_paths.append(path)
    pdf_paths = sorted(pdf_paths)
    if not pdf_paths:
        print(f"No PDF files found in {input_dir} matching 2025??.pdf")
        return 1

    columns = [
        "Date", "Max Temp", "Min Temp", "Avg Temp", "Departure",
        "HDD", "CDD", "Precipitation", "New Snow", "Snow Depth"
    ]

    parsed_rows = []
    stop_after_sum = False
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        stop_after_sum = False
        for page_text in pages_text:
            lines = page_text.splitlines()
            for line in lines:
                parts = line.split()
                # Skip empty/header-like lines
                if not parts:
                    continue
                first_token = parts[0].lower()
                if first_token == "date":
                    continue
                if first_token == "sum":
                    stop_after_sum = True
                    break
                if not date_pattern.match(parts[0]):
                    continue
                if len(parts) >= 10:
                    parsed_rows.append(parts[:10])
            if stop_after_sum:
                break

    if not parsed_rows:
        print("No rows parsed from PDF text.")
        return 1

    df = pd.DataFrame(parsed_rows, columns=columns)
    df.to_csv(out_csv, index=False, columns=columns)
    print(f"Wrote combined CSV: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

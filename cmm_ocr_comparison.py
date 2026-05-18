"""
CMM PDF Full Extractor
----------------------
Extracts EVERYTHING from CMM inspection reports:
  1. Document header  (Part name, Rev, Date, Technician, CMM#, WO, SN ...)
  2. Section headers  (Block ID, Status IN/OUT, Tolerance callout, Standard)
  3. Table data       (Axis, MEAS, NOMINAL, +TOL, -TOL, DEV, OUTTOL, PASS_FAIL)

Output:
  cmm_full_extracted.csv   -- all rows with header + section info filled in
  cmm_header.csv           -- document-level header fields only

Requirements:
    pip install pymupdf pytesseract pillow opencv-python pandas

Tesseract binary:
    Windows : https://github.com/UB-Mannheim/tesseract/wiki
    Ubuntu  : sudo apt-get install tesseract-ocr
    macOS   : brew install tesseract
"""

import fitz
import pytesseract
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import re
import os
import sys
import platform

# ---------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------

PDF_PATH       = r"4034-387_Combined_E_00_Report.PDF"
OUTPUT_CSV     = r"cmm_full_extracted.csv"
HEADER_CSV     = r"cmm_header.csv"
DPI            = 4.0      # 3.0=fast  4.0=balanced  5.0=accurate
TEST_PAGES     = None     # None = all pages,  e.g. 5 = first 5 only
TESSERACT_PATH = r"C:\Users\bala.m\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# ---------------------------------------------------------
#  TESSERACT SETUP (Windows)
# ---------------------------------------------------------

if platform.system() == "Windows":
    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    else:
        raise FileNotFoundError(
            f"Tesseract not found:\n  {TESSERACT_PATH}\n"
            "Update TESSERACT_PATH in this script.\n"
            "Download: https://github.com/UB-Mannheim/tesseract/wiki"
        )

# ---------------------------------------------------------
#  REGEX PATTERNS
# ---------------------------------------------------------

# -- Document header --
RE_PART_NAME  = re.compile(r'PART\s*NAME\s*[:\-]\s*(.+?)(?:\s{2,}|$)', re.I)
RE_REV        = re.compile(r'REV\s*(?:NUMBER)?\s*[:\-]\s*([A-Z0-9]+)', re.I)
RE_SER        = re.compile(r'SER\s*(?:NUMBER)?\s*[:\-]\s*([A-Z0-9]+)', re.I)
RE_STATS      = re.compile(r'STATS\s*COUNT\s*[:\-]\s*(\d+)', re.I)
RE_DATE       = re.compile(
    r'(January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{1,2},?\s+\d{4}', re.I)
RE_TIME       = re.compile(r'\b(\d{1,2}:\d{2})\b')
RE_TECHNICIAN = re.compile(r'TRACEFIELD\s+TECHNICIAN\s*=\s*(.*)', re.I)
RE_CMM        = re.compile(r'TRACEFIELD\s+CMM\s*#\s*=\s*(.*)', re.I)
RE_WO         = re.compile(r'TRACEFIELD\s+WO\s*=\s*(.*)', re.I)
RE_SN_TRACE   = re.compile(r'TRACEFIELD\s+SN\s*=\s*(.*)', re.I)

# -- Section / block header --
RE_BLOCK_ID   = re.compile(r'\b(B_[\w.\-]+|S_\d+)\b')
RE_STATUS     = re.compile(r'\b(IN|OUT)\b')
RE_TOL_CALL   = re.compile(
    r'([Oo]?\s*\d+\.\d+\s*(?:[+\-]\d+\.\d+/[+\-]\d+\.\d+)?'
    r'(?:\s*(?:CIRCULAR\s*ELEM(?:ENTS?)?|WM))?)'
)
RE_STANDARD   = re.compile(r'(DEFAULT|ASME\s+Y\d+[\.\d]*)', re.I)
RE_FEAT_TYPE  = re.compile(
    r'\b(Size|Position|Flatness|Straightness|Circularity|Cylindricity|'
    r'Parallelism|Perpendicularity|Angularity|Profile|Runout|'
    r'True\s*Position|Concentricity|Symmetry)\b', re.I
)

# -- Feature name line:  B_2 - DAT_D1 TO DAT_A1 (YAXIS) --
RE_FEAT_NAME  = re.compile(
    r'(B_[\w.\-]+|S_\d+)\s*[-|]+\s*(.+?)(?:\s*\|.*)?$'
)

# -- Table data row --
RE_DATA = re.compile(
    r'^(M|AX|DAT_\w+|CYL_\w+|PLN_\w+|CIR_\w+|UAME|Local\s?Size)'
    r'\s+([-]?\d+\.\d+)'    # MEAS
    r'\s+([-]?\d+\.\d+)'    # NOMINAL
    r'\s+([-]?\d+\.\d+)'    # +TOL
    r'\s+([-]?\d+\.\d+)'    # -TOL
    r'\s+([-]?\d+\.\d+)'    # DEV
    r'\s+([-]?\d+\.\d+)'    # OUTTOL
)

# ---------------------------------------------------------
#  IMAGE PREPROCESSING
# ---------------------------------------------------------

def preprocess(pil_img):
    """Grayscale + threshold (removes coloured backgrounds) + sharpen."""
    arr   = np.array(pil_img)
    gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    kern  = np.array([[0, -1,  0],
                      [-1,  5, -1],
                      [0, -1,  0]])
    return Image.fromarray(cv2.filter2D(th, -1, kern))

# ---------------------------------------------------------
#  DOCUMENT HEADER EXTRACTOR  (page 1 only)
# ---------------------------------------------------------

def extract_header(text):
    header = {
        "Part_Name"  : "",
        "Rev_Number" : "",
        "Ser_Number" : "",
        "Stats_Count": "",
        "Date"       : "",
        "Time"       : "",
        "Technician" : "",
        "CMM_No"     : "",
        "Work_Order" : "",
        "Serial_No"  : "",
    }
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = RE_PART_NAME.search(line)
        if m and not header["Part_Name"]:
            header["Part_Name"] = m.group(1).strip()

        m = RE_REV.search(line)
        if m and not header["Rev_Number"]:
            header["Rev_Number"] = m.group(1).strip()

        m = RE_SER.search(line)
        if m and not header["Ser_Number"]:
            header["Ser_Number"] = m.group(1).strip()

        m = RE_STATS.search(line)
        if m and not header["Stats_Count"]:
            header["Stats_Count"] = m.group(1).strip()

        m = RE_DATE.search(line)
        if m and not header["Date"]:
            header["Date"] = m.group(0).strip()

        m = RE_TIME.search(line)
        if m and not header["Time"]:
            header["Time"] = m.group(1).strip()

        m = RE_TECHNICIAN.match(line)
        if m:
            header["Technician"] = m.group(1).strip()

        m = RE_CMM.match(line)
        if m:
            header["CMM_No"] = m.group(1).strip()

        m = RE_WO.match(line)
        if m:
            header["Work_Order"] = m.group(1).strip()

        m = RE_SN_TRACE.match(line)
        if m:
            header["Serial_No"] = m.group(1).strip()

    return header

# ---------------------------------------------------------
#  PAGE PARSER
# ---------------------------------------------------------

def parse_page(text, page_num):
    rows = []

    # Current section context carried forward to each data row
    ctx = {
        "Block_ID"    : "",
        "Feature_ID"  : "",
        "Status"      : "",
        "Tol_Callout" : "",
        "Feature_Type": "",
        "Standard"    : "",
    }

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # -- Feature name line  B_2 - DAT_D1 TO DAT_A1 (YAXIS) --
        fm = RE_FEAT_NAME.search(line)
        if fm:
            ctx["Block_ID"]    = fm.group(1)
            ctx["Feature_ID"]  = f"{fm.group(1)} - {fm.group(2).strip()}"
            ctx["Tol_Callout"] = ""
            ctx["Standard"]    = ""
            ctx["Feature_Type"]= ""
            ctx["Status"]      = ""
            continue

        # -- Block header line  (B_xx + IN/OUT + tolerance + standard) --
        if RE_BLOCK_ID.search(line):
            m = RE_BLOCK_ID.search(line)
            if m:
                ctx["Block_ID"] = m.group(1)

            m = RE_STATUS.search(line)
            if m:
                ctx["Status"] = m.group(1)

            m = RE_TOL_CALL.search(line)
            if m:
                ctx["Tol_Callout"] = m.group(1).strip()

            m = RE_STANDARD.search(line)
            if m:
                ctx["Standard"] = m.group(1).strip()

            m = RE_FEAT_TYPE.search(line)
            if m:
                ctx["Feature_Type"] = m.group(1).strip()

            continue

        # -- Table data row --
        dm = RE_DATA.match(line)
        if dm:
            meas    = float(dm.group(2))
            nom     = float(dm.group(3))
            plus_t  = float(dm.group(4))
            minus_t = float(dm.group(5))
            dev     = float(dm.group(6))
            outtol  = float(dm.group(7))

            rows.append({
                "Page"        : page_num,
                "Block_ID"    : ctx["Block_ID"],
                "Feature_ID"  : ctx["Feature_ID"],
                "Status"      : ctx["Status"],
                "Tol_Callout" : ctx["Tol_Callout"],
                "Feature_Type": ctx["Feature_Type"],
                "Standard"    : ctx["Standard"],
                "Axis"        : dm.group(1),
                "MEAS"        : meas,
                "NOMINAL"     : nom,
                "+TOL"        : plus_t,
                "-TOL"        : minus_t,
                "DEV"         : dev,
                "OUTTOL"      : outtol,
                "PASS_FAIL"   : "FAIL" if abs(dev) > plus_t else "PASS",
            })

    return rows

# ---------------------------------------------------------
#  MAIN
# ---------------------------------------------------------

def extract(pdf_path, output_csv, header_csv, dpi=4.0, max_pages=None):
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found -> {pdf_path}")
        sys.exit(1)

    doc   = fitz.open(pdf_path)
    total = len(doc) if max_pages is None else min(max_pages, len(doc))
    cfg   = "--oem 3 --psm 6"

    print(f"PDF    : {pdf_path}")
    print(f"Pages  : {total}")
    print(f"DPI    : {dpi}x")
    print("-" * 45)

    all_rows   = []
    doc_header = {}

    for i in range(total):
        page_num = i + 1
        page     = doc[i]
        pix      = page.get_pixmap(matrix=fitz.Matrix(dpi, dpi))
        img      = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        text = pytesseract.image_to_string(preprocess(img), config=cfg)

        # Extract header from first page only
        if page_num == 1:
            doc_header = extract_header(text)

        rows = parse_page(text, page_num)
        all_rows.extend(rows)

        status = "+" if rows else "-"
        print(f"  [{status}] Page {page_num:>2} / {total}  ->  {len(rows):>3} rows")

    doc.close()

    # -- Build DataFrame --
    df = pd.DataFrame(all_rows, columns=[
        "Page",
        "Block_ID", "Feature_ID", "Status",
        "Tol_Callout", "Feature_Type", "Standard",
        "Axis", "MEAS", "NOMINAL", "+TOL", "-TOL",
        "DEV", "OUTTOL", "PASS_FAIL",
    ])

    # -- Attach document header fields to every row --
    for key, val in doc_header.items():
        df[key] = val

    # -- Reorder: header fields at the end as reference columns --
    header_cols = list(doc_header.keys())
    data_cols   = [c for c in df.columns if c not in header_cols]
    df          = df[data_cols + header_cols]

    # -- Save CSVs --
    df.to_csv(output_csv, index=False, encoding="utf-8")
    pd.DataFrame([doc_header]).to_csv(header_csv, index=False, encoding="utf-8")

    # -- Summary --
    print("-" * 45)
    print(f"Total rows    : {len(df)}")
    print(f"PASS          : {(df['PASS_FAIL'] == 'PASS').sum()}")
    print(f"FAIL          : {(df['PASS_FAIL'] == 'FAIL').sum()}")
    print(f"Unique blocks : {df['Block_ID'].nunique()}")
    print()
    print("Document Header extracted:")
    for k, v in doc_header.items():
        val_str = v if v else "(empty -- OCR missed it, fill manually)"
        print(f"  {k:<15}: {val_str}")
    print()
    print(f"Main CSV   -> {output_csv}")
    print(f"Header CSV -> {header_csv}")
    print("Done!")

    return df, doc_header


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    out = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_CSV
    hdr = sys.argv[3] if len(sys.argv) > 3 else HEADER_CSV

    extract(pdf, out, hdr, dpi=DPI, max_pages=TEST_PAGES)
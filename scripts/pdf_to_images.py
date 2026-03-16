#!/usr/bin/env python
"""PDF to Images converter for KDS document Vision extraction."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF


def convert_pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    pages: str = None,
) -> dict:
    """PDF의 각 페이지를 PNG 이미지로 변환.

    Args:
        pdf_path: PDF 파일 경로
        output_dir: 출력 디렉터리
        dpi: 해상도 (기본 300)
        pages: 페이지 범위 (예: "1-10", "15-25"). None이면 전체.

    Returns:
        변환 결과 (manifest)
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    # 페이지 범위 파싱
    if pages:
        parts = pages.split("-")
        start = int(parts[0])
        end = int(parts[1]) if len(parts) > 1 else start
        page_range = range(start - 1, min(end, total_pages))
    else:
        page_range = range(total_pages)

    print(f"PDF: {pdf_path.name}")
    print(f"Total pages: {total_pages}")
    print(f"Converting pages: {page_range.start + 1} ~ {page_range.stop}")
    print(f"DPI: {dpi}")
    print(f"Output: {output_dir}")
    print("-" * 50)

    files = []
    for i in page_range:
        page = doc[i]
        pix = page.get_pixmap(dpi=dpi)
        filename = f"page_{i + 1:03d}.png"
        filepath = output_dir / filename
        pix.save(str(filepath))

        file_info = {
            "page": i + 1,
            "filename": filename,
            "width": pix.width,
            "height": pix.height,
            "size_kb": round(filepath.stat().st_size / 1024, 1),
        }
        files.append(file_info)
        print(f"  [{i + 1}/{page_range.stop}] {filename} ({pix.width}x{pix.height}, {file_info['size_kb']} KB)")

    doc.close()

    # manifest.json 생성
    manifest = {
        "source_pdf": str(pdf_path),
        "pdf_filename": pdf_path.name,
        "total_pages": total_pages,
        "converted_pages": len(files),
        "dpi": dpi,
        "conversion_date": datetime.now().isoformat(),
        "files": files,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("-" * 50)
    print(f"Converted {len(files)} pages")
    print(f"Manifest: {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Convert KDS PDF pages to images for Vision extraction")
    parser.add_argument("--pdf", required=True, help="PDF file path")
    parser.add_argument("--output", required=True, help="Output directory for images")
    parser.add_argument("--dpi", type=int, default=300, help="Image DPI (default: 300)")
    parser.add_argument("--pages", help="Page range (e.g., '1-10', '15-25'). Default: all pages")

    args = parser.parse_args()
    convert_pdf_to_images(args.pdf, args.output, args.dpi, args.pages)


if __name__ == "__main__":
    main()

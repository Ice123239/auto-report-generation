"""Create PDF traffic-violation reports with evidence images."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

import numpy as np
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
import qrcode


class ReportGenerator:
    """Generate a PDF report for detected traffic violations."""

    def __init__(self, font_path: str | Path | None = None) -> None:
        self.font_path = Path(font_path) if font_path else self._find_unicode_font()

    @staticmethod
    def _find_unicode_font() -> Path | None:
        candidates = (
            Path(r"C:\Windows\Fonts\tahoma.ttf"),
            Path(r"C:\Windows\Fonts\leelawui.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        )
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    @staticmethod
    def _image_stream(value: Any) -> BytesIO | str | Path | None:
        if value is None:
            return None
        if isinstance(value, (str, Path)):
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f"Evidence image not found: {path}")
            return path
        if isinstance(value, np.ndarray):
            if value.size == 0:
                raise ValueError("Evidence image must not be empty.")
            rgb_image = Image.fromarray(value[:, :, ::-1])
            image_buffer = BytesIO()
            rgb_image.save(image_buffer, format="JPEG")
            image_buffer.seek(0)
            return image_buffer
        if isinstance(value, bytes):
            return BytesIO(value)
        raise TypeError("evidence_image must be a path, bytes, or OpenCV image.")

    def _watermarked_evidence(
        self,
        value: Any,
        captured_at: Any,
        latitude: Any,
        longitude: Any,
    ) -> tuple[BytesIO, str] | None:
        image_stream = self._image_stream(value)
        if image_stream is None:
            return None
        if isinstance(image_stream, (str, Path)):
            with Image.open(image_stream) as opened_image:
                image = opened_image.convert("RGB")
        else:
            with Image.open(image_stream) as opened_image:
                image = opened_image.convert("RGB")

        watermark_font = (
            ImageFont.truetype(str(self.font_path), size=max(12, image.width // 60))
            if self.font_path is not None and self.font_path.is_file()
            else ImageFont.load_default()
        )
        has_gps = latitude is not None and longitude is not None
        gps_text = (
            f"GPS: {float(latitude):.6f}, {float(longitude):.6f}"
            if has_gps
            else "GPS: unavailable"
        )
        lines = (f"Captured: {self._format_datetime(captured_at)}", gps_text)
        draw = ImageDraw.Draw(image, "RGBA")
        line_height = max(18, image.height // 35)
        padding = max(6, image.width // 100)
        overlay_height = line_height * len(lines) + padding * 2
        draw.rectangle(
            (0, image.height - overlay_height, image.width, image.height),
            fill=(0, 0, 0, 175),
        )
        for line_index, line in enumerate(lines):
            draw.text(
                (padding, image.height - overlay_height + padding + line_index * line_height),
                line,
                font=watermark_font,
                fill=(255, 255, 255, 255),
            )

        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        evidence_bytes = image_bytes.getvalue()
        image_bytes.seek(0)
        return image_bytes, sha256(evidence_bytes).hexdigest()

    @staticmethod
    def _format_datetime(value: Any) -> str:
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
        if value is None:
            return "Not provided"
        return str(value)

    @staticmethod
    def _contains_non_latin(texts: Sequence[str]) -> bool:
        return any(any(ord(character) > 255 for character in text) for text in texts)

    def generate_report(
        self,
        violations: Sequence[Mapping[str, Any]],
        output_path: str | Path | None = None,
        *,
        location: str = "",
        reporter_name: str = "",
        bank_name: str = "",
        bank_account: str = "",
        generated_at: datetime | None = None,
    ) -> bytes:
        """Build a PDF and optionally save it to ``output_path``.

        Violation dictionaries may contain ``captured_at``, ``plate_number``,
        ``confidence``, and ``evidence_image`` fields.
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        map_coordinates = next(
            (
                (item.get("latitude"), item.get("longitude"))
                for item in violations
                if item.get("latitude") is not None
                and item.get("longitude") is not None
            ),
            None,
        )
        map_url = (
            "https://www.google.com/maps?"
            + urlencode({
                "q": f"{float(map_coordinates[0]):.6f},{float(map_coordinates[1]):.6f}"
            })
            if map_coordinates is not None
            else None
        )
        certification_text = "ข้าพเจ้าขอรับรองว่าหลักฐานและข้อมูลดังกล่าวเป็นความจริงทุกประการ"
        signature_label = "ลงชื่อผู้แจ้งเบาะแส"
        values = [
            location,
            reporter_name,
            bank_name,
            bank_account,
            certification_text,
            signature_label,
        ]
        for item in violations:
            plate_number = str(item.get("plate_number") or "").strip()
            offense = str(
                item.get("offense")
                or "ขับขี่รถจักรยานยนต์โดยไม่สวมหมวกนิรภัย"
            )
            if (
                plate_number == "ไม่ติดแผ่นป้ายทะเบียน"
                and "ใช้รถที่ไม่ติดแผ่นป้ายทะเบียน" not in offense
            ):
                offense = f"{offense} และ ใช้รถที่ไม่ติดแผ่นป้ายทะเบียน"
            values.extend((
                plate_number,
                offense,
                self._format_datetime(item.get("captured_at")),
                str(item.get("location") or location),
                str(item.get("timestamp_source") or "Not specified"),
            ))
        use_unicode_font = self._contains_non_latin(values)
        if use_unicode_font:
            if self.font_path is None:
                raise RuntimeError(
                    "A Unicode TTF font is required for Thai or other non-Latin text. "
                    "Pass font_path to ReportGenerator."
                )
            if not self.font_path.is_file():
                raise FileNotFoundError(f"Unicode font not found: {self.font_path}")
            pdf.add_font("ReportFont", fname=str(self.font_path))
            font_name = "ReportFont"
        else:
            font_name = "Helvetica"

        pdf.add_page()
        pdf.set_font(font_name, size=18)
        pdf.cell(160 if map_url else 0, 12, "Traffic Violation Report", new_x="LMARGIN", new_y="NEXT")
        if map_url:
            qr_image = qrcode.make(map_url)
            qr_buffer = BytesIO()
            qr_image.save(qr_buffer, format="PNG")
            qr_buffer.seek(0)
            pdf.image(qr_buffer, x=178, y=8, w=22, h=22)
        pdf.set_font(font_name, size=11)
        pdf.cell(0, 8, f"Report generated: {self._format_datetime(generated_at or datetime.now().astimezone())}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Location: {location or 'Not provided'}", new_x="LMARGIN", new_y="NEXT")
        if map_url:
            pdf.set_text_color(0, 0, 220)
            pdf.cell(
                0,
                8,
                "Open Google Maps",
                link=map_url,
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.set_text_color(0, 0, 0)
        if reporter_name.strip():
            pdf.cell(
                0,
                8,
                f"Reporter: {reporter_name.strip()}",
                new_x="LMARGIN",
                new_y="NEXT",
            )
        if bank_name.strip():
            pdf.cell(
                0,
                8,
                f"Bank: {bank_name.strip()}",
                new_x="LMARGIN",
                new_y="NEXT",
            )
        if bank_account.strip():
            pdf.cell(
                0,
                8,
                f"Bank account: {bank_account.strip()}",
                new_x="LMARGIN",
                new_y="NEXT",
            )
        pdf.cell(0, 8, f"Violations found: {len(violations)}", new_x="LMARGIN", new_y="NEXT")

        for index, item in enumerate(violations, start=1):
            pdf.ln(5)
            pdf.set_font(font_name, size=14)
            pdf.cell(0, 9, f"Violation {index}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font(font_name, size=11)
            pdf.cell(0, 7, f"Date/time: {self._format_datetime(item.get('captured_at'))}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(
                0,
                7,
                f"Time source: {item.get('timestamp_source') or 'Not specified'}",
                new_x="LMARGIN",
                new_y="NEXT",
            )
            item_location = str(item.get("location") or location or "Not provided")
            pdf.cell(0, 7, f"Location: {item_location}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 7, f"License plate: {item.get('plate_number') or 'Not provided'}", new_x="LMARGIN", new_y="NEXT")
            plate_number = str(item.get("plate_number") or "").strip()
            offense = str(
                item.get("offense")
                or "ขับขี่รถจักรยานยนต์โดยไม่สวมหมวกนิรภัย"
            )
            if (
                plate_number == "ไม่ติดแผ่นป้ายทะเบียน"
                and "ใช้รถที่ไม่ติดแผ่นป้ายทะเบียน" not in offense
            ):
                offense = f"{offense} และ ใช้รถที่ไม่ติดแผ่นป้ายทะเบียน"
            pdf.multi_cell(0, 7, f"Offense: {offense}")
            confidence = item.get("confidence")
            if confidence is not None:
                pdf.cell(0, 7, f"Detection confidence: {float(confidence):.1%}", new_x="LMARGIN", new_y="NEXT")

            evidence_details = self._watermarked_evidence(
                item.get("evidence_image"),
                item.get("captured_at"),
                item.get("latitude"),
                item.get("longitude"),
            )
            if evidence_details is not None:
                evidence, evidence_hash = evidence_details
                if pdf.get_y() > 170:
                    pdf.add_page()
                pdf.image(evidence, x=10, y=pdf.get_y() + 2, w=190, h=105, keep_aspect_ratio=True)
                pdf.ln(110)
                if pdf.get_y() > 265:
                    pdf.add_page()
                pdf.set_font(font_name, size=8)
                pdf.multi_cell(0, 5, f"Evidence image SHA-256 (watermarked PNG): {evidence_hash}")
                pdf.set_font(font_name, size=11)

        if pdf.get_y() > 235:
            pdf.add_page()
        pdf.ln(8)
        pdf.set_font(font_name, size=11)
        pdf.multi_cell(0, 8, certification_text)
        pdf.ln(8)
        pdf.cell(0, 8, "ลงชื่อ ____________________________________", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, "ผู้แจ้งเบาะแส", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"วันที่ ____________________  {reporter_name.strip()}", new_x="LMARGIN", new_y="NEXT")

        result = bytes(pdf.output())
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(result)
        return result
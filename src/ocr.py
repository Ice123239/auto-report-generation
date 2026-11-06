"""Plate-text extraction with EasyOCR."""

from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np
import streamlit as st


@st.cache_resource(show_spinner="กำลังเตรียมระบบอ่านข้อความป้ายทะเบียน...")
def _load_ocr_reader() -> Any:
    import easyocr

    return easyocr.Reader(["th", "en"])


class PlateOCR:
    """Extract a likely license-plate string from an OpenCV BGR image."""

    _PLATE_LIKE = re.compile(r"(?=.*\d)(?=.*[ก-๙A-Za-z])[ก-๙A-Za-z0-9\s-]{3,}")
    UNREADABLE_PLATE = "ไม่ติดแผ่นป้ายทะเบียน/ไม่ชัดเจน"
    DEFAULT_MIN_CONFIDENCE = 0.3
    _UPSCALE_FACTOR = 2.5
    _LATIN_TO_DIGIT = str.maketrans({"O": "0", "o": "0", "I": "1", "i": "1", "l": "1"})
    _THAI_PLATE = re.compile(r"^(\d{1,2})([ก-๙A-Z]{1,3})(\d{1,4})$")

    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1.")
        self.min_confidence = min_confidence
        self.engine = _load_ocr_reader()

    @staticmethod
    def _ocr_detections(
        result: Any,
    ) -> list[tuple[str, float, np.ndarray | None]]:
        detections: list[tuple[str, float, np.ndarray | None]] = []

        if not isinstance(result, (list, tuple)):
            return detections
        for item in result:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            text = item[1]
            confidence = item[2]
            if not isinstance(text, str) or not isinstance(confidence, (int, float, np.number)):
                continue
            try:
                polygon = np.asarray(item[0], dtype=np.float32)
                if polygon.ndim != 2 or polygon.shape[1] < 2:
                    polygon = None
                else:
                    polygon = polygon[:, :2]
            except (TypeError, ValueError):
                polygon = None
            detections.append((text.strip(), float(confidence), polygon))

        return [
            (text, score, polygon)
            for text, score, polygon in detections
            if text
        ]

    @classmethod
    def clean_plate_text(cls, text: str) -> str:
        """Normalize OCR output to the common Thai plate layout."""
        compact = re.sub(r"[\s-]+", "", text).translate(cls._LATIN_TO_DIGIT).upper()
        match = cls._THAI_PLATE.fullmatch(compact)
        if match:
            category, letters, digits = match.groups()
            return f"{category}{letters} {digits}"
        return " ".join(text.split())

    def read_plate(self, image: np.ndarray) -> dict[str, Any]:
        """Return the best plate-like OCR candidate and all recognized text."""
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image must be a non-empty OpenCV image.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have three color channels (BGR).")

        plate_region = image[int(image.shape[0] * 0.45):, :]
        if plate_region.size == 0:
            plate_region = image
        enlarged = cv2.resize(
            plate_region,
            None,
            fx=self._UPSCALE_FACTOR,
            fy=self._UPSCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC,
        )
        grayscale = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            grayscale
        )
        _, thresholded = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        detections: list[tuple[str, float, np.ndarray | None]] = []
        for processed_image in (enhanced, thresholded):
            result = self.engine.readtext(
                processed_image,
                detail=1,
                paragraph=False,
            )
            for text, score, polygon in self._ocr_detections(result):
                original_polygon = (
                    polygon / self._UPSCALE_FACTOR
                    if polygon is not None
                    else None
                )
                detections.append((text, score, original_polygon))
        best_by_text: dict[str, tuple[str, float, np.ndarray | None]] = {}
        for detection in detections:
            previous = best_by_text.get(detection[0])
            if previous is None or detection[1] > previous[1]:
                best_by_text[detection[0]] = detection
        detections = list(best_by_text.values())
        candidates = [
            (text, score, polygon)
            for text, score, polygon in detections
            if self._PLATE_LIKE.search(text.replace("\n", " "))
        ]
        candidate, confidence, polygon = (
            max(
                candidates,
                key=lambda detection: (
                    any("\u0e01" <= character <= "\u0e59" for character in detection[0]),
                    detection[1],
                ),
            )
            if candidates
            else ("", 0.0, None)
        )
        if candidate:
            candidate = self.clean_plate_text(candidate)
        if confidence <= self.min_confidence:
            candidate = self.UNREADABLE_PLATE
            polygon = None
        plate_crop = plate_region.copy()
        if polygon is not None and polygon.size:
            x1 = max(0, int(np.floor(polygon[:, 0].min())))
            y1 = max(0, int(np.floor(polygon[:, 1].min())))
            x2 = min(plate_region.shape[1], int(np.ceil(polygon[:, 0].max())) + 1)
            y2 = min(plate_region.shape[0], int(np.ceil(polygon[:, 1].max())) + 1)
            if x2 > x1 and y2 > y1:
                plate_crop = plate_region[y1:y2, x1:x2].copy()

        return {
            "plate_number": candidate,
            "confidence": confidence,
            "raw_text": " ".join(text for text, _, _ in detections),
            "plate_crop": plate_crop,
        }
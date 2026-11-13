"""YOLOv8-based motorcycle and no-helmet violation detection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from src.image_utils import resize_image_width


@st.cache_resource(show_spinner="กำลังโหลดโมเดล YOLOv8...")
def _load_yolo_model(model_path: str) -> YOLO:
    return YOLO(model_path)


class ViolationDetector:
    """Detect motorcycle and helmet violations in the configured sidewalk ROI.

    The simulated sidewalk ROI is a full-width horizontal rectangle positioned
    by its top edge and height. A motorcycle whose bounding-box center falls in
    this region is marked as riding on the sidewalk. Adjust it for each camera.
    Helmet violations still require a model class such as ``no_helmet`` or
    ``without helmet``.
    """

    _MOTORCYCLE_LABELS = {"motorcycle", "motorbike", "motor cycle"}
    _NO_HELMET_LABELS = {
        "no helmet",
        "without helmet",
        "helmetless",
        "unhelmeted",
        "nohelmet",
        "rider no helmet",
        "rider without helmet",
    }
    _SIDEWALK_LABELS = {"sidewalk", "footpath", "pavement", "walkway"}
    _MIN_MOTORCYCLE_AREA_RATIO = 0.02

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        sidewalk_roi_top: float = 0.65,
        sidewalk_roi_height: float = 0.35,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")
        if not 0.0 <= sidewalk_roi_top <= 1.0:
            raise ValueError("sidewalk_roi_top must be between 0 and 1.")
        if not 0.05 <= sidewalk_roi_height <= 0.50:
            raise ValueError("sidewalk_roi_height must be between 0.05 and 0.50.")

        self.model_path = str(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.sidewalk_roi_top = sidewalk_roi_top
        self.sidewalk_roi_height = sidewalk_roi_height
        self.model = _load_yolo_model(self.model_path)
        self.last_sidewalk_result: dict[str, Any] | None = None

    @staticmethod
    def _normalize_label(label: str) -> str:
        return " ".join(label.lower().replace("_", " ").replace("-", " ").split())

    def _get_sidewalk_roi(
        self, image_width: int, image_height: int
    ) -> tuple[int, int, int, int]:
        polygon = self._get_sidewalk_polygon(image_width, image_height)
        x, y, width, height = cv2.boundingRect(polygon)
        return (
            x,
            y,
            x + width,
            y + height,
        )

    def _get_sidewalk_polygon(self, image_width: int, image_height: int) -> np.ndarray:
        top = self.sidewalk_roi_top
        bottom = min(1.0, top + self.sidewalk_roi_height)
        return np.asarray(
            (
                (0, round(top * (image_height - 1))),
                (image_width - 1, round(top * (image_height - 1))),
                (image_width - 1, round(bottom * (image_height - 1))),
                (0, round(bottom * (image_height - 1))),
            ),
            dtype=np.int32,
        )

    @staticmethod
    def _intersection_over_union(
        first: tuple[int, int, int, int], second: tuple[int, int, int, int]
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _crop_bounds(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        padding_x = max(8, int((max(first[2], second[2]) - min(first[0], second[0])) * 0.08))
        padding_y = max(8, int((max(first[3], second[3]) - min(first[1], second[1])) * 0.08))
        return (
            max(0, min(first[0], second[0]) - padding_x),
            max(0, min(first[1], second[1]) - padding_y),
            min(image_width, max(first[2], second[2]) + padding_x),
            min(image_height, max(first[3], second[3]) + padding_y),
        )

    @staticmethod
    def _crop_motorcycle_plate(
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = bbox
        vehicle = image[y1:y2, x1:x2]
        vehicle_height, vehicle_width = vehicle.shape[:2]
        search_y1 = int(vehicle_height * 0.35)
        fallback_y2 = int(vehicle_height * 0.85)
        return (
            vehicle[search_y1:, :].copy(),
            vehicle[search_y1:fallback_y2, :].copy(),
        )

    def detect(
        self, image: np.ndarray, captured_at: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Return violation records containing an annotated evidence crop.

        Args:
            image: Source image in OpenCV BGR format.
            captured_at: Capture time to attach to each result, if known.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image must be a non-empty OpenCV image.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have three color channels (BGR).")

        source_image = image
        image = resize_image_width(image)
        scale_x = source_image.shape[1] / image.shape[1]
        scale_y = source_image.shape[0] / image.shape[0]
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
        )
        if not results:
            roi_polygon = self._get_sidewalk_polygon(image.shape[1], image.shape[0])
            roi_x1, roi_y1, roi_x2, roi_y2 = self._get_sidewalk_roi(
                image.shape[1], image.shape[0]
            )
            annotated = image.copy()
            cv2.polylines(annotated, [roi_polygon], True, (0, 0, 255), 3)
            self.last_sidewalk_result = {
                "image": annotated,
                "class_available": True,
                "boxes": [(roi_x1, roi_y1, roi_x2, roi_y2)],
                "polygon": roi_polygon,
                "simulated": True,
            }
            return []

        result = results[0]
        names = result.names
        detections: list[dict[str, Any]] = []
        sidewalk_labels = {
            self._normalize_label(str(label))
            for label in (names.values() if isinstance(names, dict) else names)
        } & self._SIDEWALK_LABELS
        sidewalk_image = image.copy()
        roi_x1, roi_y1, roi_x2, roi_y2 = self._get_sidewalk_roi(
            image.shape[1], image.shape[0]
        )
        roi_polygon = self._get_sidewalk_polygon(image.shape[1], image.shape[0])
        sidewalk_boxes: list[tuple[int, int, int, int]] = []
        cv2.polylines(sidewalk_image, [roi_polygon], True, (0, 0, 255), 3)
        cv2.putText(
            sidewalk_image,
            "SIMULATED SIDEWALK ROI",
            (roi_x1 + 8, max(20, roi_y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        if result.boxes is None:
            self.last_sidewalk_result = {
                "image": sidewalk_image,
                "class_available": True,
                "boxes": sidewalk_boxes,
                "polygon": roi_polygon,
                "simulated": True,
            }
            return detections

        for box in result.boxes:
            class_id = int(box.cls.item())
            raw_label = names[class_id] if isinstance(names, dict) else names[class_id]
            label = self._normalize_label(str(raw_label))
            if label in self._SIDEWALK_LABELS:
                coords = box.xyxy[0].cpu().numpy()
                sx1, sy1, sx2, sy2 = (
                    int(round(float(value))) for value in coords
                )
                sx1 = max(0, min(image.shape[1], sx1))
                sx2 = max(0, min(image.shape[1], sx2))
                sy1 = max(0, min(image.shape[0], sy1))
                sy2 = max(0, min(image.shape[0], sy2))
                if sx2 > sx1 and sy2 > sy1:
                    sidewalk_boxes.append((sx1, sy1, sx2, sy2))
                    cv2.rectangle(sidewalk_image, (sx1, sy1), (sx2, sy2), (0, 0, 255), 3)
                    cv2.putText(
                        sidewalk_image,
                        f"{label} {float(box.conf.item()):.2f}",
                        (sx1, max(20, sy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                continue
            is_motorcycle = class_id == 3 or label in self._MOTORCYCLE_LABELS
            if not is_motorcycle and label not in self._NO_HELMET_LABELS:
                continue
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = (int(round(float(value))) for value in coords)
            x1 = max(0, min(image.shape[1], x1))
            x2 = max(0, min(image.shape[1], x2))
            y1 = max(0, min(image.shape[0], y1))
            y2 = max(0, min(image.shape[0], y2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "label": "motorcycle" if is_motorcycle else label,
                "class_id": class_id,
                "bbox": (x1, y1, x2, y2),
                "confidence": float(box.conf.item()),
            })

        self.last_sidewalk_result = {
            "image": sidewalk_image,
            "class_available": True,
            "boxes": sidewalk_boxes,
            "polygon": roi_polygon,
            "simulated": True,
        }
        image_area = image.shape[0] * image.shape[1]
        motorcycles = [
            item
            for item in detections
            if item["label"] in self._MOTORCYCLE_LABELS
            and (
                (item["bbox"][2] - item["bbox"][0])
                * (item["bbox"][3] - item["bbox"][1])
            ) >= image_area * self._MIN_MOTORCYCLE_AREA_RATIO
        ]
        motorcycles.sort(
            key=lambda item: (
                (item["bbox"][2] - item["bbox"][0])
                * (item["bbox"][3] - item["bbox"][1])
            ),
            reverse=True,
        )
        no_helmet = [item for item in detections if item["label"] in self._NO_HELMET_LABELS]
        violations: list[dict[str, Any]] = []

        for vehicle_index, motorcycle in enumerate(motorcycles):
            associated_riders = [
                rider for rider in no_helmet
                if self._intersection_over_union(rider["bbox"], motorcycle["bbox"]) > 0
            ]
            rider = (
                max(
                    associated_riders,
                    key=lambda item: self._intersection_over_union(
                        item["bbox"], motorcycle["bbox"]
                    ),
                )
                if associated_riders
                else None
            )
            bx1, by1, bx2, by2 = motorcycle["bbox"]
            center_x = (bx1 + bx2) // 2
            center_y = (by1 + by2) // 2
            riding_on_sidewalk = (
                cv2.pointPolygonTest(
                    roi_polygon,
                    (float(center_x), float(center_y)),
                    False,
                )
                >= 0
            )
            if not riding_on_sidewalk and rider is None:
                continue

            x1, y1, x2, y2 = self._crop_bounds(
                motorcycle["bbox"],
                rider["bbox"] if rider is not None else motorcycle["bbox"],
                image.shape[1],
                image.shape[0],
            )
            evidence = image[y1:y2, x1:x2].copy()
            plate_bbox = (
                round(motorcycle["bbox"][0] * scale_x),
                round(motorcycle["bbox"][1] * scale_y),
                round(motorcycle["bbox"][2] * scale_x),
                round(motorcycle["bbox"][3] * scale_y),
            )
            plate_search_image, plate_fallback_image = self._crop_motorcycle_plate(
                source_image,
                plate_bbox,
            )
            offense_parts: list[str] = []
            if riding_on_sidewalk:
                offense_parts.append("ขับขี่บนทางเท้า")
            if rider is not None:
                offense_parts.append("ขับขี่รถจักรยานยนต์โดยไม่สวมหมวกนิรภัย")
            crop_roi_polygon = roi_polygon.copy()
            crop_roi_polygon[:, 0] -= x1
            crop_roi_polygon[:, 1] -= y1
            crop_roi_polygon[:, 0] = np.clip(
                crop_roi_polygon[:, 0], 0, evidence.shape[1] - 1
            )
            crop_roi_polygon[:, 1] = np.clip(
                crop_roi_polygon[:, 1], 0, evidence.shape[0] - 1
            )
            cv2.polylines(evidence, [crop_roi_polygon], True, (0, 0, 255), 2)
            detections_to_draw = [(motorcycle, (0, 0, 255) if riding_on_sidewalk else (0, 165, 255))]
            if rider is not None:
                detections_to_draw.append((rider, (0, 0, 255)))
            for detection, color in detections_to_draw:
                bx1, by1, bx2, by2 = detection["bbox"]
                bx1, by1 = max(bx1 - x1, 0), max(by1 - y1, 0)
                bx2, by2 = min(bx2 - x1, evidence.shape[1]), min(by2 - y1, evidence.shape[0])
                cv2.rectangle(evidence, (bx1, by1), (bx2, by2), color, 2)
                caption = f'{detection["label"]} {detection["confidence"]:.2f}'
                cv2.putText(
                    evidence,
                    caption,
                    (bx1, max(16, by1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            violations.append({
                "captured_at": captured_at or datetime.now().astimezone(),
                "motorcycle_bbox": motorcycle["bbox"],
                "bbox_area": (
                    (motorcycle["bbox"][2] - motorcycle["bbox"][0])
                    * (motorcycle["bbox"][3] - motorcycle["bbox"][1])
                ),
                "is_primary_vehicle": vehicle_index == 0,
                "rider_bbox": rider["bbox"] if rider is not None else None,
                "confidence": min(
                    motorcycle["confidence"],
                    rider["confidence"] if rider is not None else motorcycle["confidence"],
                ),
                "offense": " และ ".join(offense_parts),
                "evidence_image": evidence,
                "plate_search_image": plate_search_image,
                "plate_fallback_image": plate_fallback_image,
                "plate_number": "",
            })

        violations.sort(key=lambda item: item["bbox_area"], reverse=True)
        if violations:
            for index, violation in enumerate(violations):
                violation["is_primary_vehicle"] = index == 0
        return violations
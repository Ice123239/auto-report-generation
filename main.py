"""Streamlit interface for traffic-violation evidence processing."""

from __future__ import annotations

from datetime import datetime
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import ExifTags, Image
import streamlit as st

from src.detector import ViolationDetector
from src.ocr import PlateOCR
from src.report_generator import ReportGenerator


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_MODEL_PATH = "yolov8n.pt"
DEFAULT_LOCATION = "ถนนสุขุมวิท เขตวัฒนา กรุงเทพมหานคร"
DEFAULT_ROI_TOP = 0.65
DEFAULT_ROI_HEIGHT = 0.35
UNREADABLE_PLATE = PlateOCR.UNREADABLE_PLATE
VIOLATION_DESCRIPTION = "ขับขี่รถจักรยานยนต์บนทางเท้า"
OFFENSE_OPTIONS = [
    "ขับขี่รถจักรยานยนต์บนทางเท้า",
    "ไม่ติดแผ่นป้ายทะเบียน/ป้ายไม่ชัดเจน",
    "ไม่สวมหมวกนิรภัย",
    "ฝ่าฝืนสัญญาณไฟจราจร",
    "ไม่พบการกระทำผิด (ไม่นำลงรายงาน)",
]
NO_VIOLATION = OFFENSE_OPTIONS[-1]
THAI_BANKS = [
    "ธนาคารกสิกรไทย (KBANK)",
    "ธนาคารไทยพาณิชย์ (SCB)",
    "ธนาคารกรุงเทพ (BBL)",
    "ธนาคารกรุงไทย (KTB)",
    "ธนาคารกรุงศรีอยุธยา (BAY)",
    "ธนาคารทหารไทยธนชาต (TTB)",
    "ธนาคารออมสิน (GSB)",
    "ธนาคาร ธ.ก.ส.",
    "ธนาคารอาคารสงเคราะห์ (GHB)",
    "ธนาคารซีไอเอ็มบี ไทย (CIMBT)",
    "ธนาคารยูโอบี (UOB)",
    "ธนาคารแลนด์ แอนด์ เฮ้าส์ (LHBANK)",
    "ธนาคารอิสลามแห่งประเทศไทย (iBank)",
]


def _draw_sidewalk_roi(
    image: np.ndarray,
    roi_top: float,
    roi_height: float,
) -> np.ndarray:
    preview = image.copy()
    image_height, image_width = preview.shape[:2]
    top = round(roi_top * (image_height - 1))
    bottom = round(min(1.0, roi_top + roi_height) * (image_height - 1))
    overlay = preview.copy()
    cv2.rectangle(overlay, (0, top), (image_width - 1, bottom), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.2, preview, 0.8, 0, dst=preview)
    cv2.rectangle(preview, (0, top), (image_width - 1, bottom), (0, 0, 255), 4)
    return preview


def _read_video_first_frame(media_bytes: bytes, suffix: str) -> np.ndarray:
    temporary_path: str | None = None
    capture: cv2.VideoCapture | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(media_bytes)
            temporary_path = temporary_file.name
        capture = cv2.VideoCapture(temporary_path)
        if not capture.isOpened():
            raise ValueError("ไม่สามารถเปิดไฟล์วิดีโอได้")
        success, frame = capture.read()
        if not success or frame is None:
            raise ValueError("ไม่สามารถอ่านเฟรมแรกของวิดีโอได้")
        return frame
    finally:
        if capture is not None:
            capture.release()
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


def _dms_to_degrees(values: tuple[object, ...]) -> float:
    degrees, minutes, seconds = (float(value) for value in values)
    return degrees + minutes / 60 + seconds / 3600


def _read_exif_gps(image_bytes: bytes) -> tuple[float, float] | None:
    """Read latitude/longitude from image EXIF without contacting a service."""
    try:
        image = Image.open(BytesIO(image_bytes))
        exif = image.getexif()
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except (OSError, AttributeError, KeyError, TypeError, ValueError):
        return None

    gps = {ExifTags.GPSTAGS.get(tag, tag): value for tag, value in gps_ifd.items()}
    latitude = gps.get("GPSLatitude")
    longitude = gps.get("GPSLongitude")
    latitude_ref = gps.get("GPSLatitudeRef")
    longitude_ref = gps.get("GPSLongitudeRef")
    if not latitude or not longitude or not latitude_ref or not longitude_ref:
        return None

    try:
        lat = _dms_to_degrees(latitude)
        lon = _dms_to_degrees(longitude)
        lat_ref = latitude_ref.decode("ascii") if isinstance(latitude_ref, bytes) else str(latitude_ref)
        lon_ref = longitude_ref.decode("ascii") if isinstance(longitude_ref, bytes) else str(longitude_ref)
        if lat_ref.upper().startswith("S"):
            lat = -lat
        if lon_ref.upper().startswith("W"):
            lon = -lon
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            return None
        return lat, lon
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _read_exif_datetime(image_bytes: bytes) -> datetime | None:
    """Read the original capture timestamp from EXIF when one is present."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            exif = image.getexif()
    except (OSError, AttributeError, TypeError, ValueError):
        return None
    for tag in (36867, 36868, 306):
        value = exif.get(tag)
        if not value:
            continue
        try:
            return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            continue
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def _reverse_geocode(latitude: float, longitude: float) -> str:
    """Look up an address only after the user opts in; coordinates leave the app."""
    query = urlencode({
        "format": "jsonv2",
        "lat": f"{latitude:.6f}",
        "lon": f"{longitude:.6f}",
        "accept-language": "th",
    })
    request = Request(
        f"https://nominatim.openstreetmap.org/reverse?{query}",
        headers={"User-Agent": "TrafficViolationReport/1.0"},
    )
    try:
        with urlopen(request, timeout=8) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"ค้นหาที่อยู่ไม่สำเร็จ: {error}") from error
    address = result.get("display_name")
    if not address:
        raise RuntimeError("บริการค้นหาที่อยู่ไม่พบข้อมูลสำหรับพิกัดนี้")
    return str(address)


def _show_history() -> None:
    history = st.session_state.get("detection_history", [])
    if not history:
        st.info("ยังไม่มีประวัติการตรวจจับในเซสชันนี้")
        return
    rows = [
            {
                "record_id": item["record_id"],
                "วันเวลา": item["captured_at"],
                "เวลาประมวลผล": item["processed_at"],
                "ไฟล์": item["filename"],
                "สถานที่": item["location"],
                "ป้ายทะเบียน": item["plate_number"] or "ไม่พบข้อความ",
                "ข้อหา": item["offense"],
                "แหล่งที่มาป้ายทะเบียน": item.get("plate_source", "สแกนด้วย AI"),
                "แหล่งที่มาวันเวลา": item.get("timestamp_source", "-"),
                "ความมั่นใจ": (
                    f"{item['confidence']:.1%}"
                    if item["confidence"] is not None
                    else "-"
                ),
            }
            for item in reversed(history)
        ]
    edited_rows = st.data_editor(
        rows,
        key="history_editor",
        num_rows="dynamic",
        column_order=(
            "วันเวลา",
            "เวลาประมวลผล",
            "ไฟล์",
            "สถานที่",
            "ป้ายทะเบียน",
            "ข้อหา",
            "แหล่งที่มาป้ายทะเบียน",
            "แหล่งที่มาวันเวลา",
            "ความมั่นใจ",
        ),
        disabled=(
            "record_id",
            "วันเวลา",
            "เวลาประมวลผล",
            "ไฟล์",
            "แหล่งที่มาป้ายทะเบียน",
            "แหล่งที่มาวันเวลา",
            "ความมั่นใจ",
        ),
        hide_index=True,
        use_container_width=True,
    )
    records_by_id = {item["record_id"]: item for item in history}
    edited_records = (
        edited_rows
        if isinstance(edited_rows, list)
        else edited_rows.to_dict(orient="records")
    )
    synchronized_history: list[dict[str, object]] = []
    for edited_row in edited_records:
        record = records_by_id.get(edited_row.get("record_id"))
        if record is None:
            location = str(edited_row.get("สถานที่") or "").strip()
            plate_number = str(edited_row.get("ป้ายทะเบียน") or "").strip()
            offense = str(edited_row.get("ข้อหา") or "").strip()
            if not any((location, plate_number, offense)):
                continue
            now = datetime.now().astimezone()
            record = {
                "record_id": sha256(
                    f"{now.isoformat()}|{location}|{plate_number}|{offense}".encode("utf-8")
                ).hexdigest(),
                "captured_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "processed_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "filename": "เพิ่มในประวัติ",
                "image_sha256": "",
                "location": location,
                "plate_number": plate_number,
                "offense": offense,
                "plate_source": "ผู้ใช้เพิ่ม",
                "timestamp_source": "ผู้ใช้ระบุ",
                "confidence": None,
                "evidence_image": None,
                "latitude": None,
                "longitude": None,
            }
        else:
            record["location"] = str(edited_row.get("สถานที่") or "").strip()
            record["plate_number"] = str(edited_row.get("ป้ายทะเบียน") or "").strip()
            record["offense"] = str(edited_row.get("ข้อหา") or "").strip()
        if _is_no_violation(record.get("offense")):
            continue
        synchronized_history.append(record)

    history[:] = synchronized_history
    if history:
        csv_buffer = StringIO(newline="")
        csv_fields = (
            "วันเวลา",
            "เวลาประมวลผล",
            "ไฟล์",
            "สถานที่",
            "ป้ายทะเบียน",
            "ข้อหา",
            "แหล่งที่มาป้ายทะเบียน",
            "แหล่งที่มาวันเวลา",
            "ความมั่นใจ",
        )
        writer = csv.DictWriter(csv_buffer, fieldnames=csv_fields)
        writer.writeheader()
        for item in reversed(history):
            writer.writerow({
                "วันเวลา": item.get("captured_at", ""),
                "เวลาประมวลผล": item.get("processed_at", ""),
                "ไฟล์": item.get("filename", ""),
                "สถานที่": item.get("location", ""),
                "ป้ายทะเบียน": item.get("plate_number", ""),
                "ข้อหา": item.get("offense", ""),
                "แหล่งที่มาป้ายทะเบียน": item.get("plate_source", ""),
                "แหล่งที่มาวันเวลา": item.get("timestamp_source", ""),
                "ความมั่นใจ": item.get("confidence", ""),
            })
        st.download_button(
            "ดาวน์โหลดประวัติเป็น CSV",
            data=csv_buffer.getvalue().encode("utf-8-sig"),
            file_name="detection_history.csv",
            mime="text/csv",
            key="history_csv_download",
        )


def _sharpness_score(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _select_clearest_video_detections(
    detections: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Keep the sharpest sampled frame for each overlapping vehicle box."""
    def detection_sharpness(item: dict[str, object]) -> float:
        image = item.get("plate_image", item.get("evidence_image"))
        return _sharpness_score(image) if isinstance(image, np.ndarray) else 0.0

    ranked = sorted(
        detections,
        key=detection_sharpness,
        reverse=True,
    )
    selected: list[dict[str, object]] = []
    for candidate in ranked:
        candidate_box = candidate.get("motorcycle_bbox")
        if not (
            isinstance(candidate_box, tuple)
            and len(candidate_box) == 4
            and all(isinstance(value, int) for value in candidate_box)
        ):
            selected.append(candidate)
            continue
        if any(
            isinstance(existing.get("motorcycle_bbox"), tuple)
            and len(existing["motorcycle_bbox"]) == 4
            and all(isinstance(value, int) for value in existing["motorcycle_bbox"])
            and ViolationDetector._intersection_over_union(
                candidate_box,
                existing["motorcycle_bbox"],
            ) >= 0.3
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return selected


def _reportable_history() -> list[dict[str, object]]:
    return [
        {
            "captured_at": item.get("captured_at"),
            "plate_number": item.get("plate_number"),
            "offense": item.get("offense"),
            "confidence": item.get("confidence"),
            "evidence_image": item.get("evidence_image"),
            "latitude": item.get("latitude"),
            "longitude": item.get("longitude"),
            "timestamp_source": item.get("timestamp_source"),
            "location": item.get("location"),
        }
        for item in st.session_state.get("detection_history", [])
        if _is_reportable_offense(item.get("offense"))
    ]


def _is_reportable_offense(offense: object) -> bool:
    value = str(offense or "").strip()
    return bool(value) and not _is_no_violation(value)


def _is_no_violation(offense: object) -> bool:
    value = str(offense or "").strip()
    return NO_VIOLATION in value or "ไม่พบการกระทำผิด" in value


st.set_page_config(
    page_title="ระบบรายงานการทำผิดกฎจราจร",
    page_icon="🚦",
    layout="wide",
)
st.session_state.setdefault("detection_history", [])

st.title("ระบบตรวจจับและรายงานการทำผิดกฎจราจร")
st.caption("อัปโหลดภาพหรือวิดีโอ ตรวจสอบภาพหลักฐาน และสร้างรายงาน PDF")

with st.sidebar:
    st.header("เมนู")
    page = st.radio(
        "ไปยัง",
        ("ประมวลผลหลักฐาน", "ประวัติการตรวจจับ", "เกี่ยวกับระบบ"),
        label_visibility="collapsed",
    )

if page == "เกี่ยวกับระบบ":
    st.subheader("เกี่ยวกับระบบ")
    st.write(
        "ระบบใช้ YOLOv8 ตรวจจับรถจักรยานยนต์และคลาสผู้ขับขี่ไม่สวมหมวกกันน็อก "
        "พร้อม EasyOCR สำหรับอ่านข้อความป้ายทะเบียน และสร้างรายงาน PDF"
    )
    st.warning(
        "เขตทางเท้าเป็นกรอบ ROI สี่เหลี่ยมเต็มความกว้างที่ปรับตำแหน่งและความสูงได้. "
        "ปรับให้ตรงกับมุมกล้องก่อนใช้งานจริง; คลาส no_helmet ต้องมีอยู่ใน weights ที่ใช้"
    )
elif page == "ประวัติการตรวจจับ":
    st.subheader("ประวัติการตรวจจับย้อนหลัง")
    _show_history()
    report_violations = _reportable_history()
    if not report_violations:
        st.warning("ไม่พบรายการกระทำความผิดสำหรับออกรายงาน PDF")
    else:
        history_report_key = sha256(
            repr([
                (
                    item.get("captured_at"),
                    item.get("plate_number"),
                    item.get("offense"),
                    item.get("location"),
                )
                for item in report_violations
            ]).encode("utf-8")
        ).hexdigest()
        if st.button("สร้างรายงาน PDF จากประวัติล่าสุด", key="history_report_button"):
            try:
                st.session_state["history_report_pdf"] = ReportGenerator().generate_report(
                    report_violations
                )
                st.session_state["history_report_key"] = history_report_key
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                st.error(f"สร้างรายงานไม่สำเร็จ: {error}")
        if (
            st.session_state.get("history_report_key") == history_report_key
            and "history_report_pdf" in st.session_state
        ):
            st.download_button(
                "ดาวน์โหลดรายงาน PDF",
                data=st.session_state["history_report_pdf"],
                file_name="traffic_violation_report.pdf",
                mime="application/pdf",
                key="history_report_download",
            )
else:
    with st.sidebar:
        st.subheader("ตั้งค่าตรวจจับ")
        st.text_input(
            "พาธไฟล์โมเดล YOLOv8",
            value=DEFAULT_MODEL_PATH,
            disabled=True,
        )
        model_path = DEFAULT_MODEL_PATH
        confidence = st.slider("ค่าความเชื่อมั่นขั้นต่ำ", 0.0, 1.0, 0.20, 0.05)
        roi_top = st.slider(
            "ตำแหน่งขอบบนทางเท้า (Y-Top)",
            min_value=0.00,
            max_value=1.00,
            value=DEFAULT_ROI_TOP,
            step=0.01,
            format="%.2f",
        )
        roi_height = st.slider(
            "ความหนาเขตทางเท้า (Height)",
            min_value=0.05,
            max_value=0.50,
            value=DEFAULT_ROI_HEIGHT,
            step=0.01,
            format="%.2f",
        )
        with st.container(border=True):
            st.subheader("ข้อมูลผู้แจ้งสำหรับรายงาน")
            reporter_name = st.text_input(
                "ชื่อผู้แจ้ง",
                key="reporter_name",
                placeholder="ไม่บังคับ",
            )
            bank_name = st.selectbox(
                "ธนาคาร",
                THAI_BANKS,
                key="bank_name",
            )
            bank_account = st.text_input(
                "เลขบัญชีธนาคาร",
                key="bank_account",
                placeholder="ไม่บังคับ",
                help="ชื่อธนาคารและเลขบัญชีจะถูกรวมใน PDF และไม่บันทึกในประวัติการตรวจจับ",
            )

    uploaded_file = st.file_uploader(
        "เลือกไฟล์ภาพหรือวิดีโอ",
        type=("jpg", "png", "mp4", "mov"),
    )

    if uploaded_file is not None:
        media_bytes = uploaded_file.getvalue()
        suffix = Path(uploaded_file.name).suffix.lower()
        image_sha256 = sha256(media_bytes).hexdigest()
        settings_key = f"{model_path}|{confidence:.2f}|{roi_top:.2f}|{roi_height:.2f}"
        media_key = sha256(media_bytes + settings_key.encode("utf-8")).hexdigest()
        gps = _read_exif_gps(media_bytes) if suffix in IMAGE_SUFFIXES else None
        exif_captured_at = (
            _read_exif_datetime(media_bytes)
            if suffix in IMAGE_SUFFIXES
            else None
        )
        same_active_evidence = (
            st.session_state.get("active_evidence_sha256") == image_sha256
        )
        duplicate_evidence = any(
            item.get("image_sha256") == image_sha256
            for item in st.session_state["detection_history"]
        ) and not same_active_evidence
        address_default = ""

        with st.sidebar:
            with st.container(border=True):
                st.subheader("สถานที่เกิดเหตุ")
                if gps:
                    st.caption(f"พิกัด GPS จาก EXIF: {gps[0]:.6f}, {gps[1]:.6f}")
                    use_reverse_geocoding = st.checkbox(
                        "ค้นหาที่อยู่ผ่าน OpenStreetMap (ส่งพิกัดออกจากแอป)",
                        value=False,
                        key=f"reverse_geocode_{media_key}",
                    )
                    if use_reverse_geocoding:
                        try:
                            with st.spinner("กำลังค้นหาที่อยู่จากพิกัด..."):
                                address_default = _reverse_geocode(*gps)
                        except RuntimeError as error:
                            st.warning(str(error))
                else:
                    st.caption("ไม่พบพิกัด GPS ใน EXIF; ใช้สถานที่เริ่มต้นซึ่งแก้ไขได้")

                fallback_location = (
                    f"{gps[0]:.6f}, {gps[1]:.6f}" if gps else DEFAULT_LOCATION
                )
                location_default = address_default or fallback_location
                location_key = f"location_{media_key}"
                if address_default and st.session_state.get(location_key) in (
                    None,
                    "",
                    fallback_location,
                ):
                    st.session_state[location_key] = address_default
                location = st.text_input(
                    "ที่อยู่หรือพิกัด",
                    value=location_default,
                    key=location_key,
                    placeholder="กรอกสถานที่เกิดเหตุ",
                )
                if suffix in IMAGE_SUFFIXES:
                    if gps is None or exif_captured_at is None:
                        missing_metadata = []
                        if gps is None:
                            missing_metadata.append("พิกัด GPS")
                        if exif_captured_at is None:
                            missing_metadata.append("วัน-เวลา")
                        warning = (
                            "ภาพนี้ไม่มีข้อมูล EXIF สำหรับ "
                            + " และ ".join(missing_metadata)
                            + "."
                        )
                        if exif_captured_at is None:
                            warning += " โปรดระบุวัน-เวลาเกิดเหตุด้านล่าง"
                        st.warning(warning)
                    if exif_captured_at is None:
                        event_date = st.date_input(
                            "วันที่เกิดเหตุ",
                            value=datetime.now().date(),
                            key=f"event_date_{media_key}",
                        )
                        event_time = st.time_input(
                            "เวลาเกิดเหตุ",
                            value=datetime.now().time().replace(microsecond=0),
                            key=f"event_time_{media_key}",
                        )
                        captured_at = datetime.combine(
                            event_date,
                            event_time,
                        ).astimezone()
                        timestamp_source = "ผู้ใช้ระบุ"
                    else:
                        captured_at = exif_captured_at
                        timestamp_source = "EXIF"
                        st.caption(
                            f"วัน-เวลาจาก EXIF: {captured_at.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                else:
                    captured_at = None
                    timestamp_source = "เวลาจากการประมวลผลวิดีโอ"
            if duplicate_evidence:
                st.warning(
                    "พบไฟล์หลักฐาน SHA-256 นี้ในประวัติแล้ว ระบบปิดการประมวลผลและสร้างรายงานซ้ำ"
                )

        st.subheader("หลักฐานต้นฉบับ")
        if suffix in IMAGE_SUFFIXES:
            try:
                source_image = Image.open(BytesIO(media_bytes)).convert("RGB")
            except (OSError, ValueError) as error:
                st.error(f"ไม่สามารถอ่านไฟล์ภาพได้: {error}")
                st.stop()
            preview_bgr = cv2.cvtColor(np.asarray(source_image), cv2.COLOR_RGB2BGR)
        else:
            source_image = None
            try:
                preview_bgr = _read_video_first_frame(media_bytes, suffix)
            except (OSError, ValueError) as error:
                st.error(f"ไม่สามารถอ่านวิดีโอได้: {error}")
                st.stop()
            st.video(media_bytes)
            st.info("วิดีโอจะถูกตรวจทุก 1 วินาทีเพื่อค้นหาภาพหลักฐาน")
        st.image(
            cv2.cvtColor(
                _draw_sidewalk_roi(preview_bgr, roi_top, roi_height),
                cv2.COLOR_BGR2RGB,
            ),
            caption=(
                "ตัวอย่างกรอบ ROI ทางเท้าแบบเรียลไทม์ "
                f"(Y-Top={roi_top:.2f}, Height={roi_height:.2f})"
            ),
            use_container_width=True,
        )
        st.caption("กรอบสีแดงแสดงตำแหน่งและความสูงของเขตทางเท้าตาม Slider")

        if st.session_state.get("processed_key") != media_key:
            for key in ("violations", "sidewalk_result", "processed_key", "report_pdf", "report_key"):
                st.session_state.pop(key, None)

        if duplicate_evidence:
            st.warning("ไฟล์นี้เคยถูกบันทึกในเซสชันนี้แล้ว เพื่อป้องกันการส่งซ้ำ")
        process_clicked = st.button(
            "ประมวลผลซ้ำและค้นหาการกระทำผิด",
            type="primary",
            disabled=duplicate_evidence,
        )
        if (
            st.session_state.get("processed_key") != media_key
            and not duplicate_evidence
        ) or process_clicked:
            for key in ("report_pdf", "report_key", "ocr_error"):
                st.session_state.pop(key, None)
            processing_status = st.status("กำลังประมวลผลหลักฐาน...", expanded=False)
            video_progress = None
            try:
                detector = ViolationDetector(
                    model_path=model_path,
                    confidence=confidence,
                    sidewalk_roi_top=roi_top,
                    sidewalk_roi_height=roi_height,
                )
                violations = []
                sidewalk_result = None
                if source_image is not None:
                    rgb_image = np.asarray(source_image)
                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    violations = detector.detect(
                        bgr_image,
                        captured_at=captured_at,
                    )
                    sidewalk_result = detector.last_sidewalk_result
                else:
                    temporary_path: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
                            temporary_file.write(media_bytes)
                            temporary_path = temporary_file.name
                        capture = cv2.VideoCapture(temporary_path)
                        if not capture.isOpened():
                            raise ValueError("ไม่สามารถเปิดไฟล์วิดีโอได้")
                        fps = capture.get(cv2.CAP_PROP_FPS)
                        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        if total_frames > 0:
                            video_progress = st.progress(
                                0.0,
                                text="กำลังตรวจสอบวิดีโอ...",
                            )
                        frame_interval = max(1, round(fps)) if fps and fps > 0 else 1
                        frame_index = 0
                        try:
                            while True:
                                success, frame = capture.read()
                                if not success:
                                    break
                                if frame_index % frame_interval == 0:
                                    captured_at = datetime.now().astimezone()
                                    violations.extend(
                                        detector.detect(frame, captured_at=captured_at)
                                    )
                                    sidewalk_result = detector.last_sidewalk_result
                                    if video_progress is not None:
                                        video_progress.progress(
                                            min(frame_index / total_frames, 1.0),
                                            text=(
                                                f"กำลังตรวจสอบวิดีโอ... "
                                                f"{frame_index}/{total_frames} เฟรม"
                                            ),
                                        )
                                    elif frame_index % (frame_interval * 10) == 0:
                                        processing_status.update(
                                            label=(
                                                f"กำลังตรวจสอบวิดีโอ "
                                                f"(เฟรมที่ {frame_index})..."
                                            )
                                        )
                                frame_index += 1
                        finally:
                            capture.release()
                    finally:
                        if temporary_path is not None:
                            Path(temporary_path).unlink(missing_ok=True)

                if source_image is None:
                    violations = _select_clearest_video_detections(violations)

                if violations:
                    violations.sort(
                        key=lambda item: item.get("bbox_area", 0),
                        reverse=True,
                    )
                    for vehicle_index, violation in enumerate(violations):
                        violation["is_primary_vehicle"] = vehicle_index == 0
                        violation["timestamp_source"] = timestamp_source
                    if gps is not None:
                        for violation in violations:
                            violation["latitude"], violation["longitude"] = gps
                    try:
                        plate_reader = PlateOCR()
                        for violation in violations:
                            plate_result = plate_reader.read_plate(
                                violation.get("plate_image", violation["evidence_image"])
                            )
                            plate_number = plate_result["plate_number"]
                            if (
                                violation.get("is_primary_vehicle")
                                and plate_number == UNREADABLE_PLATE
                            ):
                                plate_number = "ไม่ติดแผ่นป้ายทะเบียน"
                            violation["plate_number"] = plate_number
                            violation["plate_source"] = (
                                "สแกนด้วย AI"
                                if plate_result["plate_number"] != UNREADABLE_PLATE
                                else "สแกนด้วย AI (อ่านไม่ได้)"
                            )
                            violation["plate_ocr_text"] = plate_result["raw_text"]
                            violation["plate_ocr_confidence"] = plate_result["confidence"]
                            violation["plate_crop"] = plate_result["plate_crop"]
                    except (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError) as error:
                        st.session_state["ocr_error"] = str(error)
                        for vehicle_index, violation in enumerate(violations):
                            default_plate = (
                                "ไม่ติดแผ่นป้ายทะเบียน"
                                if vehicle_index == 0
                                else UNREADABLE_PLATE
                            )
                            violation.setdefault("plate_number", default_plate)
                            violation.setdefault("plate_source", "สแกนด้วย AI (อ่านไม่ได้)")
                            violation.setdefault("plate_ocr_text", "")
                            violation.setdefault("plate_ocr_confidence", 0.0)

                st.session_state["violations"] = violations
                st.session_state["sidewalk_result"] = sidewalk_result
                st.session_state["processed_key"] = media_key

                processed_at = datetime.now().astimezone()
                run_id = sha256(
                    f"{media_key}|{processed_at.isoformat()}".encode("utf-8")
                ).hexdigest()
                history = st.session_state["detection_history"]
                if same_active_evidence:
                    history[:] = [
                        item
                        for item in history
                        if item.get("image_sha256") != image_sha256
                    ]
                for index, violation in enumerate(violations, start=1):
                    record_id = f"{run_id}-{index}"
                    captured_at = violation["captured_at"]
                    history.append({
                        "record_id": record_id,
                        "captured_at": captured_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "processed_at": processed_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": uploaded_file.name,
                        "image_sha256": image_sha256,
                        "location": location,
                        "plate_number": violation.get("plate_number", ""),
                        "plate_source": violation.get("plate_source", "สแกนด้วย AI"),
                        "timestamp_source": violation.get("timestamp_source", timestamp_source),
                        "offense": violation.get("offense", VIOLATION_DESCRIPTION),
                        "confidence": violation.get("confidence"),
                        "evidence_image": violation.get("evidence_image"),
                        "latitude": violation.get("latitude"),
                        "longitude": violation.get("longitude"),
                    })
                    violation["_history_id"] = record_id
                st.session_state["active_evidence_sha256"] = image_sha256
                processing_status.update(
                    label="ประมวลผลเสร็จแล้ว",
                    state="complete",
                )
                if video_progress is not None:
                    video_progress.empty()
            except (OSError, RuntimeError, ValueError) as error:
                processing_status.update(
                    label="ประมวลผลไม่สำเร็จ",
                    state="error",
                )
                if video_progress is not None:
                    video_progress.empty()
                st.error(f"ประมวลผลไม่สำเร็จ: {error}")

        violations = st.session_state.get("violations")
        if st.session_state.get("processed_key") == media_key and violations is not None:
            st.subheader("ผลการตรวจจับ")
            sidewalk_result = st.session_state.get("sidewalk_result")
            if source_image is not None and sidewalk_result is not None:
                if sidewalk_result["class_available"]:
                    st.image(
                        cv2.cvtColor(sidewalk_result["image"], cv2.COLOR_BGR2RGB),
                        caption=(
                            "กรอบ ROI จำลองเขตทางเท้า "
                            f"(Y-Top={roi_top:.2f}, Height={roi_height:.2f})"
                        ),
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        "ไม่สามารถสร้างกรอบ ROI เขตทางเท้าสำหรับภาพนี้ได้ "
                        "ภาพด้านบนเป็นภาพต้นฉบับ"
                    )

            if st.session_state.get("ocr_error"):
                st.warning(
                    f"อ่านข้อความป้ายทะเบียนอัตโนมัติไม่สำเร็จ: "
                    f"{st.session_state['ocr_error']} — สามารถกรอกป้ายทะเบียนเองด้านล่างได้"
                )

            if not violations:
                st.warning(
                    "ไม่พบรถจักรยานยนต์ในเขต ROI ที่กำหนด กรุณาขยับ Slider "
                    "'ตำแหน่งขอบบนทางเท้า (Y-Top)' หรือ Height ให้ครอบคลุมตัวรถ "
                    "ระบบจะตรวจซ้ำอัตโนมัติเมื่อปรับค่า"
                )
            else:
                st.success(f"พบรายการที่เข้าเงื่อนไข {len(violations)} รายการ")
                report_violations = []
                for index, violation in enumerate(violations, start=1):
                    captured_at = violation["captured_at"].strftime("%Y-%m-%d %H:%M:%S")
                    st.markdown(
                        f"**รายการ {index}** · เวลา {captured_at} · "
                        f"ความมั่นใจ {violation['confidence']:.1%}"
                    )
                    st.image(
                        cv2.cvtColor(violation["evidence_image"], cv2.COLOR_BGR2RGB),
                        caption=(
                            "ภาพหลักฐานหลัก — รถคันใหญ่ที่สุด"
                            if index == 1
                            else f"ภาพหลักฐาน {index}"
                        ),
                        use_container_width=True,
                    )
                    plate_image = violation.get("plate_image")
                    if isinstance(plate_image, np.ndarray) and plate_image.size:
                        st.image(
                            cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB),
                            caption=f"ภาพ Crop ป้ายทะเบียน {index}",
                            width=320,
                        )
                    st.caption(
                        f"ข้อความ OCR: {violation.get('plate_ocr_text') or 'ไม่พบข้อความ'}"
                    )
                    plate_number = str(violation.get("plate_number") or "")
                    default_offense = OFFENSE_OPTIONS[0]
                    offense_key = f"offense_{media_key}_{index}"
                    if offense_key not in st.session_state:
                        st.session_state[offense_key] = default_offense
                    offense = st.selectbox(
                        f"เลือก/แก้ไขข้อหาความผิด รายการ {index}",
                        OFFENSE_OPTIONS,
                        key=offense_key,
                    )
                    if offense == NO_VIOLATION:
                        history_id = violation.get("_history_id")
                        st.session_state["detection_history"][:] = [
                            item
                            for item in st.session_state["detection_history"]
                            if item["record_id"] != history_id
                        ]
                        st.info(f"ตัดรายการที่ {index} ออกจากประวัติและรายงานแล้ว")
                        continue

                    if offense == OFFENSE_OPTIONS[1] and plate_number == UNREADABLE_PLATE:
                        plate_number = "ไม่ติดแผ่นป้ายทะเบียน/ป้ายไม่ชัดเจน"
                    if offense == OFFENSE_OPTIONS[2]:
                        offense = "ไม่สวมหมวกนิรภัย"
                    elif offense == OFFENSE_OPTIONS[0]:
                        offense = "ขับขี่รถจักรยานยนต์บนทางเท้า"
                    elif offense == OFFENSE_OPTIONS[3]:
                        offense = "ฝ่าฝืนสัญญาณไฟจราจร"

                    plate_number = st.text_input(
                        f"ป้ายทะเบียน รายการ {index}",
                        value=plate_number or UNREADABLE_PLATE,
                        key=f"plate_{media_key}_{index}",
                    )
                    human_verified = st.checkbox(
                        f"ยืนยัน/แก้ไขป้ายทะเบียนโดยผู้แจ้ง รายการ {index}",
                        key=f"plate_verified_{media_key}_{index}",
                    )
                    plate_source = (
                        "ยืนยัน/แก้ไขโดยผู้แจ้ง"
                        if human_verified
                        or plate_number.strip()
                        != str(violation.get("plate_number") or "").strip()
                        else violation.get("plate_source", "สแกนด้วย AI")
                    )
                    history_id = violation.get("_history_id")
                    for history_item in st.session_state["detection_history"]:
                        if history_item["record_id"] == history_id:
                            history_item["plate_number"] = plate_number.strip()
                            history_item["location"] = location
                            history_item["offense"] = offense
                            history_item["plate_source"] = plate_source
                            break
            if not violations:
                report_violations = []

            st.subheader("ประวัติการตรวจจับย้อนหลัง")
            _show_history()
            report_violations = _reportable_history()
            report_key = sha256(
                (
                    f"{media_key}|{reporter_name}|{bank_name}|{bank_account}|"
                    f"{[(item['plate_number'], item['location'], item['offense']) for item in report_violations]}"
                ).encode("utf-8")
            ).hexdigest()

            if not report_violations:
                st.warning("ไม่พบรายการกระทำความผิดสำหรับออกรายงาน PDF")
            elif st.button(
                "สร้างรายงาน PDF",
                type="primary",
                disabled=duplicate_evidence,
            ):
                try:
                    st.session_state["report_pdf"] = ReportGenerator().generate_report(
                        report_violations,
                        location=location.strip(),
                        reporter_name=reporter_name.strip(),
                        bank_name=bank_name,
                        bank_account=bank_account.strip(),
                    )
                    st.session_state["report_key"] = report_key
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    st.error(f"สร้างรายงานไม่สำเร็จ: {error}")

            if (
                st.session_state.get("report_key") == report_key
                and "report_pdf" in st.session_state
            ):
                st.download_button(
                    "ดาวน์โหลดรายงาน PDF",
                    data=st.session_state["report_pdf"],
                    file_name="traffic_violation_report.pdf",
                    mime="application/pdf",
                )

    else:
        st.info("อัปโหลดไฟล์ภาพหรือวิดีโอเพื่อเริ่มต้น")
        if st.session_state["detection_history"]:
            st.subheader("ประวัติการตรวจจับย้อนหลัง")
            _show_history()

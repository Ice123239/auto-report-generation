# Traffic Violation Report Generator

Streamlit application for processing image and video evidence, detecting motorcycle
helmet violations with YOLOv8, recognizing plate text with EasyOCR, and exporting
a PDF report.

## Setup

1. Install the dependencies with `pip install -r requirements.txt`.
2. Run the application with `streamlit run main.py`.
3. Place the YOLOv8 weights file named `yolov8n.pt` in the project working
   directory; its path is locked in the sidebar. Then upload an image or video;
   the app previews the sidewalk ROI and starts processing automatically. GPS
   coordinates are read from image EXIF
   when available; address lookup is optional and requires opting in to send those
   coordinates to OpenStreetMap's Nominatim service. Without EXIF GPS, the location
   defaults to "ถนนสุขุมวิท เขตวัฒนา กรุงเทพมหานคร" and can be edited before processing.

The detector uses COCO class ID 3 (motorcycle) and a simulated full-width rectangular
sidewalk ROI. Set its Y-Top and Height with the sidebar sliders. Vehicles whose
bounding-box center falls inside the rectangle are marked as sidewalk violations.
This is a geometric demo heuristic, not scene understanding; validate the ROI for each
fixed camera. The model filters motorcycle detections smaller than 2% of the image
area and lists qualifying vehicles from largest to smallest. The model must also
include a `no_helmet` (or `without helmet`) class to detect helmet violations. YOLO
and EasyOCR models are cached as Streamlit resources.
The confidence slider defaults to `0.20`; for close-up test images, try values from
`0.15` to `0.20`. Lower thresholds may increase false detections.

For motorcycle detections, the detector first searches the lower-middle part of the
vehicle bounding box for a white rectangular plate and falls back to that region if
no likely plate is found. EasyOCR reads candidate plate text with Thai and English
models. Review and correct the editable result before downloading the PDF;
confidence at or below 30% defaults to
"ไม่ติดแผ่นป้ายทะเบียน/ไม่ชัดเจน". The plate crop and OCR text are shown with each
detection. OCR enlarges the crop 3x and applies grayscale CLAHE contrast enhancement.
Detection history labels plates as AI-scanned or reporter-confirmed/
edited. Edit the plate and then select "สร้างรายงาน PDF". Select
"ไม่ติดแผ่นป้ายทะเบียน" only when a missing plate is visibly confirmed; the PDF then
includes the additional offense. Detection history, including processing time,
location, plate, and offense, is retained in the current Streamlit session.
The offense can also be selected or confirmed per vehicle in the results. Location,
plate, and offense are editable directly in the history table; changes are kept in the
current session and used in the PDF.
Rows can also be added or deleted in the history editor. Entries marked as
"ไม่พบการกระทำผิด (ไม่นำลงรายงาน)" are excluded from the history and PDF. The
history page can create a PDF from the latest edited records and export the table
as UTF-8 CSV. For video uploads, sampled detections with overlapping vehicle boxes
are consolidated by keeping the sharpest plate crop for OCR.
Reporter name and bank account are optional and included only in the downloaded PDF;
they are not stored in detection history.

When image EXIF provides GPS coordinates, the PDF includes a clickable Google Maps
link and a QR code that opens the same location. Evidence images receive a timestamp/GPS watermark and a SHA-256 hash of the
watermarked PNG bytes supplied to the PDF. This hash can detect changes to that
image representation, but does not independently establish when, where, or by whom
the original image was captured.
Images without EXIF capture time prompt for the event date and time, and the PDF
states whether each timestamp came from EXIF, the user, or video processing. The app
warns and disables processing/report generation if the same source-file SHA-256 is
already present in the current session history. The report includes a reporter
certification statement and signature block.
For Thai or other non-Latin text in PDFs, `ReportGenerator` uses an available
Unicode TrueType font (Tahoma/Leelawadee on Windows); pass `font_path` when the
font cannot be found automatically.
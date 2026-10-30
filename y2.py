import cv2
import torch
from ultralytics import YOLO
import threading
import time
import queue
import os
import json
import base64
import re
import ssl
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
from dotenv import load_dotenv
import random
import numpy as np
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

# Load environment variables
load_dotenv()

CONFIG = {
    "YOLO_MODEL": "C:/Users/Sahil/Documents/ai/best.pt",
    "FRAME_QUEUE_SIZE": 10,
    "HEALTH_ANALYSIS_INTERVAL": 30,
    "HEALTH_ANALYSIS_COOLDOWN": 300,
    "AI_MAX_ATTEMPTS": 3,
    "AI_RETRY_BASE_SECONDS": 2,
    "AI_RETRY_MAX_SECONDS": 30,
    "AI_REQUEST_TIMEOUT_SECONDS": 60,
    "GROQ_API_URL": "https://api.groq.com/openai/v1/chat/completions",
    "REPORT_SERVER_PORT": 8765,
    "ALERT_COOLDOWN": 30,
    "FPS_WINDOW": 30,
    "DETECTION_THRESHOLD": 0.7,
    "HEALTH_ANALYSIS_THRESHOLD": 0.8,
    "yolo_resolution": 768,
    "TRACKER_TYPE": "bytetrack",
    "DISPLAY_SIZES": {
        "ip": (1280, 720),
        "source": (1280, 720),
        "video": (960, 540)
    },
    "CAMERAS": [
        os.getenv("PRIMARY_CAMERA_1"),
        os.getenv("PRIMARY_CAMERA_2")
    ],
    "SOURCE_CAMERA_INDEX": 0,
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
    "GROQ_MODEL": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
    "SMS_GATEWAY_URL": os.getenv(
        "SMS_GATEWAY_URL",
        "http://10.204.13.191:8080/message"
    ),
    "SMS_GATEWAY_USERNAME": os.getenv("SMS_GATEWAY_USERNAME", ""),
    "SMS_GATEWAY_PASSWORD": os.getenv("SMS_GATEWAY_PASSWORD", ""),
    "SMS_GATEWAY_CA_BUNDLE": os.getenv("SMS_GATEWAY_CA_BUNDLE", ""),
    "ALERT_PHONE_NUMBER": os.getenv("ALERT_PHONE_NUMBER", ""),
    "SMS_GATEWAY_TIMEOUT_SECONDS": 15,
    "ALERT_NOTIFICATION_QUEUE_SIZE": 50,
    "VIDEO_FILE": "C:/Users/Sahil/Videos/Timeline 1.mp4",
    "BOX_THICKNESS": 4,
    "LABEL_FONT_SCALE": 1.5,
    "LABEL_FONT_THICKNESS": 4,
    "INFO_FONT_SCALE": 1.5,
    "INFO_FONT_THICKNESS": 4,
}

if CONFIG["GROQ_API_KEY"]:
    print(f"Groq vision analysis ready ({CONFIG['GROQ_MODEL']}).")
else:
    print("Warning: GROQ_API_KEY not found. Add it to .env to enable health reports.")


def sms_gateway_configuration_error():
    required_settings = (
        "SMS_GATEWAY_URL",
        "SMS_GATEWAY_USERNAME",
        "SMS_GATEWAY_PASSWORD",
        "ALERT_PHONE_NUMBER",
    )
    if not all(CONFIG[name] for name in required_settings):
        return "Set SMS_GATEWAY_URL, SMS_GATEWAY_USERNAME, SMS_GATEWAY_PASSWORD, and ALERT_PHONE_NUMBER."

    gateway_url = urlparse(CONFIG["SMS_GATEWAY_URL"])
    if gateway_url.scheme not in ("http", "https") or not gateway_url.netloc:
        return "SMS_GATEWAY_URL must be a valid HTTP or HTTPS URL."
    if not re.fullmatch(r"\+[1-9]\d{7,14}", CONFIG["ALERT_PHONE_NUMBER"]):
        return "ALERT_PHONE_NUMBER must use international format, for example +911234567890."
    return None


SMS_GATEWAY_ERROR = sms_gateway_configuration_error()
if SMS_GATEWAY_ERROR:
    if any(CONFIG[name] for name in (
        "SMS_GATEWAY_URL",
        "SMS_GATEWAY_USERNAME",
        "SMS_GATEWAY_PASSWORD",
        "ALERT_PHONE_NUMBER",
    )):
        print(f"Warning: SMS alerts are disabled. {SMS_GATEWAY_ERROR}")
else:
    print("SMS alerts enabled through the configured HTTPS phone gateway.")


class ReportRequestHandler(BaseHTTPRequestHandler):
    reports_dir = Path(__file__).resolve().parent / "reports"

    def do_GET(self):
        request_path = urlparse(self.path).path
        if request_path == "/api/reports":
            reports = []
            if self.reports_dir.exists():
                for report_path in self.reports_dir.glob("*.txt"):
                    if report_path.is_file():
                        stat = report_path.stat()
                        reports.append({
                            "name": report_path.name,
                            "createdAt": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "sizeBytes": stat.st_size,
                        })
            reports.sort(key=lambda report: report["createdAt"], reverse=True)
            self._send_response(200, json.dumps(reports), "application/json; charset=utf-8")
            return

        prefix = "/api/reports/"
        if request_path.startswith(prefix):
            name = unquote(request_path[len(prefix):])
            report_path = (self.reports_dir / name).resolve()
            if (Path(name).name != name or not name.endswith(".txt")
                    or report_path.parent != self.reports_dir.resolve()):
                self._send_response(400, "Invalid report name", "text/plain; charset=utf-8")
                return
            try:
                report_text = report_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                self._send_response(404, "Report not found", "text/plain; charset=utf-8")
                return
            self._send_response(200, report_text, "text/plain; charset=utf-8")
            return

        self._send_response(404, "Not found", "text/plain; charset=utf-8")

    def _send_response(self, status, body, content_type):
        encoded_body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded_body)))
        self.end_headers()
        self.wfile.write(encoded_body)

    def log_message(self, format_string, *args):
        return


class WildlifeMonitor:
    def __init__(self):
        self.model = YOLO(CONFIG["YOLO_MODEL"])
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        if self.device.startswith("cuda"):
            print(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
        else:
            print(f"GPU acceleration unavailable; using CPU (PyTorch CUDA build: {torch.version.cuda}).")
        self.frames_queue = queue.Queue(maxsize=CONFIG["FRAME_QUEUE_SIZE"])
        self.alert_queue = queue.Queue()
        self.phone_alert_queue = queue.Queue(maxsize=CONFIG["ALERT_NOTIFICATION_QUEUE_SIZE"])
        self.fps_times = []
        self.running = False
        self.cap = None
        self.last_health_analysis = 0
        self.last_health_analysis_by_track = {}
        self.health_analysis_pending = set()
        self.health_analysis_lock = threading.Lock()
        self.health_analysis_queue = queue.Queue(maxsize=1)
        self.report_server = None
        self.last_alert_times = {"weapon": 0, "fire": 0}
        self.track_info = {}
        self.health_analysis_status = {}

    def start(self, source):
        if source == 'ip':
            camera_url = next((url for url in CONFIG["CAMERAS"] if url), None)
            if not camera_url:
                print("Error: No valid IP camera URL found in configuration.")
                return
            self.cap = cv2.VideoCapture(camera_url)
        elif source == 'source':
            self.cap = cv2.VideoCapture(CONFIG["SOURCE_CAMERA_INDEX"])
        elif source == 'video':
            if not os.path.exists(CONFIG["VIDEO_FILE"]):
                print(f"Error: Video file not found at {CONFIG['VIDEO_FILE']}")
                return
            self.cap = cv2.VideoCapture(CONFIG["VIDEO_FILE"])

        if not self.cap or not self.cap.isOpened():
            print("Error: Unable to open video source.")
            return

        if not self.start_report_server():
            self.cap.release()
            self.cap = None
            return

        self.running = True
        threading.Thread(target=self.capture_frames, daemon=True).start()
        threading.Thread(target=self.process_frames, daemon=True).start()
        threading.Thread(target=self.handle_alerts, daemon=True).start()
        threading.Thread(target=self.handle_health_analyses, daemon=True).start()
        if SMS_GATEWAY_ERROR is None:
            threading.Thread(target=self.handle_phone_alerts, daemon=True).start()

    def start_report_server(self):
        try:
            self.report_server = ThreadingHTTPServer(
                ("127.0.0.1", CONFIG["REPORT_SERVER_PORT"]),
                ReportRequestHandler,
            )
        except OSError as error:
            print(f"[Reports] Could not start local report server: {error}")
            return False

        threading.Thread(target=self.report_server.serve_forever, daemon=True).start()
        print(f"[Reports] Local report API available at http://127.0.0.1:{CONFIG['REPORT_SERVER_PORT']}/api/reports")
        return True

    def capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.cap.get(cv2.CAP_PROP_POS_AVI_RATIO), float):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("Error: Unable to read frame from source.")
                    break
            if not self.frames_queue.full():
                self.frames_queue.put(frame)
            self.fps_times.append(time.time())
            self.fps_times = self.fps_times[-CONFIG["FPS_WINDOW"]:]

    def process_frames(self):
        cv2.namedWindow("Wildlife Monitoring", cv2.WINDOW_NORMAL)
        tracker_enabled = self.initialize_tracker()
        while self.running:
            try:
                frame = self.frames_queue.get(timeout=1)
                if frame is None:
                    continue

                results = self.model.track(
                    frame,
                    device=self.device,
                    conf=CONFIG["DETECTION_THRESHOLD"],
                    persist=True,
                    tracker=f"{CONFIG['TRACKER_TYPE']}.yaml",
                    imgsz=CONFIG["yolo_resolution"]
                ) if tracker_enabled else self.model(
                    frame,
                    device=self.device,
                    conf=CONFIG["DETECTION_THRESHOLD"],
                    imgsz=CONFIG["yolo_resolution"]
                )

                self.handle_detections(frame, results)
                self.display_frame(frame)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

    def initialize_tracker(self):
        if CONFIG["TRACKER_TYPE"] in ["bytetrack", "botsort"]:
            try:
                if hasattr(self.model, "predictor") and hasattr(self.model.predictor, "callbacks"):
                    self.model.predictor.callbacks.clear()
                return True
            except Exception as e:
                print(f"Warning: Tracker initialization failed. Using detection only mode. Error: {e}")
                return False
        return False

    def handle_detections(self, frame, results):
        now = time.time()
        health_frame = None
        if CONFIG["GROQ_API_KEY"]:
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    label = self.model.names[cls_id]
                    if (label.lower() in ['tiger', 'rhino', 'rhinoceros'] and
                            float(box.conf) > CONFIG["HEALTH_ANALYSIS_THRESHOLD"]):
                        health_frame = frame.copy()
                        break
                if health_frame is not None:
                    break

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = self.model.names[cls_id]
                if conf < CONFIG["DETECTION_THRESHOLD"]:
                    continue

                track_id = int(getattr(box, 'id', random.randint(10000, 99999)))
                self.track_info[track_id] = now

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = self.get_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, CONFIG["BOX_THICKNESS"])
                cv2.putText(frame, f"{label} #{track_id} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, CONFIG["LABEL_FONT_SCALE"], color, CONFIG["LABEL_FONT_THICKNESS"])

                alert_label = label.strip().lower()
                if alert_label in ("weapon", "fire") and now - self.last_alert_times[alert_label] > CONFIG["ALERT_COOLDOWN"]:
                    self.last_alert_times[alert_label] = now
                    self.alert_queue.put((alert_label, frame.copy(), track_id))

                if (label.lower() in ['tiger', 'rhino', 'rhinoceros'] and
                        conf > CONFIG["HEALTH_ANALYSIS_THRESHOLD"] and
                        health_frame is not None):
                    self.queue_health_analysis(
                        health_frame, label, track_id, (x1, y1, x2, y2), now
                    )

    def queue_health_analysis(self, frame, label, track_id, bounds, now):
        if not CONFIG["GROQ_API_KEY"]:
            return

        with self.health_analysis_lock:
            if track_id in self.health_analysis_pending:
                return
            if now - self.last_health_analysis < CONFIG["HEALTH_ANALYSIS_INTERVAL"]:
                return
            last_success = self.last_health_analysis_by_track.get(track_id, 0)
            if now - last_success < CONFIG["HEALTH_ANALYSIS_COOLDOWN"]:
                return
            try:
                self.health_analysis_queue.put_nowait(
                    (frame.copy(), label, track_id, bounds)
                )
            except queue.Full:
                return

            self.health_analysis_pending.add(track_id)
            self.last_health_analysis = now
            print(f"[Groq] Queued {label} #{track_id} for health analysis.")

    def handle_health_analyses(self):
        while self.running or not self.health_analysis_queue.empty():
            try:
                frame, label, track_id, bounds = self.health_analysis_queue.get(timeout=1)
            except queue.Empty:
                continue

            started_at = time.monotonic()
            try:
                succeeded = self.analyze_health(frame, label, track_id, bounds)
                if succeeded:
                    with self.health_analysis_lock:
                        self.last_health_analysis_by_track[track_id] = time.time()
            except Exception as error:
                self.health_analysis_status[track_id] = (label, time.time(), "Error")
                print(f"[Groq] Unexpected analysis error for {label} #{track_id}: {error}")
            finally:
                elapsed = time.monotonic() - started_at
                with self.health_analysis_lock:
                    self.health_analysis_pending.discard(track_id)
                print(f"[Groq] Analysis finished for {label} #{track_id} in {elapsed:.1f}s.")
                self.health_analysis_queue.task_done()

    def handle_alerts(self):
        os.makedirs("alerts", exist_ok=True)
        while self.running:
            try:
                threat_type, frame, track_id = self.alert_queue.get(timeout=1)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"alerts/{threat_type}_{timestamp}_ID{track_id}.jpg"
                cv2.imwrite(filename, frame)
                print(f"\n🚨 ALERT: {threat_type.upper()} DETECTED 🚨\nImage saved: {filename}\n")
                if SMS_GATEWAY_ERROR is None:
                    try:
                        self.phone_alert_queue.put_nowait((threat_type, track_id, filename))
                    except queue.Full:
                        print(f"[SMS] Notification queue full; {threat_type} alert for #{track_id} was not sent.")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error handling alert: {e}")

    def handle_phone_alerts(self):
        while self.running or not self.phone_alert_queue.empty():
            try:
                threat_type, track_id, image_filename = self.phone_alert_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                self.send_phone_alert(threat_type, track_id, image_filename)
            except Exception as error:
                print(f"[SMS] Unexpected notification error for {threat_type} #{track_id}: {error}")
            finally:
                self.phone_alert_queue.task_done()

    def send_phone_alert(self, threat_type, track_id, image_filename):
        message = (
            f"ALERT: {threat_type.upper()} detected by Wildlife Monitor. "
            f"ID: {track_id}. Time: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}."
        )
        # SMSGate API request format
        request_body = json.dumps({
            "phoneNumbers": [CONFIG["ALERT_PHONE_NUMBER"]],
            "textMessage": {
                "text": message
            },
        }).encode("utf-8")
        credentials = base64.b64encode(
            f"{CONFIG['SMS_GATEWAY_USERNAME']}:{CONFIG['SMS_GATEWAY_PASSWORD']}".encode("utf-8")
        ).decode("ascii")
        request = Request(
            CONFIG["SMS_GATEWAY_URL"],
            data=request_body,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/json",
                "User-Agent": "WildlifeMonitor/1.0",
            },
            method="POST",
        )
        tls_context = ssl.create_default_context(
            cafile=CONFIG["SMS_GATEWAY_CA_BUNDLE"] or None
        )

        try:
            if urlparse(CONFIG["SMS_GATEWAY_URL"]).scheme == "https":
                response_context = tls_context
            else:
                response_context = None

            with urlopen(
                request,
                timeout=CONFIG["SMS_GATEWAY_TIMEOUT_SECONDS"],
                context=response_context,
            ) as response:
                if not 200 <= response.status < 300:
                    raise RuntimeError(f"Gateway returned HTTP {response.status}")
        except HTTPError as error:
            raise RuntimeError(f"Gateway returned HTTP {error.code}") from error
        except URLError as error:
            raise RuntimeError(f"Could not reach the phone gateway: {error.reason}") from error

        print(f"[SMS] Gateway accepted {threat_type.upper()} alert for #{track_id} (image: {image_filename}).")

    def analyze_health(self, frame, label, track_id, bounds):
        try:
            if not CONFIG["GROQ_API_KEY"]:
                print("[Groq] Client is not configured; check GROQ_API_KEY.")
                return False

            self.health_analysis_status[track_id] = (label, time.time(), "Analyzing...")

            if frame is None:
                self.health_analysis_status[track_id] = (label, time.time(), "Frame error")
                return False

            frame_height, frame_width = frame.shape[:2]
            x1, y1, x2, y2 = bounds
            padding = max(8, int(max(x2 - x1, y2 - y1) * 0.15))
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(frame_width, x2 + padding)
            crop_y2 = min(frame_height, y2 + padding)
            animal_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1 or animal_crop.size == 0:
                print(f"[Groq] Invalid crop for {label} #{track_id}; using the full frame.")
                animal_crop = frame

            success, encoded_img = cv2.imencode('.jpg', animal_crop)
            if not success:
                self.health_analysis_status[track_id] = (label, time.time(), "Encoding failed")
                return False

            image_data = base64.b64encode(encoded_img.tobytes()).decode("ascii")
            request_body = json.dumps({
                "model": CONFIG["GROQ_MODEL"],
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Analyze this {label} (ID:{track_id}) for:\n"
                                "1. Health status\n2. Visible injuries\n3. Behavior\n"
                                "4. Recommendations\n5. Unique features\n\n"
                                "Be cautious: distinguish observations from uncertainty, "
                                "and do not infer medical diagnoses from an unclear image."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                            },
                        },
                    ],
                }],
                "max_completion_tokens": 1024,
            }).encode("utf-8")

            response_data = None
            for attempt in range(1, CONFIG["AI_MAX_ATTEMPTS"] + 1):
                try:
                    print(
                        f"[Groq] Submitting {label} #{track_id} "
                        f"(attempt {attempt}/{CONFIG['AI_MAX_ATTEMPTS']})..."
                    )
                    request = Request(
                        CONFIG["GROQ_API_URL"],
                        data=request_body,
                        headers={
                            "Authorization": f"Bearer {CONFIG['GROQ_API_KEY']}",
                            "Content-Type": "application/json",
                            "User-Agent": "WildlifeMonitor/1.0",
                        },
                        method="POST",
                    )
                    with urlopen(
                        request,
                        timeout=CONFIG["AI_REQUEST_TIMEOUT_SECONDS"],
                    ) as api_response:
                        response_data = json.loads(api_response.read().decode("utf-8"))
                    report_text = (
                        response_data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content")
                    )
                    break
                except HTTPError as error:
                    error_body = error.read().decode("utf-8", errors="replace")
                    status_code = self.get_http_status(error)
                    retryable = status_code == 429 or (
                        status_code is not None and 500 <= status_code <= 599
                    )
                    if not retryable or attempt == CONFIG["AI_MAX_ATTEMPTS"]:
                        raise RuntimeError(
                            f"Groq HTTP {status_code}: {error_body or error.reason}"
                        ) from error

                    delay = self.retry_delay(attempt, error)
                    print(
                        f"[Groq] Transient HTTP {status_code} for {label} #{track_id}; "
                        f"retrying in {delay:.1f}s."
                    )
                    time.sleep(delay)
                except Exception as error:
                    if not isinstance(error, (URLError, TimeoutError, ConnectionError)):
                        raise

                    if attempt == CONFIG["AI_MAX_ATTEMPTS"]:
                        raise
                    delay = self.retry_delay(attempt)
                    print(
                        f"[Groq] Network error for {label} #{track_id}: {error}; "
                        f"retrying in {delay:.1f}s."
                    )
                    time.sleep(delay)

            if report_text:
                self.health_analysis_status[track_id] = (label, time.time(), "✅ Done")
                print(f"\n🦓 {label.upper()} #{track_id} Health Report:\n{report_text}\n{'='*60}")

                reports_dir = Path(__file__).resolve().parent / "reports"
                reports_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = reports_dir / f"{label}_{timestamp}_ID{track_id}.txt"
                with report_path.open("w", encoding="utf-8") as f:
                    f.write(f"Health Report for {label} #{track_id} - {timestamp}\n\n")
                    f.write(report_text)
                print(f"[✔] Report saved to {report_path}")
                return True
            else:
                self.health_analysis_status[track_id] = (label, time.time(), "No response")
                print(f"[Groq] No health report text returned for {label} #{track_id}")
                return False

        except Exception as e:
            self.health_analysis_status[track_id] = (label, time.time(), "Error")
            print(f"[Groq] Health analysis failed for {label} #{track_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def retry_delay(attempt, error=None):
        backoff = (
            CONFIG["AI_RETRY_BASE_SECONDS"] * (2 ** (attempt - 1))
            + random.uniform(0, 1)
        )
        return max(
            min(CONFIG["AI_RETRY_MAX_SECONDS"], backoff),
            WildlifeMonitor.get_retry_after(error) if error else 0,
        )

    @staticmethod
    def get_http_status(error):
        try:
            status_code = int(getattr(error, "code", None))
        except (TypeError, ValueError):
            return None
        return status_code if 100 <= status_code <= 599 else None

    @staticmethod
    def get_retry_after(error):
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None) or getattr(error, "headers", None)
        if headers is None:
            return 0
        retry_after = headers.get("Retry-After", 0)
        try:
            return max(0, float(retry_after))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(retry_after)
                now = datetime.now(retry_at.tzinfo) if retry_at.tzinfo else datetime.now()
                return max(0, (retry_at - now).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return 0

    def get_color(self, track_id):
        np.random.seed(track_id)
        return tuple(int(x) for x in np.random.randint(0, 255, 3))

    def display_frame(self, frame):
        height, width = CONFIG["DISPLAY_SIZES"].get("source", (1280, 720))
        resized = cv2.resize(frame, (width, height))
        fps = self.calculate_fps()
        cv2.putText(resized, f"FPS: {fps:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    CONFIG["INFO_FONT_SCALE"], (0, 255, 0), CONFIG["INFO_FONT_THICKNESS"])

        now = time.time()
        for track_id, (label, ts, status) in self.health_analysis_status.items():
            if now - ts > 10:
                continue
            message = f"{label} #{track_id}: {status}"
            y_offset = 80 + 30 * (track_id % 5)
            cv2.putText(resized, message, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        CONFIG["INFO_FONT_SCALE"] * 0.8, (0, 200, 255), CONFIG["INFO_FONT_THICKNESS"])

        cv2.imshow("Wildlife Monitoring", resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop()

    def calculate_fps(self):
        if len(self.fps_times) < 2:
            return 0.0
        return (len(self.fps_times) - 1) / (self.fps_times[-1] - self.fps_times[0])

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.report_server:
            self.report_server.shutdown()
            self.report_server.server_close()
            self.report_server = None
        cv2.destroyAllWindows()
        print("Monitoring system stopped.")

if __name__ == "__main__":
    monitor = WildlifeMonitor()
    print("\nWildlife Monitoring System")
    print(f"PyTorch CUDA build: {torch.version.cuda or 'CPU-only'}")

    while True:
        choice = input("Select input source:\n1. IP Camera \n2. Source Camera \n3. Video File\n4. Quit\n> ").strip().lower()
        if choice in ['4', 'q']:
            break
        elif choice in ['1', 'c']:
            monitor.start('ip')
            break
        elif choice in ['2', 's']:
            monitor.start('source')
            break
        elif choice in ['3', 'v']:
            monitor.start('video')
            break
        else:
            print("Invalid choice. Please try again.")

    try:
        while monitor.running:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()

import cv2
import numpy as np
from pyzbar import pyzbar
import json
import time
from datetime import datetime
import requests
import logging
from ArducamPTZ import ArducamPTZ
import hashlib
from collections import defaultdict

# -------------------- Logging Setup -------------------- #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- PTZ Controller -------------------- #
class PTZController:
    def __init__(self):
        self.ptz = ArducamPTZ()
        self.ptz.init()
        self.ptz.reset()
        logger.info("PTZ initialized and centered")

        # Optimized for Quarter Barrel kegs (16⅛" wide) in 2x3 arrangement
        # QR codes on TOP of kegs - need overhead view angle
        # Pallet dimensions: ~32" x 48" (2 kegs x 3 kegs)
        self.positions = {
            1: {'pan': 90, 'tilt': 45, 'zoom': 2.2},   # Bottom pallet (2m) - steeper angle for top view
            2: {'pan': 90, 'tilt': 55, 'zoom': 1.6},   # Middle pallet - moderate angle
            3: {'pan': 90, 'tilt': 70, 'zoom': 1.0}    # Top pallet - wide FOV, steeper for top view
        }

    def move_to_level(self, level):
        """Move camera to optimal position for Quarter Barrel 2x3 pallet"""
        if level not in self.positions:
            logger.error(f"Invalid level: {level}")
            return False
        
        pos = self.positions[level]
        logger.info(f"Moving to Quarter Barrel pallet level {level} (2x3 arrangement)...")
        
        # Smooth movement sequence
        self.ptz.set_pan_angle(pos['pan'])
        time.sleep(0.8)
        self.ptz.set_tilt_angle(pos['tilt'])
        time.sleep(0.8)
        self.ptz.set_zoom(pos['zoom'])
        time.sleep(2.5)  # Extra stabilization time for zoom
        
        logger.info(f"Camera positioned: pan={pos['pan']}, tilt={pos['tilt']}, zoom={pos['zoom']}")
        logger.info(f"Optimized for top-view QR scanning of 2x3 keg arrangement")
        return True

    def fine_tune_position(self, pan_offset=0, tilt_offset=0, zoom_offset=0):
        """Manual position adjustment during scanning"""
        try:
            if pan_offset != 0:
                current_pan = self.ptz.get_pan_angle()
                self.ptz.set_pan_angle(max(0, min(180, current_pan + pan_offset)))
                
            if tilt_offset != 0:
                current_tilt = self.ptz.get_tilt_angle()
                self.ptz.set_tilt_angle(max(0, min(180, current_tilt + tilt_offset)))
                
            if zoom_offset != 0:
                current_zoom = self.ptz.get_zoom()
                self.ptz.set_zoom(max(1.0, min(5.0, current_zoom + zoom_offset)))
            
            time.sleep(1.5)
            logger.info(f"Position adjusted: pan_offset={pan_offset}, tilt_offset={tilt_offset}, zoom_offset={zoom_offset}")
        except Exception as e:
            logger.error(f"Fine-tune failed: {e}")

# -------------------- QR Code Detector -------------------- #
class QRDetector:
    def __init__(self, expected_qr_count=6):
        self.expected_qr_count = expected_qr_count

    def detect_qr_codes(self, frame):
        """Enhanced QR detection optimized for top-mounted QR codes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multiple enhancement techniques for various lighting conditions
        enhanced_frames = []
        
        # 1. CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
        enhanced_frames.append(clahe.apply(gray))
        
        # 2. Gaussian blur + adaptive threshold (good for uneven lighting)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        enhanced_frames.append(adaptive_thresh)
        
        # 3. Histogram equalization
        equalized = cv2.equalizeHist(gray)
        enhanced_frames.append(equalized)
        
        # 4. Morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        enhanced_frames.append(morph)
        
        all_qr_codes = []
        
        # Try detection on each enhanced frame
        for i, enhanced in enumerate(enhanced_frames):
            try:
                qr_codes = pyzbar.decode(enhanced, symbols=[pyzbar.ZBarSymbol.QRCODE])
                if qr_codes:
                    logger.debug(f"Enhancement method {i+1} found {len(qr_codes)} QR codes")
                all_qr_codes.extend(qr_codes)
            except Exception as e:
                logger.debug(f"Enhancement method {i+1} failed: {e}")
                continue
        
        # Remove duplicates and format results
        unique_qrs = {}
        for qr in all_qr_codes:
            try:
                data = qr.data.decode('utf-8')
                if data and data not in unique_qrs:
                    points = qr.polygon
                    if len(points) == 4:
                        center = (
                            int(sum(p.x for p in points) / 4),
                            int(sum(p.y for p in points) / 4)
                        )
                        
                        # Calculate QR code area for quality assessment
                        area = cv2.contourArea(np.array([(p.x, p.y) for p in points]))
                        
                        unique_qrs[data] = {
                            'data': data,
                            'center': center,
                            'polygon': [(p.x, p.y) for p in points],
                            'timestamp': datetime.now().isoformat(),
                            'area': area,
                            'quality_score': min(100, area / 100)  # Simple quality metric
                        }
            except (UnicodeDecodeError, AttributeError) as e:
                logger.debug(f"Failed to decode QR: {e}")
                continue
        
        # Sort by quality score (larger, clearer QR codes first)
        sorted_qrs = sorted(unique_qrs.values(), key=lambda x: x['quality_score'], reverse=True)
        
        logger.debug(f"Total unique QR codes found: {len(sorted_qrs)}")
        return sorted_qrs

    def draw_qr_codes(self, frame, qr_codes):
        """Draw QR detection results with 2x3 grid visualization"""
        height, width = frame.shape[:2]
        
        # Draw 2x3 grid overlay to show expected keg positions
        grid_color = (100, 100, 100)  # Gray
        
        # Vertical lines (3 columns)
        for i in range(1, 3):
            x = int(width * i / 3)
            cv2.line(frame, (x, 0), (x, height), grid_color, 1)
        
        # Horizontal line (2 rows)
        y = int(height / 2)
        cv2.line(frame, (0, y), (width, y), grid_color, 1)
        
        # Draw detected QR codes
        for i, qr in enumerate(qr_codes):
            # Color coding based on detection order
            colors = [(0,255,0), (0,255,255), (255,0,255), (255,255,0), (0,128,255), (255,128,0)]
            color = colors[i % len(colors)]
            
            # Draw polygon
            pts = np.array(qr['polygon'], dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 3)
            
            # Draw center point
            cv2.circle(frame, qr['center'], 8, color, -1)
            
            # Draw QR data and position number
            text = f"{i+1}: {qr['data'][:12]}..."
            cv2.putText(frame, text, (qr['center'][0]-60, qr['center'][1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw quality indicator
            quality = int(qr.get('quality_score', 0))
            cv2.putText(frame, f"Q:{quality}", (qr['center'][0]-20, qr['center'][1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame

# -------------------- Duplicate Prevention -------------------- #
class DuplicateTracker:
    def __init__(self):
        self.processed_qrs = set()
        self.batch_history = defaultdict(list)
    
    def is_duplicate(self, qr_data):
        return qr_data in self.processed_qrs
    
    def add_qr(self, qr_data, batch_id):
        self.processed_qrs.add(qr_data)
        self.batch_history[batch_id].append(qr_data)
    
    def get_new_qrs(self, qr_codes):
        new_qrs = []
        for qr in qr_codes:
            if not self.is_duplicate(qr['data']):
                new_qrs.append(qr)
        return new_qrs
    
    def get_stats(self):
        return {
            'total_processed': len(self.processed_qrs),
            'batches_created': len(self.batch_history),
            'batch_sizes': {bid: len(qrs) for bid, qrs in self.batch_history.items()}
        }

# -------------------- Main System -------------------- #
class QuarterBarrelQRSystem:
    def __init__(self, camera_id=0, cloud_endpoint="https://your-api.com/qr-upload"):
        self.camera_id = camera_id
        self.cloud_endpoint = cloud_endpoint
        self.cap = None
        self.ptz = PTZController()
        self.qr_detector = QRDetector(expected_qr_count=6)
        self.duplicate_tracker = DuplicateTracker()
        self.max_levels = 3
        self.batch_counter = 0

    def initialize_camera(self):
        """Initialize camera with settings optimized for Jetson Nano"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            logger.error("Camera not detected - check connection")
            return False
        
        # Jetson Nano optimized camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for Jetson Nano stability
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
        
        # Enhanced settings for QR detection
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 60)
        
        # Test camera
        ret, test_frame = self.cap.read()
        if ret:
            logger.info(f"Camera initialized: {test_frame.shape[1]}x{test_frame.shape[0]}")
        else:
            logger.error("Camera test failed")
            return False
        
        return True

    def scan_pallet_level(self, level):
        """Scan Quarter Barrel pallet (2x3 arrangement) for 6 QR codes"""
        logger.info(f"Scanning Quarter Barrel pallet level {level} (2x3 arrangement)")
        
        # Position camera
        if not self.ptz.move_to_level(level):
            return []
        
        collected_qrs = []
        attempts = 0
        max_attempts = 80  # More attempts for thorough scanning
        consecutive_empty_frames = 0
        max_empty_frames = 12
        
        # Camera stabilization
        logger.info("Stabilizing camera for top-view QR scanning...")
        for _ in range(10):
            ret, _ = self.cap.read()
            if ret:
                time.sleep(0.15)
        
        logger.info("Starting QR detection...")
        
        while len(collected_qrs) < 6 and attempts < max_attempts:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Frame capture failed")
                time.sleep(0.1)
                continue

            # Detect QR codes
            qr_codes = self.qr_detector.detect_qr_codes(frame)
            
            if not qr_codes:
                consecutive_empty_frames += 1
                if consecutive_empty_frames >= max_empty_frames:
                    logger.info("No QR codes detected - checking camera angle...")
                    
                    # Auto-adjust for better top view
                    if level == 3 and attempts > 20:  # Top level adjustment
                        logger.info("Adjusting zoom for better FOV coverage...")
                        self.ptz.fine_tune_position(zoom_offset=-0.2)
                    elif level == 1 and attempts > 30:  # Bottom level adjustment
                        logger.info("Adjusting tilt for better top view...")
                        self.ptz.fine_tune_position(tilt_offset=-3)
                    
                    consecutive_empty_frames = 0
                    time.sleep(1)
            else:
                consecutive_empty_frames = 0
                
                # Filter duplicates
                new_qrs = self.duplicate_tracker.get_new_qrs(qr_codes)
                
                # Add new QR codes
                for qr in new_qrs:
                    if not any(existing['data'] == qr['data'] for existing in collected_qrs):
                        qr['level'] = level
                        qr['keg_position'] = self.estimate_keg_position(qr['center'], frame.shape)
                        collected_qrs.append(qr)
                        logger.info(f"✓ QR {len(collected_qrs)}/6: {qr['data'][:15]}... (Pos: {qr['keg_position']})")

            # Visual feedback
            display_frame = self.qr_detector.draw_qr_codes(frame.copy(), collected_qrs)
            
            # Status overlay
            status_lines = [
                f"Quarter Barrel Level {level} - QR Codes: {len(collected_qrs)}/6",
                f"Attempts: {attempts}/{max_attempts}",
                f"2x3 Grid Layout - Top View Scanning",
                "Press 'q'=quit, 's'=skip, 'z'=zoom out, 't'=tilt adjust"
            ]
            
            for i, line in enumerate(status_lines):
                y_pos = 25 + (i * 22)
                color = (0, 255, 0) if len(collected_qrs) == 6 else (0, 255, 255)
                cv2.putText(display_frame, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Progress bar
            progress = len(collected_qrs) / 6
            bar_width = 400
            bar_height = 20
            cv2.rectangle(display_frame, (10, frame.shape[0]-40), 
                         (10 + bar_width, frame.shape[0]-20), (50, 50, 50), -1)
            cv2.rectangle(display_frame, (10, frame.shape[0]-40), 
                         (10 + int(bar_width * progress), frame.shape[0]-20), (0, 255, 0), -1)
            
            cv2.imshow(f"Quarter Barrel Scan - Level {level}", display_frame)
            
            # User controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Scan interrupted by user")
                break
            elif key == ord('s'):
                logger.info("Skipping to next level")
                break
            elif key == ord('z'):
                logger.info("Manual zoom adjustment")
                self.ptz.fine_tune_position(zoom_offset=-0.3)
            elif key == ord('t'):
                logger.info("Manual tilt adjustment")
                self.ptz.fine_tune_position(tilt_offset=-2)

            attempts += 1
            time.sleep(0.25)  # Balanced scanning speed

        cv2.destroyAllWindows()
        
        # Results summary
        success_rate = (len(collected_qrs) / 6) * 100
        logger.info(f"Level {level} complete: {len(collected_qrs)}/6 QR codes ({success_rate:.1f}%)")
        
        if len(collected_qrs) < 6:
            logger.warning("⚠️  Incomplete scan - possible causes:")
            logger.warning("   • QR codes not facing camera")
            logger.warning("   • Poor lighting conditions")
            logger.warning("   • Camera FOV too narrow/wide")
            logger.warning("   • QR codes damaged/dirty")
        
        return collected_qrs

    def estimate_keg_position(self, center, frame_shape):
        """Estimate keg position in 2x3 grid based on QR center"""
        height, width = frame_shape[:2]
        x, y = center
        
        # Determine column (1-3)
        col = 1 if x < width/3 else (2 if x < 2*width/3 else 3)
        
        # Determine row (1-2)
        row = 1 if y < height/2 else 2
        
        return f"R{row}C{col}"

    def create_batch_data(self, level, qr_codes):
        """Create batch data for Quarter Barrel pallets"""
        self.batch_counter += 1
        batch_id = f"QB_L{level}_2x3_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.batch_counter:03d}"
        
        # Mark as processed
        for qr in qr_codes:
            self.duplicate_tracker.add_qr(qr['data'], batch_id)
        
        return {
            'batch_id': batch_id,
            'keg_type': 'Quarter_Barrel',
            'arrangement': '2x3_grid',
            'level': level,
            'sequence_number': self.batch_counter,
            'timestamp': datetime.now().isoformat(),
            'expected_qr_count': 6,
            'actual_qr_count': len(qr_codes),
            'completion_percentage': (len(qr_codes) / 6) * 100,
            'qr_codes': qr_codes,
            'pallet_info': {
                'dimensions': '32x48_inches',
                'keg_count': 6,
                'keg_size': '16.125_inches_diameter'
            },
            'system_info': {
                'camera_id': self.camera_id,
                'platform': 'jetson_nano',
                'ptz_camera': 'arducam_12mp_imx477',
                'scanning_mode': 'top_view_qr'
            }
        }

    def upload_to_cloud(self, batch_data):
        """Upload batch to cloud endpoint"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer YOUR_API_TOKEN',
                'X-Keg-Type': 'Quarter-Barrel',
                'X-Arrangement': '2x3',
                'X-Level': str(batch_data['level'])
            }
            
            logger.info(f"Uploading {batch_data['keg_type']} batch: {batch_data['batch_id']}")
            
            response = requests.post(self.cloud_endpoint, json=batch_data, 
                                   headers=headers, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"✅ Upload successful")
                return True
            else:
                logger.error(f"❌ Upload failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return False

    def save_batch_locally(self, batch_data):
        """Save batch locally as backup"""
        filename = f"{batch_data['batch_id']}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(batch_data, f, indent=2)
            logger.info(f"💾 Saved: {filename}")
            return True
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False

    def run(self):
        """Main execution for Quarter Barrel 2x3 scanning"""
        logger.info("🍺 Quarter Barrel QR Scanning System (2x3 Grid)")
        logger.info("=" * 60)
        
        if not self.initialize_camera():
            return

        try:
            for level in range(1, 4):
                logger.info(f"\n{'='*60}")
                logger.info(f"LEVEL {level} - QUARTER BARREL PALLET (2x3)")
                logger.info(f"{'='*60}")
                
                input(f"\n📦 Position Quarter Barrel pallet at level {level}\n"
                      f"   • 6 kegs in 2×3 arrangement\n"
                      f"   • QR codes on TOP of kegs facing up\n"
                      f"   • Ensure good lighting\n"
                      f"Press Enter to start scanning...")
                
                qr_data = self.scan_pallet_level(level)
                
                if len(qr_data) == 0:
                    logger.error(f"❌ No QR codes found at level {level}")
                    retry = input("Retry? (y/n): ")
                    if retry.lower() == 'y':
                        qr_data = self.scan_pallet_level(level)
                
                if len(qr_data) > 0:
                    batch = self.create_batch_data(level, qr_data)
                    self.save_batch_locally(batch)
                    
                    if self.upload_to_cloud(batch):
                        logger.info(f"✅ Level {level} processed successfully!")
                    else:
                        logger.warning(f"⚠️ Level {level} saved locally, upload failed")
                else:
                    logger.warning(f"⚠️ Skipping level {level} - no data")
                
                if level < 3:
                    time.sleep(3)

            # Final summary
            stats = self.duplicate_tracker.get_stats()
            logger.info(f"\n🎉 SCANNING COMPLETE!")
            logger.info(f"Total QR codes: {stats['total_processed']}")
            logger.info(f"Batches created: {stats['batches_created']}")
            
        except KeyboardInterrupt:
            logger.info("\n⚠️ Interrupted by user")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

# -------------------- Entry Point -------------------- #
def main():
    print("🚀 Quarter Barrel QR Scanner")
    print("📷 Arducam PTZ Camera + Jetson Nano")
    print("📦 2×3 Keg Arrangement")
    print("🔍 Top-View QR Detection")
    print("-" * 40)
    
    system = QuarterBarrelQRSystem(
        camera_id=0,
        cloud_endpoint="https://your-api.com/qr-upload"
    )
    
    system.run()

if __name__ == "__main__":
    main()
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
import re

# -------------------- Logging Setup -------------------- #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- Automatic PTZ Controller -------------------- #
class AutoPTZController:
    def __init__(self):
        self.ptz = ArducamPTZ()
        self.ptz.init()
        self.ptz.reset()
        logger.info("PTZ Camera initialized")

        # Optimized positions for each pallet level
        self.level_positions = {
            1: {'pan': 90, 'tilt': 45, 'zoom': 2.2},   # Bottom level
            2: {'pan': 90, 'tilt': 55, 'zoom': 1.6},   # Middle level  
            3: {'pan': 90, 'tilt': 70, 'zoom': 1.0}    # Top level
        }

        # Scanning pattern offsets for thorough coverage
        self.scan_offsets = [
            {'pan': 0, 'tilt': 0, 'zoom': 0},      # Center
            {'pan': -10, 'tilt': -3, 'zoom': 0.2}, # Left-up
            {'pan': 10, 'tilt': -3, 'zoom': 0.2},  # Right-up
            {'pan': -8, 'tilt': 3, 'zoom': -0.1},  # Left-down
            {'pan': 8, 'tilt': 3, 'zoom': -0.1},   # Right-down
            {'pan': 0, 'tilt': 0, 'zoom': -0.3},   # Wide view
        ]

    def move_to_level(self, level):
        """Move camera to optimal position for pallet level"""
        if level not in self.level_positions:
            return False
        
        pos = self.level_positions[level]
        logger.info(f"Moving to pallet level {level}")
        
        self.ptz.set_pan_angle(pos['pan'])
        time.sleep(0.8)
        self.ptz.set_tilt_angle(pos['tilt'])
        time.sleep(0.8)
        self.ptz.set_zoom(pos['zoom'])
        time.sleep(2.0)
        
        return True

    def scan_positions(self, level):
        """Execute scanning pattern for thorough QR detection"""
        base_pos = self.level_positions[level]
        positions = []
        
        for offset in self.scan_offsets:
            # Calculate target position
            pan = max(0, min(180, base_pos['pan'] + offset['pan']))
            tilt = max(0, min(180, base_pos['tilt'] + offset['tilt']))
            zoom = max(1.0, min(5.0, base_pos['zoom'] + offset['zoom']))
            
            # Move to position
            self.ptz.set_pan_angle(pan)
            time.sleep(0.4)
            self.ptz.set_tilt_angle(tilt)
            time.sleep(0.4)
            self.ptz.set_zoom(zoom)
            time.sleep(1.0)
            
            positions.append({'pan': pan, 'tilt': tilt, 'zoom': zoom})
            
        return positions

# -------------------- Smart QR Detector -------------------- #
class SmartQRDetector:
    def __init__(self):
        # QR code validation patterns (customize based on your QR format)
        self.valid_qr_patterns = [
            r'^[A-Z0-9]{8,20}$',        # Alphanumeric 8-20 chars
            r'^KEG[0-9]{6,10}$',        # KEG prefix + numbers
            r'^[0-9]{10,15}$',          # Pure numeric
            r'^[A-Z]{2}[0-9]{8}[A-Z]{2}$'  # Letter-Number-Letter pattern
        ]
        
        # Track processed QR codes to avoid duplicates
        self.processed_qrs = set()
        self.level_qrs = {1: [], 2: [], 3: []}

    def is_valid_keg_qr(self, qr_data):
        """Validate if QR code is a legitimate keg QR code"""
        if not qr_data or len(qr_data) < 6:
            return False
            
        # Check against valid patterns
        for pattern in self.valid_qr_patterns:
            if re.match(pattern, qr_data.upper()):
                return True
                
        # Additional validation rules
        if qr_data.isdigit() and 8 <= len(qr_data) <= 15:
            return True
            
        if qr_data.isalnum() and 8 <= len(qr_data) <= 20:
            return True
            
        return False

    def is_duplicate(self, qr_data):
        """Check if QR code already processed"""
        return qr_data in self.processed_qrs

    def detect_valid_qr_codes(self, frame):
        """Detect and validate QR codes from frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multiple enhancement techniques for better detection
        enhanced_frames = self._enhance_frame(gray)
        
        valid_qrs = {}
        
        # Try detection on each enhanced frame
        for enhanced_frame, method in enhanced_frames:
            try:
                qr_codes = pyzbar.decode(enhanced_frame, symbols=[pyzbar.ZBarSymbol.QRCODE])
                
                for qr in qr_codes:
                    try:
                        qr_data = qr.data.decode('utf-8').strip()
                        
                        # Validate QR code
                        if (self.is_valid_keg_qr(qr_data) and 
                            not self.is_duplicate(qr_data) and 
                            qr_data not in valid_qrs):
                            
                            # Calculate center point
                            points = qr.polygon
                            if len(points) == 4:
                                center = (
                                    int(sum(p.x for p in points) / 4),
                                    int(sum(p.y for p in points) / 4)
                                )
                                
                                # Calculate quality score
                                area = cv2.contourArea(np.array([(p.x, p.y) for p in points]))
                                quality = min(100, area / 100)
                                
                                valid_qrs[qr_data] = {
                                    'data': qr_data,
                                    'center': center,
                                    'polygon': [(p.x, p.y) for p in points],
                                    'area': area,
                                    'quality': quality,
                                    'detection_method': method,
                                    'timestamp': datetime.now().isoformat()
                                }
                                
                    except UnicodeDecodeError:
                        continue
                        
            except Exception as e:
                logger.debug(f"Detection method {method} failed: {e}")
                
        # Sort by quality (best first)
        sorted_qrs = sorted(valid_qrs.values(), key=lambda x: x['quality'], reverse=True)
        return sorted_qrs

    def _enhance_frame(self, gray_frame):
        """Apply multiple enhancement techniques"""
        enhanced = []
        
        # 1. CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced.append((clahe.apply(gray_frame), "CLAHE"))
        
        # 2. Adaptive threshold
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        enhanced.append((adaptive, "Adaptive"))
        
        # 3. Histogram equalization
        equalized = cv2.equalizeHist(gray_frame)
        enhanced.append((equalized, "Histogram"))
        
        # 4. Morphological operations
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(gray_frame, cv2.MORPH_CLOSE, kernel)
        enhanced.append((morph, "Morphological"))
        
        # 5. Bilateral filter
        bilateral = cv2.bilateralFilter(gray_frame, 9, 75, 75)
        enhanced.append((bilateral, "Bilateral"))
        
        return enhanced

    def add_qr_to_level(self, level, qr_data):
        """Add QR code to specific level and global tracking"""
        if level in self.level_qrs and len(self.level_qrs[level]) < 6:
            self.level_qrs[level].append(qr_data)
            self.processed_qrs.add(qr_data['data'])
            return True
        return False

    def get_level_qr_count(self, level):
        """Get current QR count for level"""
        return len(self.level_qrs.get(level, []))

    def is_level_complete(self, level):
        """Check if level has 6 QR codes"""
        return self.get_level_qr_count(level) == 6

# -------------------- Automatic Scanner System -------------------- #
class AutoKegQRScanner:
    def __init__(self, camera_id=0, cloud_endpoint="https://your-api.com/qr-upload"):
        self.camera_id = camera_id
        self.cloud_endpoint = cloud_endpoint
        self.cap = None
        self.ptz = AutoPTZController()
        self.qr_detector = SmartQRDetector()
        
        # Scanning parameters
        self.max_scan_time_per_level = 90  # seconds
        self.stabilization_frames = 8
        self.detection_frames_per_position = 5

    def initialize_camera(self):
        """Initialize camera with optimal settings"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            logger.error("Camera initialization failed")
            return False
        
        # Set optimal camera parameters
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        
        # Auto-adjust for lighting
        self._auto_adjust_camera()
        
        logger.info("Camera initialized successfully")
        return True

    def _auto_adjust_camera(self):
        """Automatically adjust camera settings based on lighting"""
        light_samples = []
        for _ in range(5):
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                light_samples.append(np.mean(gray))
            time.sleep(0.1)
        
        if light_samples:
            avg_light = np.mean(light_samples)
            
            if avg_light < 80:  # Low light
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 60)
                self.cap.set(cv2.CAP_PROP_CONTRAST, 70)
            elif avg_light > 180:  # Bright light
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 40)
                self.cap.set(cv2.CAP_PROP_CONTRAST, 50)
            else:  # Normal light
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
                self.cap.set(cv2.CAP_PROP_CONTRAST, 60)

    def scan_pallet_level(self, level):
        """Automatically scan one pallet level for exactly 6 QR codes"""
        logger.info(f"Starting automatic scan of pallet level {level}")
        
        # Move to level position
        if not self.ptz.move_to_level(level):
            logger.error(f"Failed to position camera for level {level}")
            return []
        
        # Stabilize camera
        for _ in range(self.stabilization_frames):
            ret, _ = self.cap.read()
            if ret:
                time.sleep(0.1)
        
        start_time = time.time()
        level_qrs = []
        
        # Continue scanning until we have 6 QR codes or timeout
        while (len(level_qrs) < 6 and 
               (time.time() - start_time) < self.max_scan_time_per_level):
            
            # Execute scanning pattern
            positions = self.ptz.scan_positions(level)
            
            for pos_idx, position in enumerate(positions):
                if len(level_qrs) >= 6:
                    break
                
                # Capture multiple frames at this position
                for frame_idx in range(self.detection_frames_per_position):
                    ret, frame = self.cap.read()
                    if not ret:
                        continue
                    
                    # Detect valid QR codes
                    detected_qrs = self.qr_detector.detect_valid_qr_codes(frame)
                    
                    # Add new valid QR codes
                    for qr in detected_qrs:
                        if len(level_qrs) >= 6:
                            break
                            
                        # Check if not already in this level
                        if not any(existing['data'] == qr['data'] for existing in level_qrs):
                            qr['level'] = level
                            qr['position_index'] = pos_idx
                            qr['frame_index'] = frame_idx
                            qr['keg_position'] = self._estimate_keg_position(qr['center'], frame.shape)
                            
                            level_qrs.append(qr)
                            logger.info(f"Level {level}: Found QR {len(level_qrs)}/6 - {qr['data'][:15]}...")
                    
                    time.sleep(0.2)  # Brief pause between frames
                
                # Brief pause between positions
                time.sleep(0.3)
            
            # If still not complete, try different approach
            if len(level_qrs) < 6:
                logger.info(f"Level {level}: Only {len(level_qrs)}/6 found, adjusting...")
                # Slight adjustment to improve coverage
                time.sleep(1.0)
        
        scan_duration = time.time() - start_time
        logger.info(f"Level {level} scan complete: {len(level_qrs)}/6 QR codes in {scan_duration:.1f}s")
        
        return level_qrs

    def _estimate_keg_position(self, center, frame_shape):
        """Estimate keg position in 2x3 grid"""
        height, width = frame_shape[:2]
        x, y = center
        
        col = 1 if x < width/3 else (2 if x < 2*width/3 else 3)
        row = 1 if y < height/2 else 2
        
        return f"R{row}C{col}"

    def create_batch_data(self, level, qr_codes):
        """Create batch data for upload"""
        batch_id = f"AUTO_L{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            'batch_id': batch_id,
            'keg_type': 'Quarter_Barrel',
            'arrangement': '2x3_grid',
            'level': level,
            'timestamp': datetime.now().isoformat(),
            'qr_count': len(qr_codes),
            'qr_codes': qr_codes,
            'system_info': {
                'mode': 'fully_automatic',
                'camera_id': self.camera_id,
                'platform': 'jetson_nano_arducam_ptz'
            }
        }

    def upload_to_cloud(self, batch_data):
        """Upload batch to cloud with retry"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer YOUR_API_TOKEN',
                'X-Level': str(batch_data['level'])
            }
            
            response = requests.post(self.cloud_endpoint, json=batch_data, 
                                   headers=headers, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"✅ Level {batch_data['level']} uploaded successfully")
                return True
            else:
                logger.error(f"❌ Upload failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return False

    def save_locally(self, batch_data):
        """Save batch data locally"""
        filename = f"{batch_data['batch_id']}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(batch_data, f, indent=2)
            logger.info(f"💾 Saved locally: {filename}")
            return True
        except Exception as e:
            logger.error(f"Local save failed: {e}")
            return False

    def run_automatic_scanning(self):
        """Main automatic scanning process - no user interaction"""
        logger.info("🤖 AUTOMATIC KEG QR SCANNER STARTED")
        logger.info("📦 Scanning 3 pallet levels automatically...")
        logger.info("🔍 Target: 6 QR codes per level")
        logger.info("=" * 60)
        
        if not self.initialize_camera():
            logger.error("Camera initialization failed - exiting")
            return
        
        total_start_time = time.time()
        total_qrs = 0
        successful_levels = 0
        
        try:
            # Scan all 3 levels automatically
            for level in range(1, 4):
                logger.info(f"\n{'='*40}")
                logger.info(f"🔄 AUTO-SCANNING LEVEL {level}")
                logger.info(f"{'='*40}")
                
                # Scan this level
                level_qrs = self.scan_pallet_level(level)
                
                if level_qrs:
                    # Create and save batch
                    batch_data = self.create_batch_data(level, level_qrs)
                    self.save_locally(batch_data)
                    
                    # Try cloud upload
                    if self.upload_to_cloud(batch_data):
                        logger.info(f"✅ Level {level} complete: {len(level_qrs)} QR codes")
                    else:
                        logger.info(f"⚠️ Level {level} saved locally: {len(level_qrs)} QR codes")
                    
                    total_qrs += len(level_qrs)
                    successful_levels += 1
                else:
                    logger.warning(f"❌ Level {level}: No QR codes detected")
                
                # Brief pause before next level
                if level < 3:
                    logger.info("⏳ Moving to next level...")
                    time.sleep(3)
            
            # Final summary
            total_time = time.time() - total_start_time
            logger.info(f"\n🎉 AUTOMATIC SCANNING COMPLETE!")
            logger.info(f"⏱️  Total time: {total_time:.1f} seconds")
            logger.info(f"📊 Total QR codes: {total_qrs}")
            logger.info(f"📦 Successful levels: {successful_levels}/3")
            logger.info(f"🚀 Average: {total_qrs/(total_time/60):.1f} QR codes/minute")
            
        except Exception as e:
            logger.error(f"Scanning error: {e}")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            logger.info("🏁 System shutdown complete")

# -------------------- Main Entry Point -------------------- #
def main():
    """Main entry point - starts automatic scanning immediately"""
    logger.info("🚀 Starting Automatic Keg QR Scanner...")
    logger.info("📷 Initializing Arducam PTZ + Jetson Nano system...")
    
    # Create and run scanner automatically
    scanner = AutoKegQRScanner(
        camera_id=0,
        cloud_endpoint="https://your-api.com/qr-upload"
    )
    
    # Start automatic scanning immediately
    scanner.run_automatic_scanning()

if __name__ == "__main__":
    main()

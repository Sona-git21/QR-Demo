import cv2
import numpy as np
from pyzbar import pyzbar
import json
import time
from datetime import datetime
import requests
import logging
from ArducamPTZ import ArducamPTZ
from collections import defaultdict

# -------------------- Logging Setup -------------------- #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- CUDA Detection -------------------- #
class CUDAManager:
    def __init__(self):
        self.cuda_available = False
        self.device_count = 0
        self.init_cuda()
    
    def init_cuda(self):
        """Initialize CUDA support and check availability"""
        try:
            self.device_count = cv2.cuda.getCudaEnabledDeviceCount()
            self.cuda_available = self.device_count > 0
            
            if self.cuda_available:
                logger.info(f"✅ CUDA detected: {self.device_count} device(s)")
                
                # Get device info
                gpu_info = cv2.cuda.DeviceInfo(0)
                logger.info(f"GPU Name: {gpu_info.name()}")
                logger.info(f"Compute Capability: {gpu_info.majorVersion()}.{gpu_info.minorVersion()}")
                logger.info(f"Memory: {gpu_info.totalMemory() / (1024**3):.1f} GB")
                
                # Test CUDA operations
                test_mat = cv2.cuda.GpuMat()
                logger.info("🚀 CUDA operations ready")
                
            else:
                logger.warning("⚠️  CUDA not available - using CPU processing")
                
        except Exception as e:
            logger.error(f"CUDA initialization failed: {e}")
            self.cuda_available = False
    
    def is_available(self):
        return self.cuda_available

# -------------------- PTZ Controller -------------------- #
class PTZController:
    def __init__(self):
        self.ptz = ArducamPTZ()
        self.ptz.init()
        self.ptz.reset()
        logger.info("PTZ initialized and centered")

        # Optimized for Quarter Barrel kegs (16⅛" wide) in 2x3 arrangement
        # NEW: Expand positions to include different keg types and arrangements
        self.keg_positions = {
            'Quarter_Barrel_2x3': {
                1: {'pan': 90, 'tilt': 45, 'zoom': 2.2}, # Bottom pallet (2m)
                2: {'pan': 90, 'tilt': 55, 'zoom': 1.6}, # Middle pallet
                3: {'pan': 90, 'tilt': 70, 'zoom': 1.0}  # Top pallet
            },
            'Full_Size_2x2': { # Example: Full size kegs in a 2x2 arrangement
                1: {'pan': 90, 'tilt': 40, 'zoom': 1.8},
                2: {'pan': 90, 'tilt': 60, 'zoom': 1.2}
            },
            'Slim_Quarter_2x4': { # Example: Slim Quarter kegs in a 2x4 arrangement
                1: {'pan': 90, 'tilt': 48, 'zoom': 2.5},
                2: {'pan': 90, 'tilt': 62, 'zoom': 1.9}
            }
            # Add more configurations as needed (e.g., 'Pony_Keg_3x3', 'Corny_Keg_Single')
        }
        self.current_keg_type_arrangement = None # To store the active configuration

    def set_keg_type_arrangement(self, keg_type_arrangement_key):
        if keg_type_arrangement_key not in self.keg_positions:
            logger.error(f"Invalid keg type and arrangement key: {keg_type_arrangement_key}")
            return False
        self.current_keg_type_arrangement = keg_type_arrangement_key
        logger.info(f"Set PTZ configuration for: {keg_type_arrangement_key}")
        return True

    def move_to_level(self, level):
        """Move camera to optimal position for selected keg type and arrangement"""
        if not self.current_keg_type_arrangement:
            logger.error("No keg type and arrangement selected for PTZ. Call set_keg_type_arrangement first.")
            return False
        
        positions = self.keg_positions[self.current_keg_type_arrangement]
        
        if level not in positions:
            logger.error(f"Invalid level {level} for {self.current_keg_type_arrangement}")
            return False
        
        pos = positions[level]
        logger.info(f"Moving to {self.current_keg_type_arrangement} level {level}...")
        
        self.ptz.set_pan_angle(pos['pan'])
        time.sleep(0.8)
        self.ptz.set_tilt_angle(pos['tilt'])
        time.sleep(0.8)
        self.ptz.set_zoom(pos['zoom'])
        time.sleep(2.5)
        
        logger.info(f"Camera positioned: pan={pos['pan']}, tilt={pos['tilt']}, zoom={pos['zoom']}")
        return True

# -------------------- CUDA-Enhanced QR Detector -------------------- #
class CUDAQRDetector:
    def __init__(self): # Removed expected_qr_count from init, will be passed to scan_pallet_level
        self.cuda_manager = CUDAManager()
        
        # CUDA-based image processing filters
        if self.cuda_manager.is_available():
            self.setup_cuda_filters()
            logger.info("🚀 CUDA-accelerated QR detection enabled")
        else:
            logger.info("💻 CPU-based QR detection enabled")
    
    def setup_cuda_filters(self):
        """Initialize CUDA-based image processing filters"""
        try:
            # CUDA filters for image enhancement
            self.cuda_bilateral_filter = cv2.cuda.createBilateralFilter(cv2.CV_8UC1, -1, 80, 80)
            self.cuda_gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
            self.cuda_morph_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_CLOSE, cv2.CV_8UC1, 
                                                                   np.ones((3,3), np.uint8))
            logger.info("CUDA filters initialized successfully")
        except Exception as e:
            logger.error(f"CUDA filter setup failed: {e}")
            self.cuda_manager.cuda_available = False

    def detect_qr_codes_cuda(self, frame):
        """CUDA-accelerated QR detection"""
        if not self.cuda_manager.is_available():
            return self.detect_qr_codes_cpu(frame)
        
        try:
            # Upload frame to GPU
            gpu_frame = cv2.cuda.GpuMat()
            gpu_frame.upload(frame)
            
            # Convert to grayscale on GPU
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            
            enhanced_frames = []
            
            # 1. CUDA CLAHE enhancement
            clahe = cv2.cuda.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
            gpu_clahe = clahe.apply(gpu_gray)
            enhanced_frames.append(gpu_clahe)
            
            # 2. CUDA Bilateral filtering
            gpu_bilateral = cv2.cuda.GpuMat()
            self.cuda_bilateral_filter.apply(gpu_gray, gpu_bilateral)
            enhanced_frames.append(gpu_bilateral)
            
            # 3. CUDA Gaussian + threshold
            gpu_gaussian = cv2.cuda.GpuMat()
            self.cuda_gaussian_filter.apply(gpu_gray, gpu_gaussian)
            
            # Threshold on GPU
            gpu_thresh = cv2.cuda.threshold(gpu_gaussian, 0, 255, 
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            enhanced_frames.append(gpu_thresh)
            
            # 4. CUDA Morphological operations
            gpu_morph = cv2.cuda.GpuMat()
            self.cuda_morph_filter.apply(gpu_gray, gpu_morph)
            enhanced_frames.append(gpu_morph)
            
            # Download processed frames and detect QR codes
            all_qr_codes = []
            for i, gpu_enhanced in enumerate(enhanced_frames):
                try:
                    # Download from GPU to CPU
                    cpu_enhanced = gpu_enhanced.download()
                    
                    # QR detection on CPU (pyzbar doesn't support CUDA)
                    qr_codes = pyzbar.decode(cpu_enhanced, symbols=[pyzbar.ZBarSymbol.QRCODE])
                    
                    if qr_codes:
                        logger.debug(f"CUDA enhancement method {i+1} found {len(qr_codes)} QR codes")
                    all_qr_codes.extend(qr_codes)
                    
                except Exception as e:
                    logger.debug(f"CUDA enhancement method {i+1} failed: {e}")
                    continue
            
            return self.process_qr_results(all_qr_codes)
            
        except Exception as e:
            logger.warning(f"CUDA processing failed, falling back to CPU: {e}")
            return self.detect_qr_codes_cpu(frame)

    def detect_qr_codes_cpu(self, frame):
        """CPU-based QR detection (fallback)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        enhanced_frames = []
        
        # CPU-based enhancements
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
        enhanced_frames.append(clahe.apply(gray))
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        enhanced_frames.append(adaptive_thresh)
        
        equalized = cv2.equalizeHist(gray)
        enhanced_frames.append(equalized)
        
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        enhanced_frames.append(morph)
        
        # QR detection
        all_qr_codes = []
        for i, enhanced in enumerate(enhanced_frames):
            try:
                qr_codes = pyzbar.decode(enhanced, symbols=[pyzbar.ZBarSymbol.QRCODE])
                if qr_codes:
                    logger.debug(f"CPU enhancement method {i+1} found {len(qr_codes)} QR codes")
                all_qr_codes.extend(qr_codes)
            except Exception as e:
                logger.debug(f"CPU enhancement method {i+1} failed: {e}")
                continue
        
        return self.process_qr_results(all_qr_codes)

    def process_qr_results(self, qr_codes):
        """Process and deduplicate QR detection results"""
        unique_qrs = {}
        for qr in qr_codes:
            try:
                data = qr.data.decode('utf-8')
                if data and data not in unique_qrs:
                    points = qr.polygon
                    if len(points) == 4:
                        center = (
                            int(sum(p.x for p in points) / 4),
                            int(sum(p.y for p in points) / 4)
                        )
                        
                        area = cv2.contourArea(np.array([(p.x, p.y) for p in points]))
                        
                        unique_qrs[data] = {
                            'data': data,
                            'center': center,
                            'polygon': [(p.x, p.y) for p in points],
                            'timestamp': datetime.now().isoformat(),
                            'area': area,
                            'quality_score': min(100, area / 100)
                        }
            except (UnicodeDecodeError, AttributeError) as e:
                logger.debug(f"Failed to decode QR: {e}")
                continue
        
        sorted_qrs = sorted(unique_qrs.values(), key=lambda x: x['quality_score'], reverse=True)
        logger.debug(f"Total unique QR codes found: {len(sorted_qrs)}")
        return sorted_qrs

    def detect_qr_codes(self, frame):
        """Main QR detection method - uses CUDA if available"""
        if self.cuda_manager.is_available():
            return self.detect_qr_codes_cuda(frame)
        else:
            return self.detect_qr_codes_cpu(frame)

    # Modified to accept expected_qr_count for display
    def draw_qr_codes(self, frame, qr_codes, expected_qr_count=None):
        """Draw QR detection results with performance info and dynamic grid"""
        height, width = frame.shape[:2]
        
        # Draw dynamic grid overlay based on expected_qr_count
        grid_color = (100, 100, 100)
        if expected_qr_count:
            # Simple grid for common arrangements
            if expected_qr_count == 6: # 2x3
                for i in range(1, 3): # 2 vertical lines
                    x = int(width * i / 3)
                    cv2.line(frame, (x, 0), (x, height), grid_color, 1)
                y = int(height / 2) # 1 horizontal line
                cv2.line(frame, (0, y), (width, y), grid_color, 1)
            elif expected_qr_count == 4: # 2x2
                x = int(width / 2)
                cv2.line(frame, (x, 0), (x, height), grid_color, 1)
                y = int(height / 2)
                cv2.line(frame, (0, y), (width, y), grid_color, 1)
            # Add more specific grid logic for other arrangements if needed
        else: # Default to a simple crosshair if no specific count is given
            cv2.line(frame, (width // 2, 0), (width // 2, height), grid_color, 1)
            cv2.line(frame, (0, height // 2), (width, height // 2), grid_color, 1)
        
        # Draw QR codes
        for i, qr in enumerate(qr_codes):
            colors = [(0,255,0), (0,255,255), (255,0,255), (255,255,0), (0,128,255), (255,128,0)]
            color = colors[i % len(colors)]
            
            pts = np.array(qr['polygon'], dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 3)
            cv2.circle(frame, qr['center'], 8, color, -1)
            
            text = f"{i+1}: {qr['data'][:12]}..."
            cv2.putText(frame, text, (qr['center'][0]-60, qr['center'][1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            quality = int(qr.get('quality_score', 0))
            cv2.putText(frame, f"Q:{quality}", (qr['center'][0]-20, qr['center'][1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add performance indicator
        processing_mode = "🚀 CUDA" if self.cuda_manager.is_available() else "💻 CPU"
        cv2.putText(frame, processing_mode, (width-100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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
        return [qr for qr in qr_codes if not self.is_duplicate(qr['data'])]
    
    def get_stats(self):
        return {
            'total_processed': len(self.processed_qrs),
            'batches_created': len(self.batch_history),
            'batch_sizes': {bid: len(qrs) for bid, qrs in self.batch_history.items()}
        }

# -------------------- Main System -------------------- #
class KegQRSystem: # Renamed from CUDAQuarterBarrelQRSystem for generality
    def __init__(self, camera_id=0, cloud_endpoint="https://your-api.com/qr-upload"):
        self.camera_id = camera_id
        self.cloud_endpoint = cloud_endpoint
        self.cap = None
        self.ptz = PTZController()
        self.qr_detector = CUDAQRDetector() # Removed expected_qr_count here
        self.duplicate_tracker = DuplicateTracker()
        # self.max_levels = 3 # This will now be dynamic based on selected keg type
        self.batch_counter = 0
        
        # Performance monitoring
        self.frame_times = []
        self.detection_times = []

    def initialize_camera(self):
        """Initialize camera with Jetson Nano optimizations"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            logger.error("Camera not detected - check connection")
            return False
        
        # Jetson Nano optimized settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 15)  # Balanced for Jetson Nano
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Camera quality settings
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 60)
        
        # Test camera
        ret, test_frame = self.cap.read()
        if ret:
            logger.info(f"Camera initialized: {test_frame.shape[1]}x{test_frame.shape[0]}")
            
            # Test CUDA processing with first frame
            if self.qr_detector.cuda_manager.is_available():
                start_time = time.time()
                _ = self.qr_detector.detect_qr_codes(test_frame)
                processing_time = time.time() - start_time
                logger.info(f"CUDA processing test: {processing_time:.3f}s")
        else:
            logger.error("Camera test failed")
            return False
        
        return True

    def scan_pallet_level(self, level, keg_type, arrangement, expected_qr_count): # Added parameters
        """CUDA-accelerated scanning of a pallet level for a given keg type and arrangement"""
        logger.info(f"🔍 CUDA-Enhanced scanning: Level {level} for {keg_type} ({arrangement})")
        
        # Set PTZ configuration based on current keg type and arrangement
        ptz_key = f"{keg_type}_{arrangement}"
        if not self.ptz.set_keg_type_arrangement(ptz_key):
            logger.error(f"Failed to set PTZ for {ptz_key}. Skipping level {level}.")
            return []

        if not self.ptz.move_to_level(level):
            return []
        
        collected_qrs = []
        attempts = 0
        max_attempts = 60
        consecutive_empty_frames = 0
        max_empty_frames = 10
        
        # Camera stabilization
        logger.info("📷 Stabilizing camera...")
        for _ in range(8):
            ret, _ = self.cap.read()
            if ret:
                time.sleep(0.15)
        
        logger.info("🚀 Starting CUDA-accelerated QR detection...")
        
        while len(collected_qrs) < expected_qr_count and attempts < max_attempts:
            frame_start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # CUDA-enhanced QR detection
            detection_start = time.time()
            qr_codes = self.qr_detector.detect_qr_codes(frame)
            detection_time = time.time() - detection_start
            
            # Performance tracking
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            self.detection_times.append(detection_time)
            
            if not qr_codes:
                consecutive_empty_frames += 1
                if consecutive_empty_frames >= max_empty_frames:
                    logger.info("🔍 No QR codes - adjusting parameters...")
                    # You might add dynamic PTZ fine-tuning or exposure adjustments here
                    consecutive_empty_frames = 0
            else:
                consecutive_empty_frames = 0
                new_qrs = self.duplicate_tracker.get_new_qrs(qr_codes)
                
                for qr in new_qrs:
                    if not any(existing['data'] == qr['data'] for existing in collected_qrs):
                        qr['level'] = level
                        qr['detection_time'] = detection_time
                        collected_qrs.append(qr)
                        logger.info(f"✅ QR {len(collected_qrs)}/{expected_qr_count}: {qr['data'][:15]}... "
                                   f"({detection_time:.3f}s)")

            # Enhanced display with performance metrics, passing expected_qr_count
            display_frame = self.qr_detector.draw_qr_codes(frame.copy(), collected_qrs, expected_qr_count)
            
            # Performance overlay
            avg_detection_time = np.mean(self.detection_times[-10:]) if self.detection_times else 0
            fps = 1.0 / np.mean(self.frame_times[-10:]) if self.frame_times else 0
            
            perf_text = [
                f"{keg_type} Level {level} - QR: {len(collected_qrs)}/{expected_qr_count}",
                f"Detection Time: {detection_time:.3f}s (avg: {avg_detection_time:.3f}s)",
                f"FPS: {fps:.1f} | Processing: {'🚀 CUDA' if self.qr_detector.cuda_manager.is_available() else '💻 CPU'}",
                f"Attempts: {attempts}/{max_attempts}"
            ]
            
            for i, text in enumerate(perf_text):
                y_pos = 25 + (i * 22)
                color = (0, 255, 0) if len(collected_qrs) == expected_qr_count else (0, 255, 255)
                cv2.putText(display_frame, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.imshow(f"CUDA QR Scan - {keg_type} {arrangement} Level {level}", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                break

            attempts += 1
            time.sleep(0.1)  # Faster with CUDA

        cv2.destroyAllWindows()
        
        # Performance summary
        avg_detection = np.mean(self.detection_times) if self.detection_times else 0
        logger.info(f"🎯 Level {level} complete: {len(collected_qrs)}/{expected_qr_count} QR codes")
        logger.info(f"⚡ Average detection time: {avg_detection:.3f}s")
        
        return collected_qrs

    # Modified to accept keg_type and arrangement
    def create_batch_data(self, level, qr_codes, keg_type, arrangement):
        """Create batch with CUDA performance metrics"""
        self.batch_counter += 1
        batch_id = f"CUDA_{keg_type}_{arrangement}_L{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.batch_counter:03d}"
        
        for qr in qr_codes:
            self.duplicate_tracker.add_qr(qr['data'], batch_id)
        
        return {
            'batch_id': batch_id,
            'keg_type': keg_type, # Now dynamic
            'arrangement': arrangement, # Now dynamic
            'level': level,
            'timestamp': datetime.now().isoformat(),
            'qr_codes': qr_codes,
            'performance_metrics': {
                'cuda_enabled': self.qr_detector.cuda_manager.is_available(),
                'avg_detection_time': np.mean([qr.get('detection_time', 0) for qr in qr_codes]),
                'total_detection_time': sum([qr.get('detection_time', 0) for qr in qr_codes]),
                'gpu_device_count': self.qr_detector.cuda_manager.device_count
            },
            'system_info': {
                'platform': 'jetson_nano',
                'opencv_version': cv2.__version__,
                'cuda_support': cv2.cuda.getCudaEnabledDeviceCount() > 0,
                'processing_mode': 'CUDA' if self.qr_detector.cuda_manager.is_available() else 'CPU'
            }
        }

    def run(self):
        """Main execution with CUDA optimization for multiple keg types"""
        logger.info("🚀 CUDA-Enhanced Multi-Keg QR System")
        logger.info("=" * 60)
        
        if not self.initialize_camera():
            return

        # Define configurations for different keg types and arrangements
        # You would typically load this from a config file or get user input
        keg_configurations = {
            'Quarter_Barrel_2x3': {'levels': 3, 'expected_qrs_per_level': 6},
            'Full_Size_2x2': {'levels': 2, 'expected_qrs_per_level': 4},
            'Slim_Quarter_2x4': {'levels': 2, 'expected_qrs_per_level': 8}
            # Add more configurations here
        }

        # Example: iterate through defined configurations
        # In a real application, you might have a menu or selection logic here
        selected_config_name = 'Quarter_Barrel_2x3' # Default or user selected
        # selected_config_name = 'Full_Size_2x2' 
        # selected_config_name = 'Slim_Quarter_2x4' 

        if selected_config_name not in keg_configurations:
            logger.error(f"Configuration '{selected_config_name}' not found. Exiting.")
            return

        current_keg_type, current_arrangement = selected_config_name.rsplit('_', 1) # Assumes format like 'KegType_Arrangement'
        num_levels = keg_configurations[selected_config_name]['levels']
        expected_qrs = keg_configurations[selected_config_name]['expected_qrs_per_level']

        try:
            for level in range(1, num_levels + 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"🔍 LEVEL {level} - CUDA PROCESSING for {current_keg_type} ({current_arrangement})")
                logger.info(f"{'='*60}")
                
                input(f"\n📦 Position {current_keg_type} pallet in {current_arrangement} at level {level}\n"
                      f"Press Enter to start CUDA-accelerated scanning...")
                
                # Pass all necessary details to scan_pallet_level
                qr_data = self.scan_pallet_level(level, current_keg_type, current_arrangement, expected_qrs)
                
                if qr_data:
                    # Pass all necessary details to create_batch_data
                    batch = self.create_batch_data(level, qr_data, current_keg_type, current_arrangement)
                    
                    # Save with performance metrics
                    filename = f"{batch['batch_id']}.json"
                    with open(filename, 'w') as f:
                        json.dump(batch, f, indent=2)
                    
                    logger.info(f"💾 Saved: {filename}")
                    logger.info(f"⚡ CUDA Performance: {batch['performance_metrics']}")
                
                time.sleep(2)

            logger.info("🎉 CUDA-Enhanced scanning complete!")
            
        except KeyboardInterrupt:
            logger.info("⚠️  Interrupted by user")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

def main():
    print("🚀 CUDA-Enhanced Multi-Keg QR Scanner")
    print("⚡ Jetson Nano + GPU Acceleration")
    print("=" * 50)
    
    system = KegQRSystem() # Renamed class
    system.run()

if __name__ == "__main__":
    main()
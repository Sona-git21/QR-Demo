#!/usr/bin/env python3
"""
Arducam 12MP IMX477 PTZ Camera Testing Code for Jetson Nano
Tests pan/tilt, zoom, autofocus, and QR code detection clarity
"""

import cv2
import numpy as np
import time
import json
from datetime import datetime
import os
import argparse

# Try to import Arducam SDK
try:
    import ArducamSDK
    ARDUCAM_SDK_AVAILABLE = True
except ImportError:
    print("Warning: ArducamSDK not found. Some features may be limited.")
    ARDUCAM_SDK_AVAILABLE = False

class ArducamPTZTester:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.camera_handle = None
        
        # PTZ parameters
        self.pan_angle = 0
        self.tilt_angle = 0
        self.zoom_level = 1.0
        self.focus_value = 0
        
        # Test results storage
        self.test_results = []
        self.qr_detector = cv2.QRCodeDetector()
        
        # Camera parameters for IMX477
        self.focus_min = 0
        self.focus_max = 1023
        self.zoom_min = 1.0
        self.zoom_max = 4.0
        
        # Initialize camera
        self.initialize_camera()
    
    def initialize_camera(self):
        """Initialize the camera with proper settings for IMX477"""
        print("Initializing Arducam IMX477 PTZ camera...")
        
        # Try to use ArducamSDK if available
        if ARDUCAM_SDK_AVAILABLE:
            try:
                self.camera_handle = ArducamSDK.Py_ArduCam_autoopen(self.camera_index)
                if self.camera_handle is not None:
                    print("Arducam SDK initialized successfully")
                    # Set camera to high resolution mode
                    ArducamSDK.Py_ArduCam_setMode(self.camera_handle, 0)
                    return
            except Exception as e:
                print(f"ArducamSDK initialization failed: {e}")
        
        # Fallback to OpenCV
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # Set high resolution for IMX477 (4056x3040 max)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized with OpenCV")
    
    def capture_frame(self):
        """Capture a frame from the camera"""
        if self.camera_handle is not None:
            # Use Arducam SDK
            ret = ArducamSDK.Py_ArduCam_capture(self.camera_handle)
            if ret == 0:
                data = ArducamSDK.Py_ArduCam_getData(self.camera_handle)
                if data is not None:
                    frame = np.frombuffer(data, dtype=np.uint8)
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    return True, frame
            return False, None
        else:
            # Use OpenCV
            return self.cap.read()
    
    def set_focus(self, focus_value):
        """Set camera focus (0-1023 for IMX477)"""
        focus_value = max(self.focus_min, min(self.focus_max, focus_value))
        self.focus_value = focus_value
        
        if self.camera_handle is not None:
            try:
                # Arducam SDK focus control
                ArducamSDK.Py_ArduCam_setCtrl(self.camera_handle, 0x009A090A, focus_value)
                print(f"Focus set to: {focus_value}")
            except Exception as e:
                print(f"Focus control error: {e}")
        else:
            # OpenCV doesn't support focus control directly
            print(f"Focus set to: {focus_value} (simulated)")
    
    def auto_focus(self, frame):
        """Perform autofocus by maximizing image sharpness"""
        print("Performing autofocus...")
        best_focus = 0
        best_sharpness = 0
        
        # Test different focus values
        focus_range = range(0, 1024, 50)
        for focus_val in focus_range:
            self.set_focus(focus_val)
            time.sleep(0.1)  # Allow time for focus adjustment
            
            ret, test_frame = self.capture_frame()
            if ret:
                sharpness = self.calculate_sharpness(test_frame)
                print(f"Focus: {focus_val}, Sharpness: {sharpness:.2f}")
                
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_focus = focus_val
        
        # Set to best focus
        self.set_focus(best_focus)
        print(f"Autofocus complete. Best focus: {best_focus}, Sharpness: {best_sharpness:.2f}")
        return best_focus, best_sharpness
    
    def calculate_sharpness(self, frame):
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    def set_pan_tilt(self, pan, tilt):
        """Set pan and tilt angles"""
        self.pan_angle = max(-180, min(180, pan))
        self.tilt_angle = max(-90, min(90, tilt))
        
        if self.camera_handle is not None:
            try:
                # Arducam PTZ control
                ArducamSDK.Py_ArduCam_setCtrl(self.camera_handle, 0x009A0901, int(self.pan_angle))
                ArducamSDK.Py_ArduCam_setCtrl(self.camera_handle, 0x009A0902, int(self.tilt_angle))
                print(f"PTZ set to Pan: {self.pan_angle}°, Tilt: {self.tilt_angle}°")
            except Exception as e:
                print(f"PTZ control error: {e}")
        else:
            print(f"PTZ set to Pan: {self.pan_angle}°, Tilt: {self.tilt_angle}° (simulated)")
    
    def set_zoom(self, zoom_level):
        """Set zoom level"""
        self.zoom_level = max(self.zoom_min, min(self.zoom_max, zoom_level))
        
        if self.camera_handle is not None:
            try:
                # Arducam zoom control
                zoom_value = int((self.zoom_level - 1.0) * 100)
                ArducamSDK.Py_ArduCam_setCtrl(self.camera_handle, 0x009A0903, zoom_value)
                print(f"Zoom set to: {self.zoom_level}x")
            except Exception as e:
                print(f"Zoom control error: {e}")
        else:
            print(f"Zoom set to: {self.zoom_level}x (simulated)")
    
    def detect_qr_codes(self, frame):
        """Detect QR codes and assess their clarity"""
        data, points, _ = self.qr_detector.detectAndDecode(frame)
        
        qr_results = []
        if points is not None:
            points = points.astype(int)
            for i, qr_data in enumerate(data if isinstance(data, list) else [data]):
                if qr_data:
                    # Calculate QR code region for clarity assessment
                    if len(points.shape) == 3:
                        qr_points = points[i] if i < len(points) else points[0]
                    else:
                        qr_points = points
                    
                    # Extract QR code region
                    x_coords = qr_points[:, 0]
                    y_coords = qr_points[:, 1]
                    x1, y1 = np.min(x_coords), np.min(y_coords)
                    x2, y2 = np.max(x_coords), np.max(y_coords)
                    
                    qr_region = frame[y1:y2, x1:x2]
                    if qr_region.size > 0:
                        clarity = self.calculate_sharpness(qr_region)
                        area = (x2 - x1) * (y2 - y1)
                        
                        qr_results.append({
                            'data': qr_data,
                            'points': qr_points.tolist(),
                            'clarity': clarity,
                            'area': area,
                            'bbox': [x1, y1, x2, y2]
                        })
        
        return qr_results
    
    def draw_qr_results(self, frame, qr_results):
        """Draw QR code detection results on frame"""
        for qr in qr_results:
            points = np.array(qr['points'], dtype=int)
            cv2.polylines(frame, [points], True, (0, 255, 0), 3)
            
            # Draw clarity score
            x, y = points[0]
            cv2.putText(frame, f"Clarity: {qr['clarity']:.1f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Data: {qr['data'][:20]}...", 
                       (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def run_comprehensive_automated_test(self):
        """Run comprehensive automated test sequence with autofocus only"""
        print("Starting comprehensive automated test sequence...")
        print("This will test multiple PTZ positions, zoom levels, and autofocus")
        print("Estimated time: ~3-5 minutes")
        
        # Test positions for thorough coverage
        test_positions = [
            # Center positions with different zoom levels
            {'pan': 0, 'tilt': 0, 'zoom': 1.0, 'name': 'Center_1x'},
            {'pan': 0, 'tilt': 0, 'zoom': 1.5, 'name': 'Center_1.5x'},
            {'pan': 0, 'tilt': 0, 'zoom': 2.0, 'name': 'Center_2x'},
            {'pan': 0, 'tilt': 0, 'zoom': 2.5, 'name': 'Center_2.5x'},
            {'pan': 0, 'tilt': 0, 'zoom': 3.0, 'name': 'Center_3x'},
            
            # Pan sweep
            {'pan': -60, 'tilt': 0, 'zoom': 1.0, 'name': 'Left60_1x'},
            {'pan': -30, 'tilt': 0, 'zoom': 1.0, 'name': 'Left30_1x'},
            {'pan': 30, 'tilt': 0, 'zoom': 1.0, 'name': 'Right30_1x'},
            {'pan': 60, 'tilt': 0, 'zoom': 1.0, 'name': 'Right60_1x'},
            
            # Tilt sweep
            {'pan': 0, 'tilt': -45, 'zoom': 1.0, 'name': 'Down45_1x'},
            {'pan': 0, 'tilt': -20, 'zoom': 1.0, 'name': 'Down20_1x'},
            {'pan': 0, 'tilt': 20, 'zoom': 1.0, 'name': 'Up20_1x'},
            {'pan': 0, 'tilt': 45, 'zoom': 1.0, 'name': 'Up45_1x'},
            
            # Diagonal positions
            {'pan': -30, 'tilt': -20, 'zoom': 1.5, 'name': 'LeftDown_1.5x'},
            {'pan': 30, 'tilt': -20, 'zoom': 1.5, 'name': 'RightDown_1.5x'},
            {'pan': -30, 'tilt': 20, 'zoom': 1.5, 'name': 'LeftUp_1.5x'},
            {'pan': 30, 'tilt': 20, 'zoom': 1.5, 'name': 'RightUp_1.5x'},
            
            # High zoom tests
            {'pan': 0, 'tilt': 0, 'zoom': 4.0, 'name': 'Center_4x'},
            {'pan': -15, 'tilt': 0, 'zoom': 3.0, 'name': 'Left15_3x'},
            {'pan': 15, 'tilt': 0, 'zoom': 3.0, 'name': 'Right15_3x'},
        ]
        
        total_positions = len(test_positions)
        
        for idx, position in enumerate(test_positions, 1):
            print(f"\n=== Testing {idx}/{total_positions}: {position['name']} ===")
            print(f"Pan: {position['pan']}°, Tilt: {position['tilt']}°, Zoom: {position['zoom']}x")
            
            # Set PTZ position
            self.set_pan_tilt(position['pan'], position['tilt'])
            self.set_zoom(position['zoom'])
            print("Moving to position...")
            time.sleep(3)  # Allow time for movement
            
            # Perform autofocus
            print("Performing autofocus...")
            ret, frame = self.capture_frame()
            if ret:
                auto_focus_val, auto_sharpness = self.auto_focus(frame)
                time.sleep(1)
                
                # Capture frame at autofocus
                ret, auto_frame = self.capture_frame()
                if ret:
                    auto_qr_results = self.detect_qr_codes(auto_frame)
                    print(f"Autofocus: {auto_focus_val}, QR codes: {len(auto_qr_results)}")
                    
                    # Create test result
                    test_result = {
                        'timestamp': datetime.now().isoformat(),
                        'position': position,
                        'test_number': idx,
                        'focus_value': auto_focus_val,
                        'image_sharpness': auto_sharpness,
                        'qr_codes_detected': len(auto_qr_results),
                        'qr_results': auto_qr_results
                    }
                    
                    self.test_results.append(test_result)
                    
                    # Print results
                    print(f"Focus: {auto_focus_val} (auto)")
                    print(f"Sharpness: {auto_sharpness:.2f}")
                    print(f"QR codes detected: {len(auto_qr_results)}")
                    
                    for qr in auto_qr_results:
                        print(f"  - Data: {qr['data'][:30]}{'...' if len(qr['data']) > 30 else ''}")
                        print(f"    Clarity: {qr['clarity']:.2f}, Area: {qr['area']} pixels")
                    
                    # Save annotated image
                    annotated_frame = self.draw_qr_results(auto_frame.copy(), auto_qr_results)
                    
                    # Add test info overlay
                    cv2.putText(annotated_frame, f"Test {idx}: {position['name']}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(annotated_frame, f"Focus: {auto_focus_val} Sharpness: {auto_sharpness:.1f}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(annotated_frame, f"QR Codes: {len(auto_qr_results)}", 
                               (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"test_{idx:02d}_{position['name']}_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Saved: {filename}")
            
            # Progress update
            progress = (idx / total_positions) * 100
            print(f"Progress: {progress:.1f}% complete")
            
            time.sleep(1)  # Brief pause between tests
        
        print(f"\n=== Test Sequence Complete ===")
        self.generate_test_summary()
    
    def generate_test_summary(self):
        """Generate comprehensive test summary and analysis"""
        print("\n" + "="*60)
        print("TEST SUMMARY AND ANALYSIS")
        print("="*60)
        
        if not self.test_results:
            print("No test results available")
            return
        
        # Overall statistics
        total_tests = len(self.test_results)
        total_qr_detected = sum(result['qr_codes_detected'] for result in self.test_results)
        successful_detections = sum(1 for result in self.test_results if result['qr_codes_detected'] > 0)
        
        print(f"Total test positions: {total_tests}")
        print(f"Positions with QR detection: {successful_detections}")
        print(f"Success rate: {(successful_detections/total_tests*100):.1f}%")
        print(f"Total QR codes detected: {total_qr_detected}")
        
        # Best performing positions
        print(f"\n{'='*20} BEST PERFORMING POSITIONS {'='*20}")
        best_results = sorted(self.test_results, 
                            key=lambda x: x['qr_codes_detected'] * 1000 + x['image_sharpness'], 
                            reverse=True)[:5]
        
        for i, result in enumerate(best_results, 1):
            pos = result['position']
            print(f"{i}. {pos['name']} - Pan:{pos['pan']}° Tilt:{pos['tilt']}° Zoom:{pos['zoom']}x")
            print(f"   QR Codes: {result['qr_codes_detected']}, Sharpness: {result['image_sharpness']:.1f}")
            print(f"   Focus: {result['focus_value']} (auto)")
        
        # Analyze by zoom level
        print(f"\n{'='*20} ZOOM LEVEL ANALYSIS {'='*20}")
        zoom_analysis = {}
        for result in self.test_results:
            zoom = result['position']['zoom']
            if zoom not in zoom_analysis:
                zoom_analysis[zoom] = {'tests': 0, 'detections': 0, 'total_qr': 0, 'avg_sharpness': 0}
            
            zoom_analysis[zoom]['tests'] += 1
            if result['qr_codes_detected'] > 0:
                zoom_analysis[zoom]['detections'] += 1
            zoom_analysis[zoom]['total_qr'] += result['qr_codes_detected']
            zoom_analysis[zoom]['avg_sharpness'] += result['image_sharpness']
        
        for zoom in sorted(zoom_analysis.keys()):
            data = zoom_analysis[zoom]
            avg_sharpness = data['avg_sharpness'] / data['tests']
            success_rate = (data['detections'] / data['tests']) * 100
            print(f"Zoom {zoom}x: {success_rate:.1f}% success, Avg sharpness: {avg_sharpness:.1f}")
        
        # QR code clarity analysis
        if total_qr_detected > 0:
            print(f"\n{'='*20} QR CODE CLARITY ANALYSIS {'='*20}")
            all_qr_clarities = []
            for result in self.test_results:
                for qr in result['qr_results']:
                    all_qr_clarities.append(qr['clarity'])
            
            if all_qr_clarities:
                avg_clarity = np.mean(all_qr_clarities)
                min_clarity = np.min(all_qr_clarities)
                max_clarity = np.max(all_qr_clarities)
                
                print(f"Average QR clarity: {avg_clarity:.2f}")
                print(f"Clarity range: {min_clarity:.2f} - {max_clarity:.2f}")
                
                # Clarity quality classification
                excellent = sum(1 for c in all_qr_clarities if c > avg_clarity * 1.5)
                good = sum(1 for c in all_qr_clarities if avg_clarity < c <= avg_clarity * 1.5)
                poor = sum(1 for c in all_qr_clarities if c <= avg_clarity)
                
                print(f"Quality distribution:")
                print(f"  Excellent (>{avg_clarity*1.5:.1f}): {excellent}")
                print(f"  Good ({avg_clarity:.1f}-{avg_clarity*1.5:.1f}): {good}")
                print(f"  Poor (<{avg_clarity:.1f}): {poor}")
        
        # Recommendations
        print(f"\n{'='*20} RECOMMENDATIONS {'='*20}")
        
        # Best zoom level
        best_zoom = max(zoom_analysis.items(), 
                       key=lambda x: x[1]['detections']/x[1]['tests'])[0]
        print(f"1. Optimal zoom level: {best_zoom}x")
        
        # Best pan/tilt range
        successful_positions = [r for r in self.test_results if r['qr_codes_detected'] > 0]
        if successful_positions:
            avg_pan = np.mean([r['position']['pan'] for r in successful_positions])
            avg_tilt = np.mean([r['position']['tilt'] for r in successful_positions])
            print(f"2. Optimal positioning: Pan ~{avg_pan:.0f}°, Tilt ~{avg_tilt:.0f}°")
        
        print("\n" + "="*60)
    
    def save_test_results(self, filename="arducam_test_results.json"):
        """Save test results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"Test results saved to: {filename}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        if self.camera_handle is not None:
            ArducamSDK.Py_ArduCam_close(self.camera_handle)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Arducam PTZ Camera Tester')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    args = parser.parse_args()
    
    try:
        tester = ArducamPTZTester(args.camera)
        tester.run_comprehensive_automated_test()
        tester.save_test_results()
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'tester' in locals():
            tester.cleanup()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class HwkangLaneDetection(Node):
    def __init__(self):
        super().__init__('hwkang_lane_detection')
        
        # Parameters
        self.declare_parameter('debug', True)
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Lane detection parameters
        self.roi_top_y = 0.55      # ROI 상단 (이미지 높이의 55%)
        self.roi_bottom_y = 0.95   # ROI 하단 (이미지 높이의 95%)
        
        # Canny edge detection
        self.canny_low = 50
        self.canny_high = 150
        
        # Hough transform
        self.hough_threshold = 50
        self.min_line_length = 50
        self.max_line_gap = 150
        
        # Lane memory (smoothing)
        self.prev_left_lane = None
        self.prev_right_lane = None
        self.alpha = 0.3  # EMA smoothing factor
        
        # ROS2 subscribers & publishers
        self.sub_image = self.create_subscription(
            Image, 
            '/carla/hero/camera_front/image', 
            self.image_callback, 
            10
        )
        
        self.pub_left_edge = self.create_publisher(PointStamped, '/carla/lane/left_edge', 10)
        self.pub_right_edge = self.create_publisher(PointStamped, '/carla/lane/right_edge', 10)
        self.pub_center = self.create_publisher(PointStamped, '/carla/lane/center', 10)
        self.pub_debug_image = self.create_publisher(Image, '/carla/lane/debug_image', 1)
        self.pub_markers = self.create_publisher(MarkerArray, '/carla/lane/markers', 10)
        
        self.get_logger().info('Hwkang Lane Detection Node Started')
        self.get_logger().info('Subscribing: /carla/hero/camera_front/image')
        self.get_logger().info('Publishing: /carla/lane/left_edge, /carla/lane/right_edge, /carla/lane/center')
    
    def image_callback(self, msg: Image):
        """이미지 수신 시 호출되는 콜백"""
        try:
            # ROS Image -> OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Lane detection 수행
            left_edge, right_edge, center, debug_img = self.detect_lanes(cv_image)
            
            # Publish results
            if left_edge is not None and right_edge is not None:
                self.publish_lane_points(msg.header, left_edge, right_edge, center)
            
            # Debug image publish
            if self.debug and debug_img is not None:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
                debug_msg.header = msg.header
                self.pub_debug_image.publish(debug_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {str(e)}')
    
    def detect_lanes(self, image):
        """차선 검출 메인 함수"""
        h, w = image.shape[:2]
        
        # 1. ROI 설정
        roi_image = self.apply_roi(image)
        
        # 2. 전처리: Grayscale + Gaussian Blur
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Canny Edge Detection
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        
        # 4. Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        # 5. 좌우 차선 분리 및 평균
        left_lane, right_lane = self.separate_lanes(lines, w)
        
        # 6. Smoothing (EMA)
        if left_lane is not None:
            left_lane = np.array(left_lane, dtype=np.float32)  # 추가
            if self.prev_left_lane is None:
                self.prev_left_lane = left_lane
            else:
                self.prev_left_lane = self.alpha * left_lane + (1 - self.alpha) * self.prev_left_lane
            left_lane = self.prev_left_lane
            
        if right_lane is not None:
            right_lane = np.array(right_lane, dtype=np.float32)  # 추가
            if self.prev_right_lane is None:
                self.prev_right_lane = right_lane
            else:
                self.prev_right_lane = self.alpha * right_lane + (1 - self.alpha) * self.prev_right_lane
            right_lane = self.prev_right_lane
        
        # 7. Edge points 계산 (하단 기준)
        y_bottom = int(h * self.roi_bottom_y)
        left_edge = None
        right_edge = None
        center = None
        
        if left_lane is not None:
            x1, y1, x2, y2 = left_lane
            left_x = self.get_x_at_y(x1, y1, x2, y2, y_bottom)
            left_edge = (left_x, y_bottom)
        
        if right_lane is not None:
            x1, y1, x2, y2 = right_lane
            right_x = self.get_x_at_y(x1, y1, x2, y2, y_bottom)
            right_edge = (right_x, y_bottom)
        
        # Center point
        if left_edge is not None and right_edge is not None:
            center_x = (left_edge[0] + right_edge[0]) / 2.0
            center = (center_x, y_bottom)
        
        # Debug visualization
        debug_img = None
        if self.debug:
            debug_img = self.draw_debug(image.copy(), left_lane, right_lane, 
                                       left_edge, right_edge, center, edges)
        
        return left_edge, right_edge, center, debug_img
    
    def apply_roi(self, image):
        """ROI (Region of Interest) 적용"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 사다리꼴 ROI
        vertices = np.array([[
            (0, h),
            (w, h),
            (int(w * 0.6), int(h * self.roi_top_y)),
            (int(w * 0.4), int(h * self.roi_top_y))
        ]], dtype=np.int32)
        
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        return masked_image
    
    def separate_lanes(self, lines, image_width):
        """검출된 선들을 좌/우 차선으로 분리"""
        if lines is None:
            return None, None
        
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 기울기 계산
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            # 기울기로 좌/우 구분
            if abs(slope) < 0.5:  # 너무 수평인 선은 제외
                continue
            
            if slope < 0:  # 왼쪽 차선 (음수 기울기)
                left_lines.append(line[0])
            else:  # 오른쪽 차선 (양수 기울기)
                right_lines.append(line[0])
        
        # 평균 차선 계산
        left_lane = self.average_lines(left_lines) if left_lines else None
        right_lane = self.average_lines(right_lines) if right_lines else None
        
        return left_lane, right_lane
    
    def average_lines(self, lines):
        """여러 선분들의 평균 선 계산"""
        if not lines:
            return None
        
        x1_sum = y1_sum = x2_sum = y2_sum = 0
        for line in lines:
            x1, y1, x2, y2 = line
            x1_sum += x1
            y1_sum += y1
            x2_sum += x2
            y2_sum += y2
        
        n = len(lines)
        return np.array([x1_sum//n, y1_sum//n, x2_sum//n, y2_sum//n], dtype=np.float32)  # 이 줄 수정
    
    def get_x_at_y(self, x1, y1, x2, y2, y):
        """주어진 y 좌표에서의 x 값 계산"""
        if y2 - y1 == 0:
            return x1
        slope = (x2 - x1) / (y2 - y1)
        x = x1 + slope * (y - y1)
        return int(x)
    
    def publish_lane_points(self, header, left_edge, right_edge, center):
        """차선 포인트 publish"""
        # Left edge
        left_msg = PointStamped()
        left_msg.header = header
        left_msg.point.x = float(left_edge[0])
        left_msg.point.y = float(left_edge[1])
        left_msg.point.z = 0.0
        self.pub_left_edge.publish(left_msg)
        
        # Right edge
        right_msg = PointStamped()
        right_msg.header = header
        right_msg.point.x = float(right_edge[0])
        right_msg.point.y = float(right_edge[1])
        right_msg.point.z = 0.0
        self.pub_right_edge.publish(right_msg)
        
        # Center
        if center is not None:
            center_msg = PointStamped()
            center_msg.header = header
            center_msg.point.x = float(center[0])
            center_msg.point.y = float(center[1])
            center_msg.point.z = 0.0
            self.pub_center.publish(center_msg)
    
    def draw_debug(self, image, left_lane, right_lane, left_edge, right_edge, center, edges):
        """디버그 이미지 생성"""
        h, w = image.shape[:2]
        
        # Edge overlay
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
        
        # Draw lanes
        if left_lane is not None:
            x1, y1, x2, y2 = map(int, left_lane)  # 수정: int로 변환
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 5)
        
        if right_lane is not None:
            x1, y1, x2, y2 = map(int, right_lane)  # 수정: int로 변환
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 5)
        
        # Draw edge points
        if left_edge is not None:
            cv2.circle(overlay, (int(left_edge[0]), int(left_edge[1])), 10, (255, 0, 0), -1)
        if right_edge is not None:
            cv2.circle(overlay, (int(right_edge[0]), int(right_edge[1])), 10, (0, 0, 255), -1)
        if center is not None:
            cv2.circle(overlay, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)
            cv2.line(overlay, (int(center[0]), int(center[1])), 
                    (int(center[0]), int(center[1]) - 50), (0, 255, 0), 3)
        
        # ROI visualization
        roi_top = int(h * self.roi_top_y)
        roi_bottom = int(h * self.roi_bottom_y)
        cv2.line(overlay, (0, roi_top), (w, roi_top), (255, 255, 0), 2)
        cv2.line(overlay, (0, roi_bottom), (w, roi_bottom), (255, 255, 0), 2)
        
        # Info text
        if center is not None:
            offset = center[0] - w/2
            cv2.putText(overlay, f'Offset: {offset:.1f}px', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return overlay


def main(args=None):
    rclpy.init(args=args)
    node = HwkangLaneDetection()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
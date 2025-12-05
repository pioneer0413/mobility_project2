#!/usr/bin/env python3
# ros2_lane_detection.py
import math, cv2, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

class LaneWindowFit(Node):
    def __init__(self):
        super().__init__('lane_window_fit')

        # 1. RGB 카메라 사용 (노란색 중앙선 찾기)
        self.declare_parameter('image_topic', '/carla/hero/camera_front/image_color')
        
        # 노란색(Yellow) HSV 범위 설정
        self.hsv_yellow_lower = np.array([15, 30, 100])
        self.hsv_yellow_upper = np.array([40, 255, 255])

        # 카메라 오프셋 (왼쪽 0.5m 이동했으면 -0.5)
        self.declare_parameter('camera_offset_m', -0.5) 
        self.declare_parameter('closest_area_y_frac', 0.75) 

        # ROI & BEV (FOV 110도 대응)
        self.declare_parameter('trap_bottom_left',  0.0)
        self.declare_parameter('trap_bottom_right', 1.0)
        self.declare_parameter('trap_top_left',     0.20)
        self.declare_parameter('trap_top_right',    0.80)
        self.declare_parameter('trap_top_y',        0.50)
        self.declare_parameter('trap_bottom_y',     1.00)
        
        self.declare_parameter('bev_width',  640)
        self.declare_parameter('bev_height', 480)

        # 탐지 설정
        self.declare_parameter('n_windows', 16)
        self.declare_parameter('margin_px', 80)
        self.declare_parameter('minpix', 50) 
        self.declare_parameter('hist_frac', 0.0) 

        # Projection (오류 해결 완료)
        self.declare_parameter('poly_order', 2)
        self.declare_parameter('ema_alpha', 0.4)
        self.declare_parameter('m_per_pix_x', 0.01)
        self.declare_parameter('m_per_pix_y', 0.01)

        # Params Load
        self.cam_offset = float(self.get_parameter('camera_offset_m').get_parameter_value().double_value)
        self.closest_area_y_frac = float(self.get_parameter('closest_area_y_frac').get_parameter_value().double_value)
        self.bw = int(self.get_parameter('bev_width').get_parameter_value().integer_value)
        self.bh = int(self.get_parameter('bev_height').get_parameter_value().integer_value)
        self.minpix = int(self.get_parameter('minpix').get_parameter_value().integer_value)
        self.mx = float(self.get_parameter('m_per_pix_x').get_parameter_value().double_value)
        self.my = float(self.get_parameter('m_per_pix_y').get_parameter_value().double_value)
        self.nw = int(self.get_parameter('n_windows').get_parameter_value().integer_value)
        self.margin = int(self.get_parameter('margin_px').get_parameter_value().integer_value)
        self.hist_frac = float(self.get_parameter('hist_frac').get_parameter_value().double_value)
        self.poly_order = int(self.get_parameter('poly_order').get_parameter_value().integer_value)
        self.alpha = float(self.get_parameter('ema_alpha').get_parameter_value().double_value)
        self.tb_l = float(self.get_parameter('trap_bottom_left').get_parameter_value().double_value)
        self.tb_r = float(self.get_parameter('trap_bottom_right').get_parameter_value().double_value)
        self.tt_l = float(self.get_parameter('trap_top_left').get_parameter_value().double_value)
        self.tt_r = float(self.get_parameter('trap_top_right').get_parameter_value().double_value)
        self.ty   = float(self.get_parameter('trap_top_y').get_parameter_value().double_value)
        self.by   = float(self.get_parameter('trap_bottom_y').get_parameter_value().double_value)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub = self.create_subscription(Image, self.get_parameter('image_topic').get_parameter_value().string_value, self.cb, qos)
        
        self.pub_center = self.create_publisher(PointStamped, '/carla/lane/center', 10)
        self.pub_dbg    = self.create_publisher(Image, '/carla/lane/debug_image', 1)
        self.bridge = CvBridge()
        self.prev_fit  = None

        self.get_logger().info(f'[LaneWindowFit] Yellow Color Single Line Mode. Offset: {self.cam_offset}m')

    def _warp_perspective(self, img):
        H, W = img.shape[:2]
        src = np.float32([
            [W*self.tb_l, H*self.by], [W*self.tb_r, H*self.by],
            [W*self.tt_r, H*self.ty], [W*self.tt_l, H*self.ty]
        ])
        dst = np.float32([
            [0, self.bh], [self.bw, self.bh],
            [self.bw, 0], [0, 0]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M, (self.bw, self.bh), flags=cv2.INTER_NEAREST)
        return warped, M, Minv

    def _binary_from_rgb(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_yellow_lower, self.hsv_yellow_upper)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _find_central_peak_pixels(self, bin_bev, dbg_img=None):
        h, w = bin_bev.shape
        hist = np.sum(bin_bev, axis=0)
        
        # [Visual] 히스토그램 그리기
        if dbg_img is not None:
            max_val = np.max(hist)
            if max_val > 0:
                for x, val in enumerate(hist):
                    bar_h = int((val / max_val) * (h / 4))
                    cv2.line(dbg_img, (x, h), (x, h - bar_h), (255, 255, 255), 1)

        # 1. BEV 하단 25% 영역의 픽셀만 사용
        closest_y_limit = self.bh * self.closest_area_y_frac
        
        # 2. 히스토그램 피크 찾기 (중앙에서 가장 강한 픽셀 집합)
        center_x_start = int(w * 0.25)
        center_x_end = int(w * 0.75)
        
        hist_center = hist[center_x_start:center_x_end]
        
        if np.max(hist_center) < 5: # 피크가 너무 약하면 포기
            return None, None, 0
        
        base_x_relative = np.argmax(hist_center)
        base_x_absolute = base_x_relative + center_x_start
        
        pixel_count = np.sum(bin_bev) / 255 # 총 픽셀 수 (로그용)

        # 3. 중앙 피크를 중심으로 픽셀 수집 (Sliding Window 대신 간단한 클러스터링)
        # base_x를 중심으로 일정 폭(margin)만 픽셀을 찾음
        
        nonzero = bin_bev.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # 4. 필터링: Base_x를 중심으로 Margin 내에 있고, 하단 25% 내에 있는 픽셀만 남김
        margin_low = max(0, base_x_absolute - self.margin)
        margin_high = min(w, base_x_absolute + self.margin)
        
        central_filter = (nonzerox >= margin_low) & (nonzerox <= margin_high) & (nonzeroy >= closest_y_limit)
        
        final_x = nonzerox[central_filter]
        final_y = nonzeroy[central_filter]

        if len(final_x) < self.minpix:
            return None, None, pixel_count

        # [Visual] 측정 영역 (초록색)
        if dbg_img is not None:
            cv2.rectangle(dbg_img, (margin_low, int(closest_y_limit)), (margin_high, h), (0, 255, 0), 2)
            dbg_img[final_y, final_x] = [0, 255, 255] # Yellow Pixels

        return final_x, final_y, pixel_count


    def cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        H, W = img.shape[:2]

        mask = self._binary_from_rgb(img)
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        bev, M, Minv = self._warp_perspective(mask)
        bev_bin = (bev > 0).astype(np.uint8)*255
        
        dbg_bev = cv2.cvtColor(bev_bin, cv2.COLOR_GRAY2BGR)
        
        # 픽셀 수집
        lane_x, lane_y, pix_count = self._find_central_peak_pixels(bev_bin, dbg_bev)
        
        offset_m, angle_rad = float('nan'), 0.0
        overlay = np.zeros_like(img)

        # -------------------------------------------------------------
        # [핵심 로직] 픽셀 평균을 이용한 거리/각도 측정
        # -------------------------------------------------------------
        
        if lane_x is not None:
            
            # 1. X 좌표 평균 (중앙선 위치)
            center_x_avg_px = np.mean(lane_x)
            
            # 2. 거리 계산: (차선평균 X - 화면중앙 X) * 스케일 + 보정값
            offset_px = center_x_avg_px - (self.bw / 2)
            offset_m = offset_px * self.mx + self.cam_offset
            
            # 3. 시각화 (평균점)
            cv2.circle(dbg_bev, (int(center_x_avg_px), int(np.mean(lane_y))), 10, (255, 0, 0), -1) 
            
            # 각도: 단순화하여 0.0으로 설정
            angle_rad = 0.0 


        out = cv2.addWeighted(img, 1.0, overlay, 0.8, 0.0)
        
        thumb = cv2.resize(mask_vis, (320, 180))
        out[0:180, W-320:W] = thumb
        cv2.rectangle(out, (W-320, 0), (W, 180), (255,255,255), 2)
        cv2.putText(out, "Yellow Mask", (W-310, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        info_color = (0, 255, 0) if not math.isnan(offset_m) else (0, 0, 255)
        status_txt = f"Pixels: {pix_count} (Req: {self.minpix})"
        
        if not math.isnan(offset_m):
            txt = f'Dist={offset_m:.2f}m Ang={math.degrees(angle_rad):.1f}dg'
            pt_msg = PointStamped()
            pt_msg.header = msg.header
            pt_msg.point.x = float(offset_m)
            pt_msg.point.y = float(angle_rad)
            self.pub_center.publish(pt_msg)
            self.get_logger().info(f"🟢 {txt} | {status_txt}")
        else:
            txt = "No Line"
            self.get_logger().warn(f"🔴 Fail | {status_txt}")

        cv2.putText(out, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, info_color, 2)
        cv2.putText(out, status_txt, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)

        cv2.imshow("Lane Detection (Simple Avg)", out)
        cv2.imshow("Lane BEV Debug", dbg_bev)
        cv2.waitKey(1)

        imsg = self.bridge.cv2_to_imgmsg(out, encoding='bgr8')
        self.pub_dbg.publish(imsg)

def main():
    rclpy.init()
    node = LaneWindowFit()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
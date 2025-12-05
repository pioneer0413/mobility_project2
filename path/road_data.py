#!/usr/bin/env python3
"""
Town01 도로 정보를 기반으로
'장애물 필터링용 맵 파일(도로 polygon 집합)'을 생성하는 스크립트.

- CARLA 서버에 접속 (localhost:2000)
- Town01 맵의 모든 Driving lane을 대상으로
- 일정 간격으로 waypoint를 따라가며
  각 segment를 4점 polygon으로 만들고
- JSON 파일로 저장한다.

출력 형식 예시:
{
  "town": "Town01",
  "resolution": 2.0,
  "lanes": [
    {
      "road_id": 108,
      "lane_id": 1,
      "lane_type": "Driving",
      "is_intersection": false,
      "polygons": [
        [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
        ...
      ]
    },
    ...
  ]
}
"""

import json
import math
import argparse

import carla


def waypoint_to_segment_polygon(wp0: carla.Waypoint, wp1: carla.Waypoint):
    """
    하나의 lane segment (wp0 -> wp1)에 대해
    도로 한 조각을 나타내는 4점 polygon을 생성한다.

    양쪽 차선 경계는:
      center ± normal * lane_width/2
    로 계산한다.
    """
    # 첫 지점
    loc0 = wp0.transform.location
    yaw0 = math.radians(wp0.transform.rotation.yaw)
    half_w0 = wp0.lane_width * 0.2
    nx0 = -math.sin(yaw0)
    ny0 = math.cos(yaw0)

    # 두 번째 지점
    loc1 = wp1.transform.location
    yaw1 = math.radians(wp1.transform.rotation.yaw)
    half_w1 = wp1.lane_width * 0.2
    nx1 = -math.sin(yaw1)
    ny1 = math.cos(yaw1)

    # 좌측/우측 경계점 계산 (도로 왼쪽이 +normal 방향)
    # 순서: p0L, p0R, p1R, p1L (시계/반시계 일관되게)
    p0L = (loc0.x + nx0 * half_w0, loc0.y + ny0 * half_w0)
    p0R = (loc0.x - nx0 * half_w0, loc0.y - ny0 * half_w0)
    p1L = (loc1.x + nx1 * half_w1, loc1.y + ny1 * half_w1)
    p1R = (loc1.x - nx1 * half_w1, loc1.y - ny1 * half_w1)

    # polygon 4점 리스트 반환
    # 여기서는 (p0L, p0R, p1R, p1L)
    return [p0L, p0R, p1R, p1L]


def export_roadmap(town: str, host: str, port: int,
                   distance: float, output_path: str):
    client = carla.Client(host, port)
    client.set_timeout(10.0)

    world = client.get_world()
    amap = world.get_map()

    if amap.name != town:
        print(f"[WARN] 현재 맵은 {amap.name} 이고, 요청한 맵은 {town} 입니다.")
        print("      world.load_world(town)을 사용해서 미리 맵을 바꿔두는 것이 안전합니다.")
    else:
        print(f"[INFO] Town: {amap.name}")

    print("[INFO] Waypoints 생성 중...")
    waypoints = amap.generate_waypoints(distance=distance)
    print(f"[INFO] waypoint 개수: {len(waypoints)}")

    # lane별로 polygon segment를 모으기 위한 dict
    lanes = {}  # key: (road_id, lane_id) -> dict 정보

    for wp in waypoints:
        # Driving lane만 사용
        if wp.lane_type != carla.LaneType.Driving:
            continue

        road_id = wp.road_id
        lane_id = wp.lane_id
        key = (road_id, lane_id)

        if key not in lanes:
            lanes[key] = {
                "road_id": road_id,
                "lane_id": lane_id,
                "lane_type": "Driving",
                "is_intersection": bool(wp.is_junction),
                "polygons": []
            }

        # 현재 waypoint에서 앞쪽 distance 만큼 떨어진 다음 waypoint를 찾는다.
        next_wps = wp.next(distance)
        if not next_wps:
            continue

        # 같은 lane_id, road_id 중 가장 가까운 것 선택
        next_wp = None
        for nwp in next_wps:
            if nwp.road_id == road_id and nwp.lane_id == lane_id:
                next_wp = nwp
                break
        if next_wp is None:
            continue

        poly = waypoint_to_segment_polygon(wp, next_wp)

        # float를 적당히 반올림 (파일 크기 감소 & 사람이 보기 좋게)
        lanes[key]["polygons"].append(
            [[round(px, 3), round(py, 3)] for (px, py) in poly]
        )

    # 결과 JSON 구조 만들기
    result = {
        "town": amap.name,
        "resolution": distance,
        "lanes": list(lanes.values())
    }

    print(f"[INFO] lane 개수: {len(result['lanes'])}")
    print(f"[INFO] JSON 파일 저장: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("[INFO] 완료")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--town", type=str, default="Town01",
                   help="대상 CARLA 맵 이름 (예: Town01)")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--distance", type=float, default=2.0,
                   help="waypoint 간격 (m)")
    p.add_argument("--output", type=str, default="town01_roadmap0.2.json",
                   help="출력 JSON 파일 경로")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_roadmap(
        town=args.town,
        host=args.host,
        port=args.port,
        distance=args.distance,
        output_path=args.output
    )

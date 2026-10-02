"""TSRD Chinese traffic-sign labels used by training and inference."""

from __future__ import annotations


_NAMES = [
    ("限速5km/h", "Speed limit 5 km/h", "限速"),
    ("限速15km/h", "Speed limit 15 km/h", "限速"),
    ("限速30km/h", "Speed limit 30 km/h", "限速"),
    ("限速40km/h", "Speed limit 40 km/h", "限速"),
    ("限速50km/h", "Speed limit 50 km/h", "限速"),
    ("限速60km/h", "Speed limit 60 km/h", "限速"),
    ("限速70km/h", "Speed limit 70 km/h", "限速"),
    ("限速80km/h", "Speed limit 80 km/h", "限速"),
    ("禁止直行和左转", "No straight or left turn", "禁令"),
    ("禁止直行和右转", "No straight or right turn", "禁令"),
    ("禁止直行", "No straight ahead", "禁令"),
    ("禁止向左转弯", "No left turn", "禁令"),
    ("禁止向左向右转弯", "No left or right turn", "禁令"),
    ("禁止向右转弯", "No right turn", "禁令"),
    ("禁止超车", "No overtaking", "禁令"),
    ("禁止掉头", "No U-turn", "禁令"),
    ("禁止机动车驶入", "No motor vehicles", "禁令"),
    ("禁止鸣喇叭", "No honking", "禁令"),
    ("解除40km/h限速", "End of 40 km/h limit", "禁令"),
    ("解除50km/h限速", "End of 50 km/h limit", "禁令"),
    ("直行和向右转弯", "Straight or right turn", "指示"),
    ("直行", "Straight ahead", "指示"),
    ("向左转弯", "Turn left", "指示"),
    ("向左或向右转弯", "Turn left or right", "指示"),
    ("向右转弯", "Turn right", "指示"),
    ("靠左侧行驶", "Keep left", "指示"),
    ("靠右侧行驶", "Keep right", "指示"),
    ("环岛行驶", "Roundabout", "指示"),
    ("机动车行驶", "Motor vehicles only", "指示"),
    ("鸣喇叭", "Sound horn", "指示"),
    ("非机动车行驶", "Non-motor vehicles only", "指示"),
    ("允许掉头", "U-turn permitted", "指示"),
    ("左右绕行", "Pass either side", "指示"),
    ("注意交通信号灯", "Traffic signals ahead", "警告"),
    ("注意危险", "General caution", "警告"),
    ("注意行人", "Pedestrians", "警告"),
    ("注意非机动车", "Non-motor vehicles", "警告"),
    ("注意儿童", "Children", "警告"),
    ("向右急转弯", "Sharp right turn", "警告"),
    ("向左急转弯", "Sharp left turn", "警告"),
    ("下陡坡", "Steep descent", "警告"),
    ("上陡坡", "Steep ascent", "警告"),
    ("慢行", "Slow down", "警告"),
    ("右侧T形路口", "T-junction on right", "警告"),
    ("左侧T形路口", "T-junction on left", "警告"),
    ("村庄", "Village", "警告"),
    ("反向弯路", "Reverse curves", "警告"),
    ("无人看守铁路道口", "Unguarded railway crossing", "警告"),
    ("施工", "Road works", "警告"),
    ("连续弯路", "Winding road", "警告"),
    ("有人看守铁路道口", "Guarded railway crossing", "警告"),
    ("事故易发路段", "Accident-prone area", "警告"),
    ("停车让行", "Stop", "禁令"),
    ("禁止通行", "No thoroughfare", "禁令"),
    ("禁止停车", "No parking", "禁令"),
    ("禁止驶入", "No entry", "禁令"),
    ("减速让行", "Give way", "禁令"),
    ("停车检查", "Stop for inspection", "禁令"),
]

TSRD_LABELS = [
    (class_id, name_zh, name_en, category)
    for class_id, (name_zh, name_en, category) in enumerate(_NAMES)
]

LABEL_BY_ID = {
    class_id: {
        "id": class_id,
        "name_zh": name_zh,
        "name_en": name_en,
        "category": category,
    }
    for class_id, name_zh, name_en, category in TSRD_LABELS
}

DATASET_INFO = {
    "name": "TSRD",
    "full_name": "Chinese Traffic Sign Recognition Database",
    "region": "中国",
    "class_count": len(TSRD_LABELS),
    "train_samples": 4170,
    "test_samples": 1994,
}

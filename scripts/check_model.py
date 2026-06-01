"""打印指定 JSON 模型的 cascade 级数、弱分类器数量和阈值。"""

import json
for mname in ["custom_cascade_v1_no_hnm.json", "custom_cascade_v3_hnm.json"]:
    m = json.load(open(f"models/{mname}"))
    stages = m.get("stages", [])
    print(f"\n=== {mname} ===")
    print(f"window_size: {m.get('window_size')}")
    print(f"stages: {len(stages)}")
    for i, s in enumerate(stages):
        wc = s["weak_classifiers"]
        thr = s["threshold"]
        print(f"  Stage {i+1}: {len(wc)} weak classifiers, threshold={thr:.4f}")

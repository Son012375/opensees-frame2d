"""IFC → BuildingModel 통합 테스트.

실제 IFC 파일에서 BuildingModel을 생성하고
to_frame3d_kwargs()까지 정상 동작하는지 검증한다.
"""
import sys
import json

sys.path.insert(0, "mcp-server")
from core.building_model import BuildingModel

IFC_PATH = r"C:\Users\youm\Documents\카카오톡 받은 파일\2차시도_260121.ifc"

# IFC + 보완 config (용도, 지역 등 IFC에 없는 정보)
config = {
    "stories": [
        {"usage": "retail", "dead_load_finish": 1.5},   # 1F
        {"usage": "office", "dead_load_finish": 1.0},   # 2F
        {"usage": "office", "dead_load_finish": 0.5},   # Roof
    ],
    "region": "\uc11c\uc6b8",  # 서울
    "site_class": "S3",
    "importance": "II",
    "column_section": "H-300x300",
    "beam_x_section": "H-400x200",
    "beam_y_section": "H-400x200",
    "seismic_system": "ordinary_moment_frame",
    "auto_combinations": True,
}

print("=" * 60)
print("IFC -> BuildingModel Integration Test")
print("=" * 60)

# 1. from_ifc()
print("\n--- 1. BuildingModel.from_ifc() ---")
model = BuildingModel.from_ifc(IFC_PATH, config)
print(f"  num_stories: {model.num_stories}")
print(f"  total_height: {model.total_height}m")
print(f"  total_width_x: {model.total_width_x}m")
print(f"  total_width_y: {model.total_width_y}m")
print(f"  floor_area: {model.floor_area}m2")

# 2. 각 층 정보 확인
print("\n--- 2. Story details ---")
for s in model.stories:
    print(f"  Story {s.story}: h={s.height}m, usage={s.usage}, "
          f"finish={s.dead_load_finish} kN/m2, slab_t={s.slab_thickness}m")

# 3. to_frame3d_kwargs()
print("\n--- 3. frame3d kwargs ---")
kwargs = model.to_frame3d_kwargs()
for k, v in kwargs.items():
    print(f"  {k}: {v}")

# 4. summary()
print("\n--- 4. Summary ---")
summary = model.summary()
print(json.dumps(summary, indent=2, ensure_ascii=False))

# 5. 검증
print("\n--- 5. Validation ---")
ok = True

if model.num_stories != 3:
    print(f"  FAIL: num_stories={model.num_stories}, expected=3")
    ok = False
else:
    print(f"  PASS: num_stories=3")

if model.bays_x != [4.0, 4.0, 4.0]:
    print(f"  FAIL: bays_x={model.bays_x}, expected=[4.0,4.0,4.0]")
    ok = False
else:
    print(f"  PASS: bays_x=[4.0, 4.0, 4.0]")

if model.bays_y != [5.0, 5.0]:
    print(f"  FAIL: bays_y={model.bays_y}, expected=[5.0,5.0]")
    ok = False
else:
    print(f"  PASS: bays_y=[5.0, 5.0]")

if model.stories[0].usage != "retail":
    print(f"  FAIL: story 1 usage={model.stories[0].usage}, expected=retail")
    ok = False
else:
    print(f"  PASS: story 1 usage=retail (from config)")

# IFC 슬래브 두께 확인
if abs(model.stories[0].slab_thickness - 0.15) > 0.01:
    print(f"  FAIL: story 1 slab_thickness={model.stories[0].slab_thickness}, expected~0.15")
    ok = False
else:
    print(f"  PASS: story 1 slab_thickness={model.stories[0].slab_thickness}m (from IFC)")

if model.region != "\uc11c\uc6b8":
    print(f"  FAIL: region={model.region!r}")
    ok = False
else:
    print(f"  PASS: region=서울 (from config)")

if ok:
    print("\n>>> ALL CHECKS PASSED <<<")
else:
    print("\n>>> SOME CHECKS NEED REVIEW <<<")

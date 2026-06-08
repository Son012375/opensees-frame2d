"""ETABS 모델 빌딩 단계별 진단 스크립트.

Usage:
    .\\opensees-mcp\\Scripts\\python.exe scripts/diag_etabs_model.py --launch
    .\\opensees-mcp\\Scripts\\python.exe scripts/diag_etabs_model.py
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "mcp-server"))
from core.etabs_api import ETABSClient

_N_MM_C   = 10
_STEEL    = 1
_MAT      = "SS275"
_SEC      = "H400x200"


def step(label, ret_or_tuple):
    """Print step result, handle both scalar and tuple returns."""
    if isinstance(ret_or_tuple, tuple):
        print(f"  {label}: {ret_or_tuple}")
    else:
        status = "✅" if ret_or_tuple == 0 else f"❌(ret={ret_or_tuple})"
        print(f"  {label}: {status}")


def run(client):
    m = client.model

    print("\n─── Step 1: InitializeNewModel(10) ──────────────────")
    ret = m.InitializeNewModel(_N_MM_C)
    step("InitializeNewModel", ret)

    print("\n─── Step 2: File.NewBlank() ─────────────────────────")
    ret = m.File.NewBlank()
    step("NewBlank", ret)

    print("\n─── Step 3: SetMaterial ─────────────────────────────")
    ret = m.PropMaterial.SetMaterial(_MAT, _STEEL)
    step("SetMaterial", ret)

    ret = m.PropMaterial.SetMPIsotropic(_MAT, 210000.0, 0.3, 1.2e-5)
    step("SetMPIsotropic", ret)

    print("\n─── Step 4: GetNameList (materials) ─────────────────")
    n, names, ret = m.PropMaterial.GetNameList(0, [])
    print(f"  n={n}, names={list(names) if names else []}, ret={ret}")

    print("\n─── Step 5: SetISection ─────────────────────────────")
    ret = m.PropFrame.SetISection(_SEC, _MAT, 400.0, 200.0, 13.0, 8.0, 200.0, 13.0)
    step("SetISection", ret)

    print("\n─── Step 6: GetNameList (frame sections) ────────────")
    n, names, ret = m.PropFrame.GetNameList(0, [])
    print(f"  n={n}, names={list(names) if names else []}, ret={ret}")

    print("\n─── Step 7: AddCartesian ────────────────────────────")
    n1_name, ret = m.PointObj.AddCartesian(0.0, 0.0, 0.0, "N1")
    print(f"  N1 → name='{n1_name}', ret={ret}")

    n2_name, ret = m.PointObj.AddCartesian(3000.0, 0.0, 0.0, "N2")
    print(f"  N2 → name='{n2_name}', ret={ret}")

    print("\n─── Step 8: GetNameList (joints) ────────────────────")
    n, names, ret = m.PointObj.GetNameList(0, [])
    print(f"  n={n}, names={list(names) if names else []}, ret={ret}")

    print(f"\n─── Step 9: AddByPoint (PropName='{_SEC}') ──────────")
    e1_name, ret = m.FrameObj.AddByPoint(n1_name, n2_name, "E1", _SEC, "")
    print(f"  E1 → name='{e1_name}', ret={ret}")

    print(f"\n─── Step 9b: AddByPoint (PropName='Default') ────────")
    m.InitializeNewModel(_N_MM_C)
    m.File.NewBlank()
    m.PropMaterial.SetMaterial(_MAT, _STEEL)
    m.PropMaterial.SetMPIsotropic(_MAT, 210000.0, 0.3, 1.2e-5)
    m.PropFrame.SetISection(_SEC, _MAT, 400.0, 200.0, 13.0, 8.0, 200.0, 13.0)
    m.PointObj.AddCartesian(0.0, 0.0, 0.0, "N1")
    m.PointObj.AddCartesian(3000.0, 0.0, 0.0, "N2")
    e1b_name, ret = m.FrameObj.AddByPoint("N1", "N2", "E1b", "Default", "")
    print(f"  E1b (Default) → name='{e1b_name}', ret={ret}")

    print("\n─── Step 10: GetNameList (frames) ───────────────────")
    n, names, ret = m.FrameObj.GetNameList(0, [])
    print(f"  n={n}, names={list(names) if names else []}, ret={ret}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    try:
        client = ETABSClient.launch(visible=True) if args.launch else ETABSClient.attach()
        print("Connected ✓")
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

    try:
        run(client)
    except Exception as e:
        import traceback
        print(f"\n❌ {e}")
        traceback.print_exc()
    finally:
        input("\n[Enter] to close ETABS …")
        if args.launch:
            client.close()


if __name__ == "__main__":
    main()

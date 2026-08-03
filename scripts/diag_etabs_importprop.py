"""ETABS PropFrame.ImportProp + GetSectProps 단위 동작 체계적 진단.

KoreanKS21.xml은 cm 단위로 단면을 정의한다 (<LENGTH_UNITS>cm</LENGTH_UNITS>).
ETABS COM API의 ImportProp이 cm 라이브러리를 다양한 모델 단위에 적재할 때
어떻게 동작하는지 세 가지 시나리오로 검증한다.

각 테스트는 완전히 깨끗한 모델 상태에서 시작 (InitializeNewModel + File.NewBlank).

사용법:
    .\\opensees-mcp\\Scripts\\python.exe scripts/diag_etabs_importprop.py --launch
    .\\opensees-mcp\\Scripts\\python.exe scripts/diag_etabs_importprop.py            # attach
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "mcp-server"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.etabs_api import ETABSClient   # noqa: E402

# ── ETABS unit codes ──────────────────────────────────────────────────────
_KN_M_C  = 6      # kN, m, °C   — library's natural SI base
_N_MM_C  = 10     # N, mm, °C   — our benchmark working units
_STEEL   = 1

# ── KS test section ───────────────────────────────────────────────────────
_KS_LIB  = "KoreanKS21.xml"
_KS_SEC  = "H 400x200x8/13"

# Expected KS D 3502 tabulated values (per unit system)
_EXPECTED = {
    "kN-m": {"A": 8.412e-3, "I33": 2.37e-4, "J": 3.57e-6},   # m², m⁴
    "N-mm": {"A": 8412.0,    "I33": 2.37e8,  "J": 3.57e5},   # mm², mm⁴
}


def _setup_material(model, e_in_current_units: float) -> None:
    """SS275 with E in whatever units the model is currently in."""
    model.PropMaterial.SetMaterial("SS275", _STEEL)
    model.PropMaterial.SetMPIsotropic("SS275", e_in_current_units, 0.3, 1.2e-5)


def _get_props(model, name: str) -> dict:
    """Returns full GetSectProps tuple as a dict."""
    p = model.PropFrame.GetSectProps(name)
    return {
        "A":   p[0], "As2": p[1], "As3": p[2], "J": p[3],
        "I22": p[4], "I33": p[5], "S22": p[6], "S33": p[7],
        "Z22": p[8], "Z33": p[9], "R22": p[10], "R33": p[11],
        "ret": p[-1],
    }


def _print_props(actual: dict, expected: dict, unit_label: str) -> bool:
    """Pretty-print actual vs expected.  Returns True if A and I33 are within 1%."""
    print(f"  GetSectProps ret = {actual['ret']}")
    print(f"  Returned tuple:")
    print(f"    A    = {actual['A']:>16,.6g}     (expected {expected['A']:>12,.6g})")
    print(f"    I33  = {actual['I33']:>16,.6g}     (expected {expected['I33']:>12,.6g})")
    print(f"    J    = {actual['J']:>16,.6g}     (expected {expected['J']:>12,.6g})")
    print(f"    I22  = {actual['I22']:>16,.6g}")
    print(f"    S33  = {actual['S33']:>16,.6g}     Z33 = {actual['Z33']:>14,.6g}")

    ok = True
    if expected["A"] > 0 and abs(actual["A"]) > 0:
        a_pct = abs(actual["A"] - expected["A"]) / expected["A"] * 100
        i_pct = abs(actual["I33"] - expected["I33"]) / expected["I33"] * 100
        verdict_A = "PASS ✓" if a_pct < 1.0 else f"FAIL (off by {a_pct:.1f}%)"
        verdict_I = "PASS ✓" if i_pct < 1.0 else f"FAIL (off by {i_pct:.1f}%)"
        print(f"  A   match ({unit_label}): {verdict_A}")
        print(f"  I33 match ({unit_label}): {verdict_I}")
        ok = a_pct < 1.0 and i_pct < 1.0
    elif actual["A"] == 0:
        print(f"  → ALL ZEROS — ImportProp didn't populate the section (or GetSectProps doesn't read it)")
        ok = False
    else:
        # Maybe wrong unit scaling — report the ratio
        ratio = actual["A"] / expected["A"]
        print(f"  → A ratio to expected ({unit_label}): {ratio:.2e}")
        ok = False
    return ok


def test1_all_knm(client) -> bool:
    """Pure kN-m throughout: init/material/ImportProp/GetSectProps."""
    print("\n" + "═" * 72)
    print("  Test 1: 처음부터 끝까지 kN-m (라이브러리 SI base와 일치)")
    print("═" * 72)
    m = client.model
    m.InitializeNewModel(_KN_M_C)
    m.File.NewBlank()
    m.SetPresentUnits(_KN_M_C)
    _setup_material(m, 210_000_000.0)   # E = 2.1e8 kN/m² (= 210 GPa = 210000 N/mm²)
    ret = m.PropFrame.ImportProp("H400x200_T1", "SS275", _KS_LIB, _KS_SEC)
    print(f"  ImportProp('H400x200_T1', 'SS275', '{_KS_LIB}', '{_KS_SEC}') ret = {ret}")
    actual = _get_props(m, "H400x200_T1")
    return _print_props(actual, _EXPECTED["kN-m"], "kN-m")


def test2_knm_then_nmm(client) -> bool:
    """Test 1 직후 N-mm으로 단위만 전환 → 같은 단면 재조회."""
    print("\n" + "═" * 72)
    print("  Test 2: Test 1 직후 N-mm으로 전환 → 같은 단면을 N-mm로 재조회")
    print("═" * 72)
    m = client.model
    m.SetPresentUnits(_N_MM_C)
    actual = _get_props(m, "H400x200_T1")
    return _print_props(actual, _EXPECTED["N-mm"], "N-mm")


def test3_all_nmm(client) -> bool:
    """Pure N-mm throughout: 새 모델 시작해서 처음부터 N-mm."""
    print("\n" + "═" * 72)
    print("  Test 3: 새 모델로 처음부터 끝까지 N-mm (이전 실패 패턴 재현)")
    print("═" * 72)
    m = client.model
    m.InitializeNewModel(_N_MM_C)
    m.File.NewBlank()
    m.SetPresentUnits(_N_MM_C)
    _setup_material(m, 210_000.0)        # E = 210000 N/mm²
    ret = m.PropFrame.ImportProp("H400x200_T3", "SS275", _KS_LIB, _KS_SEC)
    print(f"  ImportProp('H400x200_T3', 'SS275', '{_KS_LIB}', '{_KS_SEC}') ret = {ret}")
    actual = _get_props(m, "H400x200_T3")
    return _print_props(actual, _EXPECTED["N-mm"], "N-mm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    print("\n" + "═" * 72)
    print("  ETABS ImportProp/GetSectProps 단위 동작 진단")
    print("  Library: KoreanKS21.xml   Section: H 400x200x8/13")
    print("═" * 72)

    try:
        client = ETABSClient.launch(visible=True) if args.launch else ETABSClient.attach()
        print("\nConnected ✓")
    except Exception as e:
        print(f"\n❌ ETABS 연결 실패: {e}")
        sys.exit(1)

    try:
        results = {
            "Test 1 (kN-m 전체)":          test1_all_knm(client),
            "Test 2 (Test1 + N-mm 전환)": test2_knm_then_nmm(client),
            "Test 3 (N-mm 전체)":          test3_all_nmm(client),
        }

        print("\n" + "═" * 72)
        print("  결과 요약")
        print("═" * 72)
        for label, passed in results.items():
            mark = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {mark}   {label}")

        print("\n  해석:")
        if results["Test 1 (kN-m 전체)"]:
            print("    • kN-m 환경에서는 ImportProp + GetSectProps 정상 →")
            print("      → 향후 librarian helper: 일시적으로 kN-m로 import → 우리가 mm 변환")
        if results["Test 2 (Test1 + N-mm 전환)"]:
            print("    • 단위 전환만 해도 ETABS가 자동 변환 →")
            print("      → 모델은 N-mm로 두고 ImportProp 결과 그대로 사용 가능")
        if results["Test 3 (N-mm 전체)"]:
            print("    • N-mm으로도 ImportProp이 깔끔하게 동작 →")
            print("      → 이전 실패는 material/units 순서 버그였을 가능성")
        if not any(results.values()):
            print("    • 어떤 조합도 정확한 값을 못 줌 →")
            print("      → COM API ImportProp 자체가 cm 라이브러리에 깨짐 (XML 파싱 우회 필요)")

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

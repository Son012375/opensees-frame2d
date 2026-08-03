"""
ETABS 23 Python API connector (comtypes + TLB 방식).

사용 예:
    from mcp_server.core.etabs_api import ETABSClient

    # 실행 중인 ETABS에 연결 (attach)
    client = ETABSClient.attach()

    # ETABS를 새로 실행 (launch)
    client = ETABSClient.launch()

    # context manager 지원
    with ETABSClient.launch() as client:
        client.set_units("kN_m_C")
        client.run_analysis()
        drifts = client.get_story_drifts()
"""

import sys
from pathlib import Path
from typing import Optional

ETABS_INSTALL_DIR = Path(r"C:\Program Files\Computers and Structures\ETABS 23")
_TLB_PATH = ETABS_INSTALL_DIR / "NativeAPI" / "x64" / "ETABSv1.tlb"
ETABS_EXE = ETABS_INSTALL_DIR / "ETABS.exe"

# eUnits enum (ETABS API §eUnits)
UNITS = {
    "lb_in_F":  1,
    "lb_ft_F":  2,
    "kip_in_F": 3,
    "kip_ft_F": 4,
    "kgf_m_C":  5,
    "kN_m_C":   6,   # SI 권장
    "tf_m_C":   7,
    "kN_mm_C":  8,
    "kgf_mm_C": 9,
    "N_mm_C":  10,
}


def _get_etabs_lib():
    """comtypes로 ETABSv1 TLB 모듈 로드 (최초 호출 시 gen 캐시 생성)."""
    try:
        import comtypes.client  # noqa: F401
    except ImportError:
        raise ImportError(
            "comtypes가 설치되지 않았습니다. pip install comtypes 를 실행하세요."
        )
    if not _TLB_PATH.exists():
        raise FileNotFoundError(
            f"ETABS TLB를 찾을 수 없습니다: {_TLB_PATH}\n"
            "ETABS 23이 설치되어 있는지 확인하세요."
        )
    return comtypes.client.GetModule(str(_TLB_PATH))


def _make_helper():
    """ETABSv1.Helper CoClass 인스턴스를 cHelper 인터페이스로 반환."""
    import comtypes.client
    lib = _get_etabs_lib()
    return comtypes.client.CreateObject(lib.Helper, interface=lib.cHelper)


class ETABSClient:
    """ETABS SapModel 래퍼.

    Attributes:
        model: cSapModel — 모든 ETABS API의 진입점
    """

    def __init__(self, etabs_object, sap_model):
        self._etabs_object = etabs_object  # cOAPI / ETABSObject
        self.model = sap_model             # cSapModel

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def attach(cls) -> "ETABSClient":
        """이미 실행 중인 ETABS 인스턴스에 연결.

        Raises:
            RuntimeError: ETABS가 실행 중이지 않을 때
        """
        helper = _make_helper()
        try:
            etabs_obj = helper.GetObject("CSI.ETABS.API.ETABSObject")
        except Exception as e:
            raise RuntimeError(
                f"실행 중인 ETABS를 찾을 수 없습니다: {e}\n"
                "ETABS를 먼저 열거나 ETABSClient.launch()를 사용하세요."
            ) from e
        if etabs_obj is None:
            raise RuntimeError(
                "실행 중인 ETABS를 찾을 수 없습니다.\n"
                "ETABS를 먼저 열거나 ETABSClient.launch()를 사용하세요."
            )
        sap_model = etabs_obj.SapModel
        return cls(etabs_obj, sap_model)

    @classmethod
    def launch(
        cls,
        model_path: Optional[str] = None,
        visible: bool = True,
    ) -> "ETABSClient":
        """ETABS를 새로 실행하고 연결.

        Args:
            model_path: 열 .edb 파일 경로 (None → 새 빈 모델)
            visible:    GUI 표시 여부
        """
        if not ETABS_EXE.exists():
            raise FileNotFoundError(f"ETABS 실행 파일을 찾을 수 없습니다: {ETABS_EXE}")

        helper = _make_helper()
        try:
            etabs_obj = helper.CreateObject(str(ETABS_EXE))
        except Exception as e:
            raise RuntimeError(f"ETABS 실행 실패: {e}") from e

        etabs_obj.ApplicationStart()
        sap_model = etabs_obj.SapModel
        sap_model.InitializeNewModel()

        if not visible:
            sap_model.SetModelIsLocked(False)

        if model_path:
            ret = sap_model.File.OpenFile(model_path)
            if ret != 0:
                raise RuntimeError(f"모델 파일 열기 실패 (ret={ret}): {model_path}")

        return cls(etabs_obj, sap_model)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self, save: bool = False, save_path: str = ""):
        """연결 종료. save=True이면 저장 후 종료."""
        try:
            if save:
                self.model.File.Save(save_path)
            self._etabs_object.ApplicationExit(False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Convenience sub-object shortcuts
    # ------------------------------------------------------------------

    @property
    def frame(self):
        return self.model.FrameObj

    @property
    def point(self):
        return self.model.PointObj

    @property
    def area(self):
        return self.model.AreaObj

    @property
    def load_patterns(self):
        return self.model.LoadPatterns

    @property
    def load_cases(self):
        return self.model.LoadCases

    @property
    def story(self):
        return self.model.Story

    @property
    def analyze(self):
        return self.model.Analyze

    @property
    def results(self):
        return self.model.Results

    @property
    def database(self):
        return self.model.DatabaseTables

    # ------------------------------------------------------------------
    # Unit helpers
    # ------------------------------------------------------------------

    def set_units(self, unit_key: str = "kN_m_C") -> None:
        """모델 단위 설정."""
        val = UNITS.get(unit_key)
        if val is None:
            raise ValueError(f"지원 단위 키: {list(UNITS)}")
        self.model.SetPresentUnits(val)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def run_analysis(self) -> None:
        """해석 실행."""
        ret = self.model.Analyze.RunAnalysis()
        if ret != 0:
            raise RuntimeError(f"해석 실행 실패 (ret={ret})")

    # ------------------------------------------------------------------
    # Result extraction helpers
    # ------------------------------------------------------------------

    def get_base_reactions(self, load_cases: Optional[list] = None) -> dict:
        """기저 반력 (kN, m, °C 단위 권장).

        Args:
            load_cases: 출력할 하중케이스 이름 목록. None이면 전체 선택.

        Returns:
            {'load_case': [...], 'Fx': [...], ..., 'Mz': [...]}
        """
        setup = self.model.Results.Setup
        setup.DeselectAllCasesAndCombosForOutput()

        if load_cases is None:
            n, names, _ = self.model.LoadCases.GetNameList(0, [])
            load_cases = list(names) if names else []

        for name in load_cases:
            setup.SetCaseSelectedForOutput(name, True)

        # BaseReact: 13 in/out params + retval
        # NumberResults, LoadCase, StepType, StepNum, FX, FY, FZ, MX, My, MZ, GX, GY, GZ, ret
        (n, lc, step_type, step_num,
         fx, fy, fz, mx, my, mz,
         gx, gy, gz, ret) = self.model.Results.BaseReact(
            0, [], [], [], [], [], [], [], [], [], 0.0, 0.0, 0.0
        )
        if ret != 0:
            raise RuntimeError(f"기저 반력 추출 실패 (ret={ret})")
        return {
            "load_case": list(lc) if lc else [],
            "Fx": list(fx) if fx else [], "Fy": list(fy) if fy else [], "Fz": list(fz) if fz else [],
            "Mx": list(mx) if mx else [], "My": list(my) if my else [], "Mz": list(mz) if mz else [],
        }

    def get_modal_periods(self) -> list:
        """모달 주기 리스트 (초)."""
        # ModalPeriod: 8 in/out params + retval
        (n, lc, step_type, step_num,
         period, freq, circ_freq, eig_val, ret) = self.model.Results.ModalPeriod(
            0, [], [], [], [], [], [], []
        )
        if ret != 0:
            raise RuntimeError(f"모달 주기 추출 실패 (ret={ret})")
        return list(period) if period else []

    def get_story_drifts(self) -> list:
        """층간변위비 리스트.

        Returns:
            [{'story': str, 'load_case': str, 'direction': str, 'drift': float}, ...]
        """
        # StoryDrifts: 11 in/out params + retval
        (n, story, lc, step_type, step_num,
         direction, drift, label, x, y, z, ret) = self.model.Results.StoryDrifts(
            0, [], [], [], [], [], [], [], [], [], []
        )
        if ret != 0:
            raise RuntimeError(f"층간변위 추출 실패 (ret={ret})")
        if not story:
            return []
        return [
            {
                "story": story[i],
                "load_case": lc[i],
                "direction": direction[i],
                "drift": drift[i],
            }
            for i in range(n)
        ]

    def get_joint_displacements(self, joint_name: str, load_case: str) -> dict:
        """특정 절점의 6-DOF 변위."""
        setup = self.model.Results.Setup
        setup.DeselectAllCasesAndCombosForOutput()
        setup.SetCaseSelectedForOutput(load_case, True)
        # JointDispl: Name(in), ItemTypeElm(in), 12 in/out params + retval
        (n, obj, elm, lc_out, step_type, step_num,
         u1, u2, u3, r1, r2, r3, ret) = self.model.Results.JointDispl(
            joint_name, 0,
            0, [], [], [], [], [], [], [], [], [], []
        )
        if ret != 0:
            raise RuntimeError(f"절점 변위 추출 실패 (ret={ret})")
        return {
            "U1": list(u1) if u1 else [], "U2": list(u2) if u2 else [], "U3": list(u3) if u3 else [],
            "R1": list(r1) if r1 else [], "R2": list(r2) if r2 else [], "R3": list(r3) if r3 else [],
        }

    def get_frame_forces(self, frame_name: str, load_case: str) -> dict:
        """부재력 (P, V2, V3, T, M2, M3) — 부재 국소 좌표계."""
        setup = self.model.Results.Setup
        setup.DeselectAllCasesAndCombosForOutput()
        setup.SetCaseSelectedForOutput(load_case, True)
        # FrameForce: Name(in), ItemTypeElm(in), 14 in/out params + retval
        (n, obj, obj_sta, elm, elm_sta, lc_out, step_type, step_num,
         p, v2, v3, t, m2, m3, ret) = self.model.Results.FrameForce(
            frame_name, 0,
            0, [], [], [], [], [], [], [],
            [], [], [], [], [], []
        )
        if ret != 0:
            raise RuntimeError(f"부재력 추출 실패 (ret={ret})")
        return {
            "station":  list(obj_sta) if obj_sta else [],
            "P":  list(p)  if p  else [], "V2": list(v2) if v2 else [],
            "V3": list(v3) if v3 else [], "T":  list(t)  if t  else [],
            "M2": list(m2) if m2 else [], "M3": list(m3) if m3 else [],
        }

    # ------------------------------------------------------------------
    # Model info helpers
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict:
        """모델 기본 정보 요약."""
        n_pts, pts, _ = self.model.PointObj.GetNameList(0, [])
        n_fr,  fr,  _ = self.model.FrameObj.GetNameList(0, [])
        n_ar,  ar,  _ = self.model.AreaObj.GetNameList(0, [])
        n_lp,  lp,  _ = self.model.LoadPatterns.GetNameList(0, [])

        # GetStories: 8 in/out params + retval
        (n_st, st, base_e, heights,
         is_m, sim, sp_a, sp_h, _) = self.model.Story.GetStories(
            0, [], [], [], [], [], [], []
        )

        return {
            "n_joints":        n_pts,
            "n_frames":        n_fr,
            "n_areas":         n_ar,
            "n_load_patterns": n_lp,
            "load_patterns":   list(lp) if lp else [],
            "n_stories":       n_st,
            "stories":         list(st) if st else [],
        }

    # ------------------------------------------------------------------
    # Database table helper (범용 결과 추출)
    # ------------------------------------------------------------------

    def get_table(self, table_key: str) -> list[dict]:
        """ETABS 데이터베이스 테이블을 dict 리스트로 반환.

        table_key 예: "Story Drifts", "Modal Participating Mass Ratios"
        전체 키 목록은 설치 경로의 'Table and Field Keys.xml' 참조.

        GetTableForDisplayArray 시그니처:
          TableKey(in), FieldKeyList(in/out), GroupName(in),
          TableVersion(in/out), FieldsKeysIncluded(in/out),
          NumberRecords(in/out), TableData(in/out), retval
        """
        (field_key_list, table_version, fields_included,
         n_records, table_data, ret) = self.model.DatabaseTables.GetTableForDisplayArray(
            table_key, [], "", 0, [], 0, []
        )
        if ret != 0:
            raise RuntimeError(f"테이블 '{table_key}' 추출 실패 (ret={ret})")

        keys = list(fields_included) if fields_included else []
        vals = list(table_data) if table_data else []
        n_f = len(keys)
        if n_f == 0 or n_records == 0:
            return []

        return [
            {keys[j]: vals[i * n_f + j] for j in range(n_f)}
            for i in range(n_records)
        ]

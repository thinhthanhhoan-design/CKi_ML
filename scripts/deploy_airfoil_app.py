from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import gdown

# Patch pathlib.WindowsPath on non-Windows systems to avoid "cannot instantiate 'WindowsPath' on your system" error when unpickling models
import pathlib
if os.name != "nt":
    def _windows_path_new(cls, *args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, str):
                new_args.append(arg.replace("\\", "/"))
            else:
                new_args.append(arg)
        return pathlib.PosixPath(*new_args)
    pathlib.WindowsPath.__new__ = _windows_path_new



import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Patch joblib.load globally to cache loaded models and prevent OOM
import joblib
_original_joblib_load = joblib.load
_loaded_models_cache = {}

def cached_joblib_load(*args, **kwargs):
    filename = args[0] if args else kwargs.get("filename")
    if filename and isinstance(filename, (str, Path)):
        abs_path = os.path.abspath(filename)
        if abs_path in _loaded_models_cache:
            return _loaded_models_cache[abs_path]
        model = _original_joblib_load(*args, **kwargs)
        _loaded_models_cache[abs_path] = model
        return model
    return _original_joblib_load(*args, **kwargs)

joblib.load = cached_joblib_load

os.environ.setdefault("DAY5_DISABLE_NEURALFOIL", "0")
import day5
from day5 import Config, TrustRegionCSTOptimizer

# Legacy Day4 CM artifacts were serialized while the training script ran as
# `__main__`. Streamlit executes this file as module `main`, so expose the
# compatibility classes in both namespaces before any joblib load occurs.
_PICKLE_COMPAT_CLASSES = (
    day5.ResearchConfig,
    day5.StrictFeatureValidator,
    day5.RegimeWiseConformalCalibrator,
    day5.PerRegimeCalibration,
    day5.GeometryOODDetector,
    day5.FlowOODDetector,
    day5.TrustRegionManager,
    day5.BootstrapRegimeQuantileEnsemble,
    day5.Day5CdResearchInterface,
    day5.RegimeSpecificRegressor,
)
for _compat_class in _PICKLE_COMPAT_CLASSES:
    globals()[_compat_class.__name__] = _compat_class
    for _module_name in ("main", "__main__"):
        _module = sys.modules.get(_module_name)
        if _module is not None:
            setattr(_module, _compat_class.__name__, _compat_class)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_OUT = PROJECT_ROOT / "outputs" / "deploy_app"
CL2_ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "day4_cl_2_hgb_improved"


def to_abs(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def use_cl2_artifacts(cfg: Config) -> Config:
    """Lock the deploy app to the validated Day4 CL2 artifact set."""
    cfg.cl_model_path = str(CL2_ARTIFACT_ROOT / "models" / "cl_model.joblib")
    cfg.clmax_model_path = str(CL2_ARTIFACT_ROOT / "models" / "clmax_model.joblib")
    cfg.stall_model_path = str(CL2_ARTIFACT_ROOT / "models" / "stall_model.joblib")
    cfg.cl_feature_columns_json = str(CL2_ARTIFACT_ROOT / "artifacts" / "feature_columns.json")
    return cfg


def required_model_paths(cfg: Config) -> Dict[str, str]:
    return {
        "CL": to_abs(cfg.cl_model_path),
        "CD": to_abs(cfg.cd_model_path),
        "CM": to_abs(cfg.cm_model_path),
        "CLmax": to_abs(cfg.clmax_model_path),
        "Stall": to_abs(cfg.stall_model_path),
        "CL schema": to_abs(cfg.cl_feature_columns_json),
    }


def missing_model_paths(cfg: Config) -> Dict[str, str]:
    return {name: path for name, path in required_model_paths(cfg).items() if not Path(path).exists()}


# Cấu hình ID thư mục Google Drive dùng chung chứa đầy đủ các file mô hình
# Khi điền ID này, ứng dụng sẽ tự động tải các mô hình về ở chế độ nền khi phát hiện thiếu file.
# Định dạng ví dụ: "1a2b3c4d5e6f..." (phần mã ký tự sau /folders/ trong link chia sẻ)
SHARED_FOLDER_DRIVE_ID = "https://drive.google.com/drive/folders/1LyTiC50dQCj0PkfBFuJOiQw1txrbKj4G?usp=drive_link"

# Cấu hình Google Drive File ID hoặc Liên kết chia sẻ mặc định cho các mô hình (dự phòng)
MODEL_DRIVE_IDS = {
    "CL": "",        # Google Drive ID cho cl_model.joblib
    "CD": "",        # Google Drive ID cho day5_cd_v3_interface.pkl
    "CM": "",        # Google Drive ID cho cm_model.pkl
    "CLmax": "",     # Google Drive ID cho clmax_model.joblib
    "Stall": "",     # Google Drive ID cho stall_model.joblib
    "CL schema": "", # Google Drive ID cho feature_columns.json
}


def download_file_from_drive(drive_id: str, output_path: str) -> bool:
    try:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        res = gdown.download(id=drive_id, output=str(out_path), quiet=True)
        return res is not None and out_path.exists()
    except Exception as e:
        st.error(f"Lỗi khi tải xuống từ Google Drive: {e}")
        return False


def download_folder_from_drive(folder_id: str, temp_output_dir: str) -> bool:
    try:
        import gdown
        import re
        clean_id = folder_id.strip()
        if "drive.google.com" in clean_id:
            match = re.search(r"/folders/([a-zA-Z0-9_-]+)", clean_id)
            if match:
                clean_id = match.group(1)
        out_path = Path(temp_output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        res = gdown.download_folder(id=clean_id, output=str(out_path), quiet=True)
        return res is not None
    except Exception as e:
        st.error(f"Lỗi khi tải thư mục từ Google Drive: {e}")
        return False


def organize_downloaded_models(temp_dir: str, miss: Dict[str, str]) -> Dict[str, bool]:
    import shutil
    temp_path = Path(temp_dir)
    found_files = {}
    targets = {os.path.basename(path).lower(): (name, path) for name, path in miss.items()}
    
    for p in temp_path.rglob("*"):
        if p.is_file():
            filename_lc = p.name.lower()
            if filename_lc in targets:
                name, dest_path = targets[filename_lc]
                dest_p = Path(dest_path)
                dest_p.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dest_p))
                found_files[name] = True
    return found_files




def parse_airfoil_dat_bytes(data: bytes) -> np.ndarray:
    rows: List[Tuple[float, float]] = []
    text = data.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        parts = line.strip().replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x_val = float(parts[0])
            y_val = float(parts[1])
        except Exception:
            continue
        if -0.1 <= x_val <= 1.1 and -0.8 <= y_val <= 0.8:
            rows.append((x_val, y_val))
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 20:
        raise ValueError("File khong co du bang toa do airfoil hop le.")
    return arr[:, :2]


def save_uploaded_airfoil(uploaded_file) -> Path:
    upload_dir = APP_OUT / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower() or ".dat"
    path = upload_dir / f"baseline_{int(time.time())}{suffix}"
    path.write_bytes(uploaded_file.getvalue())
    return path


def make_config(
    baseline_path: Path,
    run_dir: Path,
    reynolds: float,
    aoa_values: List[float],
    maxiter: int,
    popsize: int,
    outer_loops: int,
) -> Config:
    cfg = use_cl2_artifacts(Config())
    cfg.baseline_dat_path = str(baseline_path)
    cfg.allow_synthetic_baseline = False
    cfg.output_dir = str(run_dir)
    cfg.reynolds = float(reynolds)
    cfg.aoa_range = [float(v) for v in aoa_values]
    cfg.maxiter = int(maxiter)
    cfg.popsize = int(popsize)
    cfg.outer_loops = int(outer_loops)
    cfg.top_n_export = 5
    cfg.raw_dataset_path = "" # Disable fallback to 4.3 GB raw dataset to avoid OOM
    cfg.geometry_table_path = to_abs(cfg.geometry_table_path)
    cfg.cl_model_path = to_abs(cfg.cl_model_path)
    cfg.cd_model_path = to_abs(cfg.cd_model_path)
    cfg.cm_model_path = to_abs(cfg.cm_model_path)
    cfg.clmax_model_path = to_abs(cfg.clmax_model_path)
    cfg.stall_model_path = to_abs(cfg.stall_model_path)
    cfg.cl_feature_columns_json = to_abs(cfg.cl_feature_columns_json)
    return cfg


@st.cache_resource(show_spinner=False)
def build_optimizer_for_prediction(
    baseline_path: str,
    reynolds: float,
    aoa_csv: str,
) -> TrustRegionCSTOptimizer:
    run_dir = APP_OUT / "preview"
    aoa_values = [float(x.strip()) for x in aoa_csv.split(",") if x.strip()]
    cfg = make_config(Path(baseline_path), run_dir, reynolds, aoa_values, maxiter=1, popsize=3, outer_loops=1)
    day5.NEURALFOIL_AVAILABLE = False
    return TrustRegionCSTOptimizer(cfg)


def plot_geometry(coords: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(coords[:, 0], coords[:, 1], color="#1b4d3e", linewidth=1.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05)
    y_abs_max = max(float(np.abs(coords[:, 1]).max()), 0.15)
    ylim = max(0.18, y_abs_max * 1.25)
    ax.set_ylim(-ylim, ylim)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Hình học cánh cơ sở (Baseline airfoil geometry)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    return buf.getvalue()


def file_download_bytes(path: Path) -> bytes:
    return path.read_bytes()


def make_candidate_zip(run_dir: Path, candidate_label: str) -> bytes:
    dat_path = run_dir / "airfoils" / f"{candidate_label}.dat"
    overlay_path = run_dir / "figures" / f"{candidate_label}_overlay.png"
    if not overlay_path.exists():
        overlay_path = run_dir / "figures" / "top_candidates_geometry_overlay.png"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if dat_path.exists():
            zf.write(dat_path, dat_path.name)
        if overlay_path.exists():
            zf.write(overlay_path, overlay_path.name)
        detailed = run_dir / "tables" / "top_candidates_detailed.csv"
        polars = run_dir / "tables" / "top_candidate_polars.csv"
        if detailed.exists():
            zf.write(detailed, detailed.name)
        if polars.exists():
            zf.write(polars, polars.name)
    return buf.getvalue()


def create_single_candidate_overlay(run_dir: Path, candidate_label: str) -> Path | None:
    baseline = run_dir / "airfoils" / "baseline.dat"
    cand = run_dir / "airfoils" / f"{candidate_label}.dat"
    if not baseline.exists() or not cand.exists():
        return None
    base_coords = parse_airfoil_dat_bytes(baseline.read_bytes())
    cand_coords = parse_airfoil_dat_bytes(cand.read_bytes())
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(base_coords[:, 0], base_coords[:, 1], "k--", linewidth=1.7, label="Cánh gốc (baseline)")
    ax.plot(cand_coords[:, 0], cand_coords[:, 1], color="#c84b31", linewidth=1.8, label=f"Cánh tối ưu ({candidate_label})")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05)
    y_abs_max = max(
        float(np.abs(base_coords[:, 1]).max()),
        float(np.abs(cand_coords[:, 1]).max()),
        0.15
    )
    ylim = max(0.18, y_abs_max * 1.25)
    ax.set_ylim(-ylim, ylim)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title(f"So sánh Cánh gốc vs Cánh tối ưu {candidate_label}")
    fig.tight_layout()
    out = run_dir / "figures" / f"{candidate_label}_overlay.png"
    fig.savefig(out, format="png", dpi=180)
    plt.close(fig)
    return out


def run_optimization(baseline_path: Path, reynolds: float, aoa_csv: str, maxiter: int, popsize: int, outer_loops: int) -> Path:
    run_dir = APP_OUT / "runs" / time.strftime("%Y%m%d_%H%M%S")
    aoa_values = [float(x.strip()) for x in aoa_csv.split(",") if x.strip()]
    cfg = make_config(baseline_path, run_dir, reynolds, aoa_values, maxiter, popsize, outer_loops)
    # Enable NeuralFoil for backend report/plot generation in day5.py if available
    try:
        import neuralfoil as nf
        day5.NEURALFOIL_AVAILABLE = nf is not None
    except ImportError:
        day5.NEURALFOIL_AVAILABLE = False
    opt = TrustRegionCSTOptimizer(cfg)
    opt.run()
    # Force garbage collection to free temporary optimization objects
    import gc
    gc.collect()
    top_path = run_dir / "tables" / "top_candidates_detailed.csv"
    if not top_path.exists():
        failure_reports = sorted((run_dir / "reports").glob("*failure*.md"))
        details = failure_reports[-1].read_text(encoding="utf-8", errors="ignore") if failure_reports else ""
        raise RuntimeError(
            "Bộ tối ưu hóa không tạo được danh sách Top 5."
            + (f"\n\n{details}" if details else " Không tìm thấy báo cáo lỗi (failure report).")
        )
    return run_dir


st.set_page_config(page_title="Airfoil Optimizer", page_icon="A", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f7f8f4; }
    .metric-card {
        border: 1px solid #d7ddd2;
        padding: 0.85rem 1rem;
        border-radius: 8px;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Bộ tối ưu hóa Mô hình thay thế Cánh máy bay (Airfoil Surrogate Optimizer)")
st.caption(
    "Tải lên tệp tọa độ .dat/.txt, dự báo CL/CD/CM, sau đó tối ưu hóa biên dạng CST trơn "
    "với manifold Day 2 và các ràng buộc CAD nghiêm ngặt."
)

# Kiểm tra thiếu mô hình ở mức toàn cục trước khi render các thành phần khác
base_cfg = use_cl2_artifacts(Config())
miss = missing_model_paths(base_cfg)

if miss:
    # 1. Tự động tải từ SHARED_FOLDER_DRIVE_ID nếu được cấu hình sẵn
    if SHARED_FOLDER_DRIVE_ID.strip():
        if "auto_download_attempted" not in st.session_state:
            st.session_state["auto_download_attempted"] = False

        if not st.session_state["auto_download_attempted"]:
            st.session_state["auto_download_attempted"] = True
            st.info("🔄 **Phát hiện thiếu các tệp mô hình. Đang tiến hành tự động tải từ Google Drive Folder cấu hình sẵn...**")
            
            temp_dir = APP_OUT / "temp_download"
            if temp_dir.exists():
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
            
            with st.spinner("Đang tự động tải toàn bộ thư mục mô hình từ Google Drive (có thể mất vài phút do dung lượng lớn)..."):
                success = download_folder_from_drive(SHARED_FOLDER_DRIVE_ID.strip(), str(temp_dir))
                
            if success:
                found_files = organize_downloaded_models(str(temp_dir), miss)
                still_missing = [n for n in miss.keys() if n not in found_files]
                
                if not still_missing:
                    st.success("🎉 Tự động tải xuống và sắp xếp tất cả các mô hình thành công!")
                    if temp_dir.exists():
                        import shutil
                        try:
                            shutil.rmtree(temp_dir)
                        except Exception:
                            pass
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error(
                        f"❌ Đã tải thư mục tự động thành công nhưng thiếu các tệp sau trong thư mục: {', '.join(still_missing)}. "
                        "Vui lòng kiểm tra lại xem bạn đã upload đầy đủ các tệp mô hình lên thư mục Google Drive chưa."
                    )
            else:
                st.error("❌ Tự động tải thư mục từ Google Drive thất bại! Vui lòng kiểm tra lại cấu hình SHARED_FOLDER_DRIVE_ID và quyền chia sẻ của thư mục (phải ở chế độ 'Bất kỳ ai có liên kết đều có thể xem').")

    # 2. Giao diện tải thủ công dự phòng nếu không tự động tải được hoặc không cấu hình
    st.warning("⚠️ **Thiếu các tệp mô hình (model artifacts) để chạy ứng dụng!**")
    st.write(
        "Do giới hạn kích thước tệp của GitHub (100MB), các tệp mô hình lớn không được tải lên kho lưu trữ. "
        "Bạn hãy tải tất cả các tệp mô hình bị thiếu lên **cùng một thư mục Google Drive**, bật chia sẻ công khai "
        "(*'Bất kỳ ai có liên kết đều có thể xem'*), sau đó dán liên kết của thư mục đó vào ô bên dưới để ứng dụng tự động tải về."
    )
    
    with st.expander("Danh sách các tệp mô hình cần có trong thư mục Google Drive:"):
        for name, path in miss.items():
            st.markdown(f"- **{name}**: `{os.path.basename(path)}`")
            
    with st.form("download_models_folder_form"):
        st.subheader("Cách 1 (Khuyên dùng): Tải toàn bộ bằng liên kết Thư mục Google Drive")
        folder_input = st.text_input(
            "Liên kết Thư mục Google Drive chứa các mô hình:",
            placeholder="Dán liên kết thư mục (ví dụ: https://drive.google.com/drive/folders/...)",
            help="Đảm bảo thư mục này chứa đầy đủ các tệp mô hình ở trên và được chia sẻ công khai."
        )
        
        submitted_folder = st.form_submit_button("Tải xuống và tự động sắp xếp", type="primary")
        
        if submitted_folder:
            if not folder_input.strip():
                st.error("Vui lòng nhập liên kết thư mục Google Drive!")
            else:
                raw_input = folder_input.strip()
                folder_id = raw_input
                if "drive.google.com" in raw_input:
                    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", raw_input)
                    if match:
                        folder_id = match.group(1)
                
                temp_dir = APP_OUT / "temp_download"
                if temp_dir.exists():
                    import shutil
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
                
                with st.spinner("Đang tải toàn bộ thư mục từ Google Drive (quá trình này có thể mất vài phút do dung lượng các mô hình lớn)..."):
                    success = download_folder_from_drive(folder_id, str(temp_dir))
                    
                if success:
                    found_files = organize_downloaded_models(str(temp_dir), miss)
                    still_missing = [n for n in miss.keys() if n not in found_files]
                    
                    if not still_missing:
                        st.balloons()
                        st.success("🎉 Tải xuống và tự động sắp xếp tất cả các mô hình thành công!")
                        if temp_dir.exists():
                            import shutil
                            try:
                                shutil.rmtree(temp_dir)
                            except Exception:
                                pass
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(
                            f"Đã tải thư mục thành công nhưng thiếu các tệp sau trong thư mục: {', '.join(still_missing)}. "
                            "Vui lòng kiểm tra lại xem bạn đã upload đầy đủ các tệp mô hình lên thư mục Google Drive chưa."
                        )
                else:
                    st.error("❌ Tải thư mục thất bại! Vui lòng kiểm tra lại quyền chia sẻ thư mục Google Drive (phải ở chế độ 'Bất kỳ ai có liên kết đều có thể xem').")

    with st.expander("Cách 2: Tải thủ công từng tệp mô hình bằng liên kết riêng lẻ (Nâng cao)"):
        with st.form("download_individual_models_form"):
            drive_inputs = {}
            for name, path in miss.items():
                default_val = MODEL_DRIVE_IDS.get(name, "")
                drive_inputs[name] = st.text_input(
                    f"Liên kết hoặc File ID cho **{name}** (`{os.path.basename(path)}`):",
                    value=default_val
                )
            submitted_ind = st.form_submit_button("Tải xuống từng tệp")
            if submitted_ind:
                empty_fields = [n for n, val in drive_inputs.items() if not val.strip()]
                if empty_fields:
                    st.error(f"Vui lòng nhập đầy đủ thông tin cho: {', '.join(empty_fields)}")
                else:
                    success_count = 0
                    for name, path in miss.items():
                        raw_input = drive_inputs[name].strip()
                        drive_id = raw_input
                        if "drive.google.com" in raw_input:
                            match = re.search(r"/d/([a-zA-Z0-9_-]+)", raw_input)
                            if match:
                                drive_id = match.group(1)
                            else:
                                match_id = re.search(r"id=([a-zA-Z0-9_-]+)", raw_input)
                                if match_id:
                                    drive_id = match_id.group(1)
                        
                        with st.spinner(f"Đang tải {name} ({os.path.basename(path)})..."):
                            if download_file_from_drive(drive_id, path):
                                success_count += 1
                                st.success(f"✅ Đã tải xong {name}!")
                            else:
                                st.error(f"❌ Tải {name} thất bại! Vui lòng kiểm tra lại link file.")
                                
                    if success_count == len(miss):
                        st.balloons()
                        st.success("Tải xuống tất cả các mô hình thành công! Đang tải lại ứng dụng...")
                        time.sleep(2)
                        st.rerun()
    st.stop()



with st.sidebar:
    st.header("Cấu hình chạy (Run Settings)")
    reynolds = st.number_input("Số Reynolds (Reynolds)", min_value=10_000.0, max_value=20_000_000.0, value=500_000.0, step=50_000.0)
    aoa_csv = st.text_input("Danh sách góc tấn AoA", value="0,2,4,6,8")
    maxiter = st.number_input("Số lần lặp tối đa (maxiter)", min_value=1, max_value=80, value=15, step=1)
    popsize = st.number_input("Kích thước quần thể (popsize)", min_value=2, max_value=20, value=8, step=1)
    outer_loops = st.number_input("Số vòng lặp ngoài (outer loops)", min_value=1, max_value=8, value=1, step=1)

    st.success("Tất cả mô hình đã sẵn sàng (CL2 + CD + CM).")
    st.caption("Nguồn mô hình CL: outputs/day4_cl_2_hgb_improved")



uploaded = st.file_uploader("Chọn tệp biên dạng cánh (.dat hoặc .txt)", type=["dat", "txt"])

if uploaded is None:
    st.info("Vui lòng tải lên một tệp biên dạng cánh để bắt đầu.")
    st.stop()

try:
    coords = parse_airfoil_dat_bytes(uploaded.getvalue())
    baseline_path = save_uploaded_airfoil(uploaded)
except Exception as exc:
    st.error(f"Không đọc được tệp biên dạng cánh: {exc}")
    st.stop()

left, right = st.columns([1.2, 1.0])
with left:
    st.subheader("Hình học gốc (Baseline Geometry)")
    st.image(plot_geometry(coords), use_container_width=True)
with right:
    st.subheader("Dự báo gốc (Baseline Prediction)")
    if miss:
        st.warning("Cần huấn luyện/xuất đủ các mô hình trước khi dự báo và tối ưu hóa.")
    else:
        try:
            opt_preview = build_optimizer_for_prediction(str(baseline_path), float(reynolds), aoa_csv)
            baseline_polar = opt_preview.baseline_polar.copy()
            st.dataframe(baseline_polar.round(6), use_container_width=True)
            st.dataframe(pd.DataFrame([opt_preview.baseline_summary]).round(6), use_container_width=True)
        except Exception as exc:
            import sklearn
            import sys
            st.error(
                f"Dự báo gốc thất bại: {exc}\n\n"
                f"**Thông tin môi trường chạy Streamlit:**\n"
                f"- Python version: `{sys.version}`\n"
                f"- Scikit-learn version: `{sklearn.__version__}`\n"
                f"- Executable: `{sys.executable}`"
            )

st.divider()

run_disabled = bool(miss)
if st.button("Tối ưu hóa và tạo Top 5 cánh tốt nhất", type="primary", disabled=run_disabled):
    with st.spinner("Đang chạy tối ưu hóa bằng mô hình surrogate đã huấn luyện. Bước này có thể mất vài phút..."):
        try:
            run_dir = run_optimization(baseline_path, float(reynolds), aoa_csv, int(maxiter), int(popsize), int(outer_loops))
            st.session_state["last_run_dir"] = str(run_dir)
        except Exception as exc:
            st.error(f"Tối ưu hóa thất bại: {exc}")

if "last_run_dir" not in st.session_state:
    st.stop()

run_dir = Path(st.session_state["last_run_dir"])
top_path = run_dir / "tables" / "top_candidates_detailed.csv"
polar_path = run_dir / "tables" / "top_candidate_polars.csv"

if not top_path.exists():
    st.warning("Chưa có bảng danh sách ứng viên tối ưu tốt nhất.")
    st.stop()

top_df = pd.read_csv(top_path)
polars = pd.read_csv(polar_path) if polar_path.exists() else pd.DataFrame()

# Check for optimization warnings
warning_path = run_dir / "tables" / "optimization_warning.txt"
if warning_path.exists():
    st.warning("⚠️ **Cảnh báo (Warning):** Không tìm thấy ứng viên cánh nào thỏa mãn toàn bộ các ràng buộc nghiêm ngặt về hình học và độ mượt (CAD, Manifold, Peak Count). Đang hiển thị kết quả cứu hộ tốt nhất (fallback). Bạn nên tăng `maxiter`/`popsize` hoặc nới lỏng các giới hạn chạy.")

# Highlight the absolute best candidate (Rank 1)
best_candidate = top_df.iloc[0]
st.success(f"🏆 **Biên dạng tốt nhất (Best Profile): {best_candidate['candidate_label']}** "
           f"(Tổng điểm: **{best_candidate['total_score']:.4f}**, Cải thiện L/D: **{best_candidate['ld_improvement_pct']:.2f}%**, "
           f"Cải thiện CLmax: **{best_candidate['clmax_improvement_pct']:.2f}%**)")

st.subheader("Top 5 ứng viên tối ưu nhất (Top 5 Optimized Candidates)")
show_cols = [
    "candidate_rank", "candidate_label", "selection_score", "total_score",
    "ld_mean", "ld_improvement_pct", "clmax_improvement_pct", "stall_improvement_deg",
    "mean_shape_delta", "geometry_rms_delta", "shape_diversity_score",
    "cd_mean", "cm_abs_mean", "cm_delta_mean", "cad_score", "manifold_score",
    "curvature_p99_ratio", "oscillation_count",
]
show_cols = [c for c in show_cols if c in top_df.columns]
st.dataframe(top_df[show_cols].round(6), use_container_width=True)

if len(top_df) < 5:
    st.info(
        f"💡 **Thông tin:** Chỉ tìm thấy {len(top_df)} ứng viên thỏa mãn hoàn toàn các ràng buộc nghiêm ngặt "
        f"(bao gồm cải thiện khí động học, kiểm tra độ mượt CAD, Descriptor Manifold, giới hạn Peak Count, và độ bất định). "
        f"Để tìm kiếm được nhiều ứng viên hơn (đủ Top 5), bạn nên tăng **Số lần lặp tối đa (maxiter)** (ví dụ: 15-20) "
        f"hoặc **Kích thước quần thể (popsize)** (ví dụ: 6-10) ở Sidebar bên trái."
    )

fig_overlay = run_dir / "figures" / "top_candidates_geometry_overlay.png"
fig_polars = run_dir / "figures" / "top_candidates_polar_comparison.png"
fig_polars_nf = run_dir / "figures" / "top_candidates_neuralfoil_polar_comparison.png"

# Display Geometry Overlay
if fig_overlay.exists():
    st.subheader("So sánh hình học (Geometry Comparison)")
    st.image(str(fig_overlay), caption="Bản vẽ xếp chồng hình học (Geometry Overlay)", use_container_width=True)

# Display Aerodynamic Polar Comparisons (Surrogate Model Predictions expanded to full width)
st.subheader("So sánh hiệu suất khí động học (Aerodynamic Performance Comparison)")
if fig_polars.exists():
    st.image(str(fig_polars), caption="So sánh CL, CD, CM và L/D (Mô hình Surrogate dự báo)", use_container_width=True)
else:
    st.info("Chưa có biểu đồ so sánh từ mô hình surrogate.")

labels = top_df["candidate_label"].astype(str).tolist()
selected = st.radio("Chọn ứng viên tối ưu để tải về", labels, horizontal=True)
overlay_path = create_single_candidate_overlay(run_dir, selected)

# Function to generate aerodynamic comparison table
def generate_comparison_table(base_metrics, cand_metrics) -> pd.DataFrame:
    rows = []
    # CL
    b_cl = base_metrics.get('cl_mean', 0.0)
    c_cl = cand_metrics.get('cl_mean', 0.0)
    diff_cl = ((c_cl - b_cl) / max(b_cl, 1e-9)) * 100
    rows.append({
        "Chỉ số (Metric)": "Hệ số nâng trung bình (CL mean)",
        "Cánh cơ sở (Baseline)": f"{b_cl:.6f}",
        "Cánh tối ưu (Optimized)": f"{c_cl:.6f}",
        "Thay đổi (Change)": f"{diff_cl:+.2f}%"
    })
    # CD
    b_cd = base_metrics.get('cd_mean', 0.0)
    c_cd = cand_metrics.get('cd_mean', 0.0)
    diff_cd = ((c_cd - b_cd) / max(b_cd, 1e-9)) * 100
    rows.append({
        "Chỉ số (Metric)": "Hệ số cản trung bình (CD mean)",
        "Cánh cơ sở (Baseline)": f"{b_cd:.6f}",
        "Cánh tối ưu (Optimized)": f"{c_cd:.6f}",
        "Thay đổi (Change)": f"{diff_cd:+.2f}%"
    })
    # CM
    b_cm = base_metrics.get('cm_abs_mean', 0.0)
    c_cm = cand_metrics.get('cm_abs_mean', 0.0)
    diff_cm = ((c_cm - b_cm) / max(b_cm, 1e-9)) * 100
    rows.append({
        "Chỉ số (Metric)": "Hệ số mô-men trung bình (CM abs mean)",
        "Cánh cơ sở (Baseline)": f"{b_cm:.6f}",
        "Cánh tối ưu (Optimized)": f"{c_cm:.6f}",
        "Thay đổi (Change)": f"{diff_cm:+.2f}%"
    })
    # L/D
    b_ld = base_metrics.get('ld_mean', 0.0)
    c_ld = cand_metrics.get('ld_mean', 0.0)
    diff_ld = ((c_ld - b_ld) / max(b_ld, 1e-9)) * 100
    rows.append({
        "Chỉ số (Metric)": "Hiệu suất khí động học (L/D mean)",
        "Cánh cơ sở (Baseline)": f"{b_ld:.6f}",
        "Cánh tối ưu (Optimized)": f"{c_ld:.6f}",
        "Thay đổi (Change)": f"{diff_ld:+.2f}%"
    })
    return pd.DataFrame(rows)

col_img, col_tbl = st.columns([1.2, 1.0])
with col_img:
    if overlay_path and overlay_path.exists():
        st.image(str(overlay_path), caption=f"Baseline vs {selected}", use_container_width=True)
with col_tbl:
    st.markdown(f"### So sánh hiệu suất khí động học ({selected} vs Cánh gốc)")
    baseline_summary_path = run_dir / "tables" / "baseline_summary.csv"
    if baseline_summary_path.exists():
        try:
            base_summary_df = pd.read_csv(baseline_summary_path)
            base_metrics = base_summary_df.iloc[0]
            cand_metrics = top_df[top_df["candidate_label"] == selected].iloc[0]
            comparison_df = generate_comparison_table(base_metrics, cand_metrics)
            st.table(comparison_df)
        except Exception as e:
            st.error(f"Lỗi khi hiển thị bảng so sánh: {e}")
    else:
        st.info("Không tìm thấy dữ liệu tóm tắt cánh cơ sở.")

dat_path = run_dir / "airfoils" / f"{selected}.dat"
download_col1, download_col2, download_col3 = st.columns(3)
if dat_path.exists():
    download_col1.download_button(
        "Tải xuống tệp .dat",
        data=file_download_bytes(dat_path),
        file_name=f"{selected}.dat",
        mime="text/plain",
    )
if overlay_path and overlay_path.exists():
    download_col2.download_button(
        "Tải ảnh vẽ đè hình học",
        data=file_download_bytes(overlay_path),
        file_name=f"{selected}_overlay.png",
        mime="image/png",
    )
download_col3.download_button(
    "Tải xuống trọn bộ gói thiết kế (ZIP)",
    data=make_candidate_zip(run_dir, selected),
    file_name=f"{selected}_package.zip",
    mime="application/zip",
)

with st.expander("Thư mục chạy tối ưu hóa (Run folder)"):
    st.code(str(run_dir))
    if polars is not None and not polars.empty:
        st.dataframe(polars.round(6), use_container_width=True)

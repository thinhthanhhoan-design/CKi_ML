# Dự Án Tối Ưu Hóa Biên Dạng Cánh Máy Bay (Airfoil Surrogate Optimizer)

Dự án này là một framework toàn diện ứng dụng Học máy (Machine Learning) làm mô hình thay thế (Surrogate Models) để dự báo và tối ưu hóa hiệu suất khí động học (hệ số nâng CL, hệ số cản CD, hệ số mô-men CM) của biên dạng cánh máy bay sử dụng tham số hóa CST và vùng ràng buộc hình học (Trust Region / Manifold).

---

## 📂 Cấu Trúc Thư Mục Dự Án

Thư mục làm việc của dự án được tổ chức gọn gàng và khoa học như sau:

*   **`scripts/`**: Chứa toàn bộ mã nguồn Python thực thi của dự án từ các bước tiền xử lý, huấn luyện đến triển khai:
    *   `day1.py` đến `day4_*.py`: Các kịch bản huấn luyện mô hình theo từng giai đoạn.
    *   `day5.py`: Bộ tối ưu hóa cánh máy bay chính thức sử dụng CST Trust Region.
    *   `deploy_airfoil_app.py`: Mã nguồn chính của ứng dụng web Streamlit.
    *   `guided_search.py` & `neuralFoil.py`: Các module bổ trợ tối ưu hóa và liên kết với thư viện NeuralFoil.
*   **`test/`**: Chứa các tệp biên dạng cánh dạng `.txt` riêng lẻ phục vụ quá trình chạy thử nghiệm và kiểm thử trên giao diện ứng dụng (ví dụ: `hs1404.txt`, `fx6184.txt`, `NACA2751.txt`...).
*   **`airfoil/`**: Cơ sở dữ liệu chứa hàng ngàn tệp tọa độ biên dạng cánh `.dat` mặc định.
*   **`reports/`**: Lưu trữ các báo cáo markdown kết quả huấn luyện mô hình, tệp nhật ký tối ưu hóa (`.log`) và các tệp phân tích văn bản trung gian.
*   **`tables/`**: Chứa các tệp dữ liệu lớn dạng bảng đã được dọn dẹp và nén gọn gàng (`.csv.gz`).
*   **`outputs/`**: Chứa các kết quả đầu ra của quá trình huấn luyện và chạy thử bao gồm:
    *   `day4_cl_2_hgb_improved/`: Chứa mô hình CL cải tiến (HistGradientBoosting).
    *   `day4c_cd_v3_final/`: Chứa mô hình CD đã được hiệu chuẩn.
    *   `day4_cm_reset/`: Chứa mô hình CM.
    *   `deploy_app/`: Lưu vết các lượt chạy tối ưu hóa (`runs/`) và các tệp tải lên (`uploads/`) của ứng dụng Streamlit.
*   **`requirements.txt`**: Danh sách thư viện phụ thuộc của dự án.
*   **`.gitignore`**: Danh sách các tệp tin dung lượng lớn, thư mục ảo (`.venv`) và dữ liệu nháp được bỏ qua không đưa lên GitHub.

---

## ⚙️ Quy Trình Chạy Pipeline (Day 1 - Day 5)

Dự án được xây dựng theo quy trình phát triển tuần tự 5 ngày:

1.  **Day 1 (Tiền xử lý):** Làm sạch dữ liệu thô, lọc bỏ các dòng lỗi vật lý và nén dữ liệu lớn.
2.  **Day 2 (Thiết lập Manifold):** Tham số hóa biên dạng cánh bằng CST và sử dụng thuật toán MinCovDet + PCA + KDE để xác định vùng an toàn hình học (Descriptor Manifold).
3.  **Day 3 (Huấn luyện CD/CM):** Huấn luyện các mô hình thay thế dự báo CD, CM và thực hiện hiệu chuẩn độ bất định bằng phân vị (Conformal Calibration).
4.  **Day 4 (Tối ưu hóa CL & Vùng tin cậy):** Nâng cấp mô hình dự báo CL bằng thuật toán HistGradientBoosting nâng cao, thiết lập hệ thống cảnh báo vùng OOD (Out-Of-Distribution).
5.  **Day 5 (Tối ưu hóa thiết kế cánh):** Kết hợp các mô hình thay thế và bộ lọc CAD mượt mà để thực hiện tìm kiếm biên dạng cánh tốt nhất thỏa mãn các điều kiện khí động học bằng thuật toán tiến hóa Differential Evolution hoặc Tối ưu hóa Bayes (Optuna).

---

## 💻 Hướng Dẫn Chạy Ứng Dụng Streamlit Cục Bộ (Local)

### 1. Chuẩn bị môi trường
Mở Terminal tại thư mục dự án và kích hoạt môi trường ảo:
```powershell
# Kích hoạt môi trường ảo (Windows)
.\.venv\Scripts\Activate.ps1
```

### 2. Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng
```bash
streamlit run scripts/deploy_airfoil_app.py
```
Sau khi khởi chạy thành công, trình duyệt sẽ tự động mở trang web local tại địa chỉ: `http://localhost:8501`.

---

## 🖱️ Hướng Dẫn Sử Dụng Giao Diện Ứng Dụng

Sau khi ứng dụng đã sẵn sàng (cục bộ hoặc trên Cloud), dưới đây là các bước để thực hiện tối ưu hóa cánh máy bay:

### Bước 1: Tải lên tệp biên dạng cánh máy bay cơ sở (Baseline Airfoil)
- Tại mục **"Chọn tệp biên dạng cánh (.dat hoặc .txt)"**, nhấp chuột chọn hoặc kéo thả tệp tọa độ cánh.
- *Gợi ý*: Các tệp cánh kiểm thử mẫu đã được chuẩn bị sẵn trong thư mục [test/](file:///e:/Project2/test) (ví dụ: `test/NACA2751.txt`, `test/hs1404.txt`).
- Sau khi tải lên thành công, hệ thống sẽ tự động hiển thị bản vẽ hình học cánh ở bên trái và bảng dự báo các chỉ số khí động học cơ sở (CL, CD, CM, L/D) ở bên phải.

### Bước 2: Cấu hình tham số tối ưu hóa (Sidebar bên trái)
- **Số Reynolds**: Phù hợp với điều kiện vận hành thực tế của cánh (mặc định: `500,000`).
- **Danh sách góc tấn AoA**: Nhập các góc tấn phân tách bằng dấu phẩy (ví dụ: `0,2,4,6,8`).
- **Số lần lặp tối đa (maxiter)**: Quy định số thế hệ tiến hóa. Khuyên dùng đặt từ **`5` đến `15`** để tối ưu hóa tốc độ chạy trên web app.
- **Kích thước quần thể (popsize)**: Số lượng ứng viên trong mỗi quần thể. Khuyên dùng đặt từ **`4` đến `8`**.
- **Số vòng lặp ngoài**: Số lần chạy mở rộng vùng tin cậy (mặc định: `1`).

### Bước 3: Chạy tối ưu hóa
- Nhấp vào nút **"Tối ưu hóa và tạo Top 5 cánh tốt nhất"** màu đỏ ở cuối trang chính.
- Hệ thống sẽ chạy thuật toán tiến hóa kết hợp lọc các ràng buộc CAD mượt và Manifold an toàn. Quá trình tính toán thường mất từ **1 đến 3 phút**.

### Bước 4: Xem và tải về kết quả thiết kế
- **Cánh tốt nhất (Best Profile)**: Hiển thị nổi bật ở trên cùng kèm điểm cải thiện khí động học (%).
- **Bảng Top 5 ứng viên**: So sánh chi tiết tất cả các chỉ số chất lượng thiết kế của 5 ứng viên hàng đầu.
- **Biểu đồ so sánh**: Xem bản vẽ hình học cánh xếp chồng và biểu đồ so sánh đường đặc tính khí động học (Polar).
- **Tải tệp tin**:
  - Tải riêng lẻ tệp tọa độ `.dat` của cánh tối ưu.
  - Tải ảnh so sánh biên dạng hình học (`.png`).
  - Tải trọn bộ gói thiết kế dạng nén `.zip` (chứa toàn bộ file `.dat` ứng viên, polars khí động học, bảng so sánh chi tiết dạng `.csv`).


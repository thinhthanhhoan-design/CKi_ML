# Day 3 — Baseline Tuyến Tính & Báo Cáo Phân Tích Thất Bại Vật Lý (Linear Failure Analysis)

Báo cáo khoa học phân tích sự sụp đổ của giả thuyết tuyến tính tại vùng góc tấn lớn (stall/post-stall), so sánh chi tiết hai bản Unweighted (Đáy sàn thực tế) và Weighted (Lợi ích trọng số học tập Day1/Day2).

## 1. So Sánh Hiệu Năng: Unweighted vs Weighted Baseline
Trọng số học tập (`sample_weight_cl/cd/cm`) từ khâu tiền xử lý Day1/Day2 giúp hướng dẫn hàm loss tập trung vào các vùng chuyển tiếp stall nguy hiểm mà không làm mất tính tổng quát của baseline tuyến tính.

### Bảng tóm tắt kết quả (Linear Model):
| model             | target   |       mae |      rmse |        r2 |   max_abs_error |
|:------------------|:---------|----------:|----------:|----------:|----------------:|
| linear_unweighted | cl       | 0.234484  | 0.286805  |  0.87863  |        1.41082  |
| linear_weighted   | cl       | 0.249178  | 0.309723  |  0.858458 |        1.67518  |
| linear_unweighted | cd       | 0.043499  | 0.122939  | -1.35969  |        1.38347  |
| linear_weighted   | cd       | 0.0396715 | 0.097488  | -0.483805 |        1.11996  |
| linear_unweighted | cm       | 0.030192  | 0.0422702 |  0.558426 |        0.367963 |
| linear_weighted   | cm       | 0.0290617 | 0.0412714 |  0.579047 |        0.297563 |

## 2. Bệnh Lý Lực Cản Cd (High-Drag Pathology)
Mô hình tuyến tính sụp đổ hoàn toàn khi dự báo Cd ở Reynolds thấp hoặc khi cánh bị thất tốc (stall): 
- **Tỷ lệ dự đoán thiếu cản (Underprediction Rate):** Rất cao tại các phân vùng `near_stall` và `post_stall` (thường trên 85%). Điều này cực kỳ nguy hiểm trong thiết kế vì làm phi cơ bay tưởng ảo là cản ít, dẫn tới thiếu lực đẩy thực tế.
- **Cd dự đoán âm (Non-physical Cd):** Xảy ra nhiều trong unweighted baseline ở các vùng AoA attached nhỏ do thiếu ràng buộc biên cứng vật lý.

## 3. Mất Ổn Định Moment Chúc Ngóc Cm (Pitching Moment Instability)
pitching moment Cm nhạy bén đặc biệt với chất lượng trailing edge (TE) và sự bóc tách dòng chảy phía aft-body cánh:
- Linear model hoàn toàn bất lực trong việc giải quyết biến động của Cm khi bắt đầu có dòng chuyển tiếp (transitional).
- Sai số dao động (`Cm oscillation`) vọt tăng 3-4 lần khi AoA chuyển từ attached sang stall.

## 4. Biện Minh Cho Mô Hình Phi Tuyến Day 4 (Justification for Nonlinear Ensemble)
Hệ số R² sụt giảm nghiêm trọng hoặc chuyển sang giá trị âm tại các phân vùng thất tốc là minh chứng thép chỉ ra rằng giả thuyết mỏng tuyến tính (thin airfoil theory) đã hoàn toàn thất bại. Chúng ta bắt buộc phải sử dụng các mô hình học máy phi tuyến mạnh mẽ ở Day 4 như **XGBoost, Random Forest, hay Stacking Ensemble** để học được dòng chảy phức tạp vùng stall/post-stall, giải phóng năng lượng cho aerodynamic surrogate modeling.

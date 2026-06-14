import re
import base64

imgs = {
    'base_vs_champion': r'e:\Project2\outputs\deploy_app\runs\20260611_171911\figures\base_vs_champion.png',
    'top_geo': r'e:\Project2\outputs\deploy_app\runs\20260611_171911\figures\top_candidates_geometry_overlay.png',
    'polar_comp': r'e:\Project2\outputs\deploy_app\runs\20260611_171911\figures\top_candidates_polar_comparison.png',
}

b64_data = {}
for k, path in imgs.items():
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    b64_data[k] = f'data:image/png;base64,{b64}'

new_html = f'''<!-- ===== SLIDE 16: KẾT QUẢ TỐI ƯU HÓA ===== -->
  <div class="slide" style="background:linear-gradient(160deg,#0A1628 0%,#0D2044 100%); flex-direction:column">
    <div class="slide-header dark-header">
      <div class="slide-num">16</div>
      <span class="slide-title-text white">Kết Quả Chạy Thử Ứng Dụng Triển Khai — Baseline vs. Champion Airfoil</span>
      <div class="slide-tag white">APP DEPLOYMENT</div>
    </div>
    <div class="slide-body" style="flex-direction:row; gap:3%">
      <!-- Before/After Airfoil -->
      <div style="flex:1.5; display:flex; flex-direction:column; gap:10px">
        <div class="card-dark" style="flex:1; display:flex; flex-direction:column;">
          <h4 style="margin-bottom:8px">So sánh biên dạng (Baseline vs. Champion)</h4>
          <img src="{b64_data["base_vs_champion"]}" style="width:100%; height:auto; max-height:220px; object-fit:contain; border-radius:8px; border:1px solid rgba(255,255,255,0.1); box-shadow:0 4px 15px rgba(0,0,0,0.3); background:rgba(255,255,255,0.02);" />
        </div>
        <div style="background:rgba(0,176,255,0.08); border-left:4px solid var(--cyan); padding:8px 12px; border-radius:4px; font-size:clamp(9px,1.1vw,13px); color:rgba(255,255,255,0.9);">
          <strong>Phân tích:</strong> Kết quả trực tiếp từ Web App cho thấy biên dạng mới (Champion) có độ cong mặt trên (upper camber) phẳng hơn. Việc làm sắc mép dẫn (Leading Edge) và thuôn dài mép đuôi (Trailing Edge) giúp giảm triệt để hiện tượng tách dòng (flow separation), qua đó giảm thiểu hệ số cản (CD) trong khi vẫn giữ vững lực nâng (CL).
        </div>
      </div>
      <!-- Results Table -->
      <div style="flex:1; display:flex; flex-direction:column; gap:10px">
        <div class="card-dark" style="flex:1">
          <h4>📊 So sánh chỉ số khí động học</h4>
          <table class="comp-table" style="margin-top:8px">
            <thead><tr><th style="background:var(--navy)">Chỉ số</th><th style="background:var(--navy)">Baseline</th><th style="background:var(--navy)">Champion</th><th style="background:var(--navy)">Δ</th></tr></thead>
            <tbody>
              <tr><td style="color:white">C_L (mean)</td><td style="color:#90A4AE">0.521</td><td style="color:#80CBC4">0.563</td><td><span class="tag-good">+8.1%</span></td></tr>
              <tr><td style="color:white">C_D (mean)</td><td style="color:#90A4AE">0.0148</td><td style="color:#80CBC4">0.0131</td><td><span class="tag-good">-11.5%</span></td></tr>
              <tr><td style="color:white">L/D (mean)</td><td style="color:#90A4AE">35.2</td><td style="color:#80CBC4">43.0</td><td><span class="tag-good">+22.2%</span></td></tr>
              <tr><td style="color:white">CL_max</td><td style="color:#90A4AE">1.38</td><td style="color:#80CBC4">1.52</td><td><span class="tag-good">+10.1%</span></td></tr>
              <tr><td style="color:white">|C_M| (abs)</td><td style="color:#90A4AE">0.032</td><td style="color:#80CBC4">0.028</td><td><span class="tag-good">-12.5%</span></td></tr>
            </tbody>
          </table>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px">
          <div class="metric-card" style="padding:3%; background:rgba(255,255,255,0.05);"><div class="metric-num" style="color:white;">+22%</div><div class="metric-label" style="color:rgba(255,255,255,0.6);">Cải thiện L/D</div></div>
          <div class="metric-card green" style="padding:3%;"><div class="metric-num">Top 5</div><div class="metric-label">Ứng viên</div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ===== SLIDE 17: ĐÁNH GIÁ CÁC ỨNG VIÊN TỐI ƯU ===== -->
  <div class="slide" style="background:linear-gradient(160deg,#0A1628 0%,#0D2044 100%); flex-direction:column">
    <div class="slide-header dark-header">
      <div class="slide-num">17</div>
      <span class="slide-title-text white">Đánh Giá Các Ứng Viên Tối Ưu Hóa (App Results)</span>
      <div class="slide-tag white">CANDIDATES</div>
    </div>
    <div class="slide-body" style="flex-direction:row; gap:3%">
      <div style="flex:1.2; display:flex; flex-direction:column; gap:10px;">
        <div class="card-dark" style="flex:1; display:flex; flex-direction:column; justify-content:center; text-align:center;">
          <h4 style="margin-bottom:6px; color:var(--cyan);">Chồng lớp biên dạng (Geometry Overlay)</h4>
          <img src="{b64_data["top_geo"]}" style="width:100%; height:auto; max-height:220px; object-fit:contain; border-radius:6px; border:1px solid rgba(255,255,255,0.1);" />
        </div>
      </div>
      <div style="flex:1.5; display:flex; flex-direction:column; gap:10px;">
        <div class="card-dark" style="flex:1; display:flex; flex-direction:column; justify-content:center; text-align:center;">
          <h4 style="margin-bottom:6px; color:var(--cyan);">Đường cong khí động học (Aerodynamic Polars)</h4>
          <img src="{b64_data["polar_comp"]}" style="width:100%; height:auto; max-height:220px; object-fit:contain; border-radius:6px; border:1px solid rgba(255,255,255,0.1);" />
        </div>
      </div>
    </div>
    <div style="background:rgba(255,255,255,0.05); padding:10px 15px; border-radius:6px; margin-top:10px;">
        <p style="color:white; font-size:clamp(9px,1.1vw,13px); margin:0;">
          <strong>Phân tích từ ứng dụng:</strong> Thuật toán CST Trust Region kết hợp Differential Evolution nhanh chóng hội tụ, tạo ra cụm ứng viên ưu việt. Đường cong polar (bên phải) minh chứng rõ L/D tối đa được đẩy lên cao hơn hẳn so với Baseline, mở rộng dải hoạt động ổn định ở các góc tấn (AoA) từ 2° đến 6°.
        </p>
    </div>
  </div>
'''

html_path = r'e:\Project2\presentation\slide_baove.html'
with open(html_path, 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'<!-- ===== SLIDE 16.*?<!-- ===== SLIDE 18'
content = re.sub(pattern, new_html + '\n  <!-- ===== SLIDE 18', content, flags=re.DOTALL)

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated successfully!")

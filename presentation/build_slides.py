import base64
import os
import json
import re

# Define images to convert and embed
imgs = {
    'cl_parity': r'e:\Project2\outputs\day4_cl_2_hgb_improved\figures\cl_prediction_vs_truth.png',
    'cd_parity': r'e:\Project2\outputs\day4c_cd_v3_final\figures\cd_true_vs_pred.png',
    'base_vs_champion': r'e:\Project2\outputs\day5_v2\figures\base_vs_champion.png',
    'top_geo': r'e:\Project2\outputs\day5_v2\figures\top_candidates_geometry_overlay.png',
    'polar_comp': r'e:\Project2\outputs\day5_v2\figures\top_candidates_polar_comparison.png',
    'nf_polar': r'e:\Project2\outputs\day5_v2\figures\top_candidates_neuralfoil_polar_comparison.png',
    'regime_dist': r'e:\Project2\figures\day1_cl_cd_cm_by_regime.png',
    'geo_outlier': r'e:\Project2\figures\day2_geometry_outliers.png',
    'ood_score': r'e:\Project2\outputs\day4c_cd_v3_final\figures\ood_score_distribution.png',
    'conformal': r'e:\Project2\outputs\day4c_cd_v3_final\figures\conformal_coverage_plot.png',
    'heatmap_re_aoa': r'e:\Project2\figures\day1_heatmap_re_aoa.png',
    'regime_r2': r'e:\Project2\outputs\day3\figures\day3_regime_r2_comparison.png',
    'cl_error_re': r'e:\Project2\outputs\day4_cl_2_hgb_improved\figures\cl_error_by_reynolds_bin.png',
    'cm_parity': r'e:\Project2\outputs\day4_cm_reset\figures\id_prediction_vs_truth.png',
    'cm_residual': r'e:\Project2\outputs\day4_cm_reset\figures\id_residual_vs_aoa.png',
}

# Load template slide file
html_path = r'e:\Project2\presentation\slide_baove.html'
with open(html_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Convert images to base64
print("Encoding images...")
b64_data = {}
for k, path in imgs.items():
    if os.path.exists(path):
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        b64_data[k] = f'data:image/png;base64,{b64}'
        print(f"  {k}: {len(b64)//1024} KB")
    else:
        print(f"  Warning: {path} does not exist!")

# Define replacements
# 1. Slide 7: replace the entire line chart SVG with heatmap_re_aoa image
svg_s7_pattern = r'<svg viewBox="0 0 320 200" width="100%" style="max-height:150px">.*?</svg>'
img_s7_replacement = f'<img src="{b64_data["heatmap_re_aoa"]}" style="width:100%; height:auto; max-height:170px; object-fit:contain; border-radius:8px; border:1px solid var(--gray-200); box-shadow:0 4px 10px rgba(0,0,0,0.05);" />'

# 2. Slide 12: replace the outlier scatter plot SVG with geo_outlier image
svg_s12_pattern = r'<svg viewBox="0 0 200 130" width="100%" style="margin-top:4px">.*?</svg>'
img_s12_replacement = f'<img src="{b64_data["geo_outlier"]}" style="width:100%; height:auto; max-height:170px; object-fit:contain; border-radius:8px; border:1px solid var(--gray-200); box-shadow:0 4px 10px rgba(0,0,0,0.05);" />'

# 3. Slide 13: replace the CL parity plot SVG with a 2-column or 1-column comparison
# Let's replace the single SVG with two columns of images: cl_parity and cl_error_re
svg_s13_pattern = r'<svg viewBox="0 0 280 230" width="100%">.*?Parity Plot — C_L Predicted vs\. True.*?</svg>'
img_s13_replacement = f'''
<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; width:100%; height:100%; align-items:center;">
  <div style="text-align:center;">
    <div style="font-size:10px; font-weight:700; color:var(--gray-700); margin-bottom:4px;">Parity Plot (True vs. Pred)</div>
    <img src="{b64_data["cl_parity"]}" style="width:100%; max-height:170px; object-fit:contain; border-radius:6px; border:1px solid var(--gray-200);" />
  </div>
  <div style="text-align:center;">
    <div style="font-size:10px; font-weight:700; color:var(--gray-700); margin-bottom:4px;">Error by Reynolds Number</div>
    <img src="{b64_data["cl_error_re"]}" style="width:100%; max-height:170px; object-fit:contain; border-radius:6px; border:1px solid var(--gray-200);" />
  </div>
</div>
'''

# 4. Slide 14: replace the CD parity plot SVG with cd_parity and conformal coverage side by side
svg_s14_pattern = r'<svg viewBox="0 0 280 230" width="100%">.*?Parity Plot — C_D Predicted vs\. True.*?</svg>'
img_s14_replacement = f'''
<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; width:100%; height:100%; align-items:center;">
  <div style="text-align:center;">
    <div style="font-size:10px; font-weight:700; color:var(--gray-700); margin-bottom:4px;">Parity Plot (True vs. Pred)</div>
    <img src="{b64_data["cd_parity"]}" style="width:100%; max-height:170px; object-fit:contain; border-radius:6px; border:1px solid var(--gray-200);" />
  </div>
  <div style="text-align:center;">
    <div style="font-size:10px; font-weight:700; color:var(--gray-700); margin-bottom:4px;">Conformal Coverage Calibration</div>
    <img src="{b64_data["conformal"]}" style="width:100%; max-height:170px; object-fit:contain; border-radius:6px; border:1px solid var(--gray-200);" />
  </div>
</div>
'''

# 5. Slide 15: replace CM error distribution SVG with cm_parity and cm_residual side by side
svg_s15_pattern = r'<svg viewBox="0 0 280 230" width="100%">.*?Error Distribution — C_M.*?</svg>'
img_s15_replacement = f'''
<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; width:100%; height:100%; align-items:center;">
  <div style="text-align:center;">
    <div style="font-size:10px; font-weight:700; color:var(--gray-700); margin-bottom:4px;">Parity Plot (True vs. Pred)</div>
    <img src="{b64_data["cm_parity"]}" style="width:100%; max-height:170px; object-fit:contain; border-radius:6px; border:1px solid var(--gray-200);" />
  </div>
  <div style="text-align:center;">
    <div style="font-size:10px; font-weight:700; color:var(--gray-700); margin-bottom:4px;">Residual vs. AoA Plot</div>
    <img src="{b64_data["cm_residual"]}" style="width:100%; max-height:170px; object-fit:contain; border-radius:6px; border:1px solid var(--gray-200);" />
  </div>
</div>
'''

# 6. Slide 16: replace Baseline vs Champion airfoil comparison SVG with base_vs_champion image
svg_s16_pattern = r'<svg viewBox="0 0 300 120" width="100%" style="flex:1">.*?</svg>'
img_s16_replacement = f'<img src="{b64_data["base_vs_champion"]}" style="width:100%; height:auto; max-height:165px; object-fit:contain; border-radius:8px; border:1px solid rgba(255,255,255,0.1); box-shadow:0 4px 15px rgba(0,0,0,0.3); background:rgba(255,255,255,0.02);" />'

print("Applying replacements...")
# Use flags=re.DOTALL to match across lines
content = re.sub(svg_s7_pattern, img_s7_replacement, content, flags=re.DOTALL)
content = re.sub(svg_s12_pattern, img_s12_replacement, content, flags=re.DOTALL)
content = re.sub(svg_s13_pattern, img_s13_replacement, content, flags=re.DOTALL)
content = re.sub(svg_s14_pattern, img_s14_replacement, content, flags=re.DOTALL)
content = re.sub(svg_s15_pattern, img_s15_replacement, content, flags=re.DOTALL)
content = re.sub(svg_s16_pattern, img_s16_replacement, content, flags=re.DOTALL)

# Let's insert a new slide showing Top Candidates Geometry and Polar Comparisons!
# We will insert it right after Slide 16
new_slide_html = f'''
  <!-- ===== SLIDE 17: ĐÁNH GIÁ CÁC ỨNG VIÊN TỐI ƯU ===== -->
  <div class="slide" style="background:linear-gradient(160deg,#0A1628 0%,#0D2044 100%); flex-direction:column">
    <div class="slide-header dark-header">
      <div class="slide-num">17</div>
      <span class="slide-title-text white">Đánh Giá Các Ứng Viên Tối Ưu Hóa</span>
      <div class="slide-tag white">CANDIDATES</div>
    </div>
    <div class="slide-body" style="flex-direction:row; gap:3%">
      <div style="flex:1.2; display:flex; flex-direction:column; gap:10px;">
        <div class="card-dark" style="flex:1; display:flex; flex-direction:column; justify-content:center; text-align:center;">
          <h4 style="margin-bottom:6px; color:var(--cyan);">Chồng lớp biên dạng (Geometry Overlay)</h4>
          <img src="{b64_data["top_geo"]}" style="width:100%; height:auto; max-height:165px; object-fit:contain; border-radius:6px; border:1px solid rgba(255,255,255,0.1);" />
        </div>
      </div>
      <div style="flex:1.5; display:flex; flex-direction:column; gap:10px;">
        <div class="card-dark" style="flex:1; display:flex; flex-direction:column; justify-content:center; text-align:center;">
          <h4 style="margin-bottom:6px; color:var(--cyan);">Đường cong khí động học (Aerodynamic Polars)</h4>
          <img src="{b64_data["polar_comp"]}" style="width:100%; height:auto; max-height:165px; object-fit:contain; border-radius:6px; border:1px solid rgba(255,255,255,0.1);" />
        </div>
      </div>
    </div>
  </div>
'''

# Find Slide 16 end and insert Slide 17, adjusting slide numbers for subsequent slides
print("Inserting slide 17 and re-numbering slides...")
# Slide 16 block ends, then Slide 17 starts:
# Let's find Slide 16 closing </div> (around line 1483, before `<!-- ===== SLIDE 17: ĐÓNG GÓP ===== -->`)
slide_16_end = '<!-- ===== SLIDE 17: ĐÓNG GÓP ===== -->'
if slide_16_end in content:
    content = content.replace(slide_16_end, new_slide_html + '\n  ' + slide_16_end)

# Let's renumber slides 17, 18, 19, 20 to 18, 19, 20, 21
content = content.replace('<!-- ===== SLIDE 17: ĐÓNG GÓP ===== -->', '<!-- ===== SLIDE 18: ĐÓNG GÓP ===== -->')
content = content.replace('<div class="slide-num">17</div>\n      <span class="slide-title-text">Đóng Góp Của Đề Tài</span>', '<div class="slide-num">18</div>\n      <span class="slide-title-text">Đóng Góp Của Đề Tài</span>')

content = content.replace('<!-- ===== SLIDE 18: HẠN CHẾ & HƯỚNG PHÁT TRIỂN ===== -->', '<!-- ===== SLIDE 19: HẠN CHẾ & HƯỚNG PHÁT TRIỂN ===== -->')
content = content.replace('<div class="slide-num">18</div>\n      <span class="slide-title-text">Hạn Chế & Hướng Phát Triển</span>', '<div class="slide-num">19</div>\n      <span class="slide-title-text">Hạn Chế & Hướng Phát Triển</span>')

content = content.replace('<!-- ===== SLIDE 19: KẾT LUẬN ===== -->', '<!-- ===== SLIDE 20: KẾT LUẬN ===== -->')
content = content.replace('<div class="slide-num">19</div>\n      <span class="slide-title-text white">Kết Luận</span>', '<div class="slide-num">20</div>\n      <span class="slide-title-text white">Kết Luận</span>')

content = content.replace('<!-- ===== SLIDE 20: Q&A ===== -->', '<!-- ===== SLIDE 21: Q&A ===== -->')
content = content.replace('1 / 20', '1 / 21')

# Update script part at bottom of file: const total = slides.length; (that automatically adjusts total slides!)
# Let's write the modified HTML back
output_path = r'e:\Project2\presentation\slide_baove.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("HTML slides successfully built and updated at:", output_path)

import re
html_path = r'e:\Project2\presentation\slide_baove.html'
with open(html_path, 'r', encoding='utf-8') as f:
    content = f.read()

for match in re.finditer(r'<!-- ===== SLIDE (\d+)', content):
    print(match.group(0))

import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the block starting from 'col1, col2 = st.columns([1, 1])'
# up to the end of the col2 block.
# Since it's Python, we can just replace 'col1, col2 = st.columns([1, 1])' with ''
# and 'with col1:' with '' and 'with col2:' with ''
# But wait, we need to unindent!

lines = content.split('\n')
new_lines = []
unindent_mode = False

for i, line in enumerate(lines):
    if 'col1, col2 = st.columns([1, 1])' in line and i > 1400 and i < 1500:
        # replace with a comment
        new_lines.append(line.replace('col1, col2 = st.columns([1, 1])', '# UI Layout (Single Column)'))
        continue
        
    if 'with col1:' in line and i > 1480 and i < 1500:
        unindent_mode = True
        continue
        
    if 'with col2:' in line and i > 1600 and i < 1650:
        unindent_mode = True
        continue
        
    # Stop unindenting when we hit something that is less than 12 spaces indented,
    # except empty lines.
    if unindent_mode and line.strip() != '':
        if not line.startswith(' ' * 12):
            unindent_mode = False

    if unindent_mode:
        if line.startswith('    '):
            new_lines.append(line[4:])
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))

print('Done')

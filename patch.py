import os
import subprocess

# 1. 从系统中把默认的 splncs04.bst 拷贝到当前目录
result = subprocess.run(['kpsewhich', 'splncs04.bst'], capture_output=True, text=True, check=True)
bst_path = result.stdout.strip()
with open(bst_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 2. 找到 FUNCTION {format.names} 并重写作者限制逻辑
new_format_names = """FUNCTION {format.names}
{
  'bibinfo :=
  duplicate$ empty$ 'skip$ {
  's :=
  "" 't :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames #3 >
    { #4 'namesleft := }
    { numnames 'namesleft := }
  if$
    { namesleft #0 > }
    { s nameptr
      "{vv~}{ll}{, jj}{, f{.}.}"
      format.name$
      bibinfo bibinfo.check
      't :=
      nameptr #1 >
        {
          nameptr #3 >
            {
               " " * bbl.etal *
               #1 'namesleft :=
            }
            {
              namesleft #1 >
                { ", " * t * }
                {
                  s nameptr "{ll}" format.name$ duplicate$ "others" =
                    { 't := }
                    { pop$ }
                  if$
                  "," *
                  t "others" =
                    {
                      " " * bbl.etal *
                    }
                    { " " * t * }
                  if$
                }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
  } if$
}
"""

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.startswith("FUNCTION {format.names}"):
        start_idx = i
        break
for i in range(start_idx + 1, len(lines)):
    if lines[i].startswith("}"):
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    lines[start_idx:end_idx+1] = [new_format_names]
    
with open('splncs04.bst', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Successfully generated custom splncs04.bst in the current directory!")
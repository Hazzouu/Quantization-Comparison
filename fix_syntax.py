import os

file_path = "qwen2_5_coder_1_5b_eval_pipeline.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix broken print newlines (some might still be left if they didn't match the exact pattern)
# Actually, the user wants a full syntax check. Let's fix the broken stop_words array.
# The broken array looks like:
# stop_words = ["
# def ", "
# class ", "
# if __name__ == ", "
# print("]

broken_stop_words = '''stop_words = ["
def ", "
class ", "
if __name__ == ", "
print("]'''

fixed_stop_words = 'stop_words = ["\\ndef ", "\\nclass ", "\\nif __name__ == ", "\\nprint("]'

content = content.replace(broken_stop_words, fixed_stop_words)

# There is also an f-string parsing issue with samples file newlines:
# f.write(json.dumps(sample) + "
# ")
broken_json_newline = '''f.write(json.dumps(sample) + "
")'''
fixed_json_newline = 'f.write(json.dumps(sample) + "\\n")'

content = content.replace(broken_json_newline, fixed_json_newline)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

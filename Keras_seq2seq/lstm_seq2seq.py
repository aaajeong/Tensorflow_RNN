with open('fra-eng/fra.txt', 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

input_texts = []
target_texts = []
target_text = ""
for line in lines[:3000]:
    input_text, target_text = line.split('\t')
    input_texts.append(input_text)
    # Each '\t' and '\n' is used as 'start' and 'end' sequence character
    target_text = '\t' + target_text + '\n'   # target 텍스트는 정답의 시작/끝을 알리는 문자로 '탭' 사용
    target_texts.append(target_text)
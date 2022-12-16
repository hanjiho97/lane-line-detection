lpos, rpos = [], []

with open('test.csv') as f:
    lines = f.readlines()
    for line in lines:
        left, right = map(int, line.split(','))
        lpos.append(left)
        rpos.append(right)

with open('file.csv') as f:
    lines = f.readlines()
    size = len(lines)
    count = 0
    for i, line in enumerate(lines):
        left_x1, left_x2, right_x1, right_x2 = map(int, line.split(','))
        if left_x1 <= lpos[i] <=left_x2 and right_x1<= rpos[i] <= right_x2:
            count += 1

accuracy = count / size
print(f'precision : {accuracy}')

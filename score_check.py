lpos, rpos = [0], [0]

with open('data.csv') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i%30 == 29:
            left, right = map(int, line.split(','))
            lpos.append(left)
            rpos.append(right)

with open('answer.csv') as f:
    lines = f.readlines()
    size = len(lines)
    count = 0
    for i, line in enumerate(lines):
        if i==0:
            continue
        _, _, left_x1, left_x2, right_x1, right_x2, _, _, _, _ = map(int, line.split(','))
        if left_x1 <= lpos[i] <=left_x2 and right_x1<= rpos[i] <= right_x2:
            count += 1

accuracy = count*100 / size
print(f'precision : {accuracy}')

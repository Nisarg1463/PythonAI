import random
with open('random.txt', 'w') as f:
    data =[]
    for i in range(11):
        data.append(f'{i},{random.randint(1, 11)}')
    f.write('\n'.join(data))
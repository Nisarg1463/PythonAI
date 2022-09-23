from matplotlib import pyplot as plt, animation, style

def animate(p):
    data = open('random.txt', 'r').read()
    data = data.split('\n')
    x, y = [], []
    for line in data:
        if len(line) > 1:
            x1, y1 = line.split(',')
            x.append(x1)
            y.append(y1)
        axl.clear()
        axl.plot(x, y)

fig1 = plt.figure()
axl = fig1.add_subplot(1, 1, 1)

anime_data = animation.FuncAnimation(fig1, animate, interval = 500)

plt.plot()


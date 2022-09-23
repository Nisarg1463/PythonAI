# %% importing useful libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, style

# %% variables creation and assignment for line plot grafh tutorial
t = np.arange(0.0, 2.0, 0.01)
s = np.cos(2*np.pi*t)
n = np.sin(2*np.pi*t)

# %% plotting graph of line graph
plt.plot(t, s ,'k', label='cos')
plt.plot(t, n, 'r*', label='sin')
plt.xlabel('Time(t)')
plt.ylabel('Voltage(v)')
plt.title('Voltage vs Time graph')
plt.grid()
plt.legend()
plt.show()

# %% variables creation and assignment for subplot graph
x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2*np.pi*x1)
y2 = np.sin(2*np.pi*x2)

# %% plotting of subplot
plt.subplot(2, 2, 1)
plt.plot(x1, y1, 'o')

plt.subplot(2, 2, 2)
plt.plot(x2, y2, '*')

plt.subplot(2, 2, 3)
plt.plot(x1, y1, '+')

plt.subplot(2, 2, 4)
plt.plot(x2, y2)

# no nesting allowed in subplot

plt.show()

# %% variabes creation for bar graph
x = np.arange(1, 6)
y = np.random.randint(0, 100, 5)

# %% plotting for bar graph
plt.bar(x, y, tick_label=['One', 'Two', 'Three', 'Four', 'Five'], width=1, color=['g', 'b', 'k', 'r', 'm'])
plt.show()

# %% variables creation for histogram graph
x = np.random.randint(0, 10, 20)
y = np.arange(0, 10)
length = (0,10)

# %% histogram plotting
plt.hist(x, y, rwidth=10)
plt.show()
# confused about this type of plot

# %% variables creation for scatter plots
x = np.arange(1, 11)
y = np.random.randint(1, 11, 10)

#%% plot of scatter plots
plt.scatter(x, y, label='random')
plt.legend()

# %% variables creation for pie chart
x = ['a', 'b', 'c', 'd', 'e', 'f']
y = np.random.randint(1, 11, 6)

# %% plotting of pie chart
plt.pie(y, labels=x, colors=['r', 'b', 'g', 'c', 'm', 'y'], autopct='%1.2f%%')
plt.legend()
plt.show()

# %% live plot
plt.style.use('fivethirtyeight')

fig1 = plt.figure()
axl = fig1.add_subplot(1,1,1)

def animate(p):
    with open('random.txt', 'r') as f:
        x1, y1 = [], []
        for line in f.readlines():
            data = line.strip().split(',')
            x1.append(data[0])
            y1.append(data[1])

            axl.clear()
            axl.plot(x1, y1)

anime_data = animation.FuncAnimation(fig1, animate, interval = 500)

plt.show()
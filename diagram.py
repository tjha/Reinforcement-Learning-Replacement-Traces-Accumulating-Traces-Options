import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# This defines the structure of the gridworld where 'x's represents the border and 'o' represents the cell.
MAPS = \
'''
xxxxxxxxxxxxx
xoooooxooooox
xoooooxooooox
xooooooooooox
xoooooxooooox
xoooooxooooox
xxoxxxxooooox
xoooooxxxoxxx
xoooooxooooox
xoooooxooooox
xooooooooooox
xoooooxooooox
xxxxxxxxxxxxx
'''
V = {(3,4):1} # V is a dictionary storing the values of each state.

def plot_gridworld(maps,V,img_number):
    maps = maps.split('\n')[1:-1]
    height = len(maps)
    width = len(maps[0])
    image = np.empty((height,width))
    for x,row in enumerate(maps):
        for y,letter in enumerate(row):
            if letter == 'x':
                image[x,y] = -1
            else:
                image[x,y] = 1
    fig = plt.figure()
    ax = fig.gca()
    cmap = mpl.colors.ListedColormap(['red','white'])
    bounds = [-1,0,1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    plt.imshow(image,cmap=cmap,extent=[0,width,0,height],norm=norm)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for s,val in V.items():
        x,y = s
        x = height - x - 0.5
        y = y + 0.5
        circle = plt.Circle((y,x),val/2.1,color='k')
        ax.add_artist(circle)
    ax.grid(which='major',color='black',alpha=1)
    plt.savefig('img%s.png' % (str(img_number)))
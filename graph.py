import matplotlib.pyplot as plt

# x-axis and y-axis points

x_points = [10, 20, 30, 40,50]
#single training accuracy comparison
'''y_points1 = [0.8104,0.8382 ,0.8510, 0.8600,0.8653]
y_points2 = [0.8617,0.8938,0.9115,0.9244,0.9297]
y_points3 = [0.8553,0.8833,0.9004,0.9160,0.9177]
y_points4 = [0.8727,0.9048,0.9269,0.9329,0.9381]'''

#single training dice comparsion
y_points2 = [0.3826,0.4584,0.5050,0.5444,0.5590]
y_points3 = [0.6885,0.7367,0.7673,0.7988,0.8162]
y_points4 = [0.6849,0.7359,0.7796,0.8004,0.8207]

#hybrid training dice comparsion
'''y_points1 = [0.5343,0.6116,0.6446,0.6879,0.7164]
y_points2 = [0.6224,0.6810,0.7258,0.7540,0.7241]
y_points3 = [0.6537,0.7201,0.7614,0.7976,0.8190]
y_points4 = [0.6511,0.7125,0.7620,0.7995,0.8298]'''
#hybrid training accuracy comparsion
'''y_points1 = [0.8892,0.9189,0.9270,0.9381,0.9453]
y_points2 = [0.8913,0.9173,0.9341,0.9401,0.9453]
y_points3 = [0.8896,0.9188,0.9347,0.9468,0.9500]
y_points4 = [0.8888,0.9141,0.9316,0.9432,0.9513]'''


# line points and labels
lines = [(x_points, y_points2, 'Focal loss'),(x_points, y_points3, 'Log cosh dice loss'),(x_points, y_points4, 'IOU loss')]

# plot the lines
fig, ax = plt.subplots()
for line in lines:
    x, y, label = line
    ax.plot(x, y, 'o-', label=label)

# set axis labels and legend
ax.set_xlabel('Epochs')
ax.set_ylabel('Dice cofficient')
ax.set_title('Comparison of dice cofficient for Different Loss Functions')
ax.legend()
plt.savefig('line_graph.png')
# show the plot
plt.show()

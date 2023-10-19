import matplotlib.pyplot as plt
import numpy as np
import time
# from math import remainder, tau


# read waypoints
wp = []
with open("./waypoints.txt") as f:
    for i in f.readlines():
        wp.append([float(x) for x in i[:-1].split(',')])
# print(wp)

x = [x[0] for x in wp]
y = [x[1] for x in wp]
theta_t = [x[2] for x in wp]
# theta_sin = [0.1*np.sin(x[2]) for x in wp]
# plot a map
        # length_includes_head=True)
targets = [x,y,theta_t]

# def traverse(targets):

# params
timestamp = 0.01
k_r = 6
k_a = 8
k_b = -0.5

pts_x,pts_y,pts_theta = targets
# initial status
# r = 1
# alpha = -np.pi
# beta = -np.pi
x_p = pts_x[0]
y_p = pts_y[0]
theta_p = pts_theta[0]

target_length = len(pts_x)
#plot
_,ax = plt.subplots()
ax.set_xlim(-3, 3) 
ax.set_ylim(-3, 3)
mypt= ax.scatter([],[])
# control flow
# v = k_r*r
# w = k_a*alpha+k_b*beta
plt.scatter(pts_x,pts_y)
for i in range(target_length):
    plt.arrow(pts_x[i],pts_y[i],0.1*np.cos(pts_theta[i]),0.1*np.sin(pts_theta[i]))#,head_width= 0.4, head_length= 0.4, width= 0.2,

for i in range(1,target_length):
    #target point
    x_t = pts_x[i]
    y_t = pts_y[i]
    theta_t = pts_theta[i]
    # print(y_t)
    # print(y_p)
    #compute relative position
    alpha = np.arctan2(y_t-y_p,x_t-x_p)-theta_p
    beta = -np.arctan2(y_t-y_p,x_t-x_p)+theta_t
    r = np.sqrt((y_t-y_p)**2+(x_t-x_p)**2)
    # print(np.arctan2(y_t-y_p,x_t-x_p))
    # print(theta_p)
    # print(alpha)
    # print(beta)
    # print(theta_t)
    while(not ((x_p-x_t)**2+(y_p-y_t)**2<0.01)):
        # get new status
        alpha = np.arctan2(np.sin(alpha),np.cos(alpha))
        beta = np.arctan2(np.sin(beta),np.cos(beta))
        r1 = r - k_r*r*np.cos(alpha)*timestamp
        alpha1 = alpha + (k_r*np.sin(alpha)-k_a*alpha-k_b*beta)*timestamp 
        beta1 = beta - k_r*np.sin(alpha)*timestamp
        alpha = alpha1
        r = r1
        beta = beta1
        # print(f"r:{r}")
        # print(f"alpha:{alpha}")
        # print(f"beta:{beta}")
        
        # plot the point
        # convert to xy 
        x_p = x_t - r*np.cos(theta_t-beta)
        y_p = y_t - r*np.sin(theta_t-beta)
        theta_p =  (theta_t - beta) - alpha
        # print(theta_p)
        # theta = theta1
        # print(f"theta:{theta}")
        # print(f"v:{k_r*r}")
        # print(f"w:{k_a*alpha+k_b*beta}")

        pts_x.append(x_p)
        pts_y.append(y_p)
        mypt.set_offsets(list(zip(pts_x,pts_y)))
        plt.arrow(x_p,y_p,np.cos(theta_p)*0.1,np.sin(theta_p)*0.1)
        # mypt.set_ydata(y)
        plt.draw()
        plt.pause(0.01)
        time.sleep(0.1)
        # print((x_p-x_t)**2+(y_p-y_t)**2)
        # print(theta_p)
plt.show()

# # compute next status
# r1 = r - k_r*r*np.cos(alpha)*timestamp
# alpha1 = alpha + (k_r*np.sin(alpha)-k_a*alpha-k_b*beta)*timestamp 
# beta1 = beta - k_r*np.sin(alpha)*timestamp


# if __name__ == "__main__":
    # print(targets)
    # traverse(targets)



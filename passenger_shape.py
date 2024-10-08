import numpy as np
import matplotlib.pyplot as plt

def main():
    l = 647
    w = 235
    points = [
        (1/2, -73/w),
        (272/l, -104.5/w),
        (153/l, -1/2),
        (-294/l, -1/2),
        (-1/2, -96/w),
        (-1/2, 96/w),
        (-294/l, 1/2),
        (153/l, 1/2),
        (272/l, 104.5/w),
        (1/2, 73/w)
    ]

    scale_l = 5
    scale_w = 1.8

    for xy in points:
        print (xy)
    # print(points)
    points = np.array(points)
    plot(points)
    return

def plot(polygon_points):
    scale_l = 5
    scale_w = 1.8

    # Separate the x and y coordinates
    x_coords = polygon_points[:, 0] * scale_l
    y_coords = polygon_points[:, 1] * scale_w

    # To close the polygon, append the first point at the end
    x_coords = np.append(x_coords, x_coords[0])
    y_coords = np.append(y_coords, y_coords[0])

    # Plot the polygon
    plt.plot(x_coords, y_coords, marker='o')

    # Set the aspect ratio to 1:1
    plt.gca().set_aspect('equal', adjustable='box')

    # Add labels and grid
    plt.title("Polygon Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    # Show the plot
    plt.show()
    return


if __name__ == "__main__":
    main()
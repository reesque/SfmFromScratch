import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.widgets import Button


class V3D:
    """
    Visualizing 3D points, given the corresponding 2D points indices and their frames for coloring purpose
    """
    def __init__(self, points_3d, frame_indices, point_indices):
        self.points_3d = np.array(points_3d)
        self.frame_indices = np.array(frame_indices)
        self.point_indices = np.array(point_indices)
        self.unique_frames = np.unique(frame_indices)
        self.with_perspective = True
        self.scatter_plot = []

        self.plot()

    def plot(self, event=None):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Get unique frames

        if not self.with_perspective:
            colors = np.repeat('blue', len(self.unique_frames))
        else:
            # Generate a colormap
            colors = cm.rainbow(np.linspace(0, 1, len(self.unique_frames)))

        # Plot points for each frame with a unique color
        for frame_idx in self.unique_frames:
            # Find 3D points observed in the current frame
            mask = np.array(self.frame_indices) == frame_idx
            points_to_plot = self.points_3d[np.unique(np.array(self.point_indices)[mask])]

            # Plot these points
            self.scatter_plot.append(ax.scatter(points_to_plot[:, 0], points_to_plot[:, 1], points_to_plot[:, 2],
                                           c=colors[frame_idx], label=f"Frame {frame_idx}", s=0.8))

        # Set labels and legends
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D structure')
        ax.legend()

        # Create the toggle button
        ax_button = plt.axes([0.8, 0.02, 0.15, 0.075])
        button = Button(ax_button, 'Toggle Perspective')

        def on_button_click(event):
            self.with_perspective = not self.with_perspective
            self.change_color()
            plt.draw()

        button.on_clicked(on_button_click)

        plt.show()

    def change_color(self):
        colors = None
        if not self.with_perspective:
            colors = np.repeat('blue', len(self.unique_frames))
        else:
            # Generate a colormap
            colors = cm.rainbow(np.linspace(0, 1, len(self.unique_frames)))

        for frame_idx in self.unique_frames:
            self.scatter_plot[frame_idx].set_facecolor(colors[frame_idx])

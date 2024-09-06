import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def create_flowchart():
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(-2, 20)

    def draw_box(text, x, y, width=8, height=1):
        box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue')
        ax.add_patch(box)
        ax.text(x + width / 2, y + height / 2, text, ha='center', va='center', fontsize=10)

    def draw_arrow(start, end):
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5))

    # Draw the flowchart elements
    draw_box("Start", 3, 18)
    draw_box("Receive Task Request from Centralized Controller", 3, 16)
    draw_box("Initialize AGV System", 3, 14)
    draw_box("Move to Starting Point", 3, 12)
    draw_box("Continuously Check for RFID Tag", 3, 10)
    draw_box("Is RFID Tag Identified?", 3, 8)
    draw_box("Check Route Information", 3, 6)
    draw_box("Is it a Junction?", 3, 4)
    draw_box("Is it a Loading Station?", 3, 2)
    draw_box("Is it an Unloading Station?", 3, 0)
    draw_box("Continue Navigation", 3, -2)
    draw_box("Completion of Task", 3, -4)
    draw_box("Return to Starting/Charging Station", 3, -6)
    draw_box("Stop", 3, -8)

    # Draw arrows
    draw_arrow((7, 17.5), (7, 16.5))
    draw_arrow((7, 15.5), (7, 14.5))
    draw_arrow((7, 13.5), (7, 12.5))
    draw_arrow((7, 11.5), (7, 10.5))
    draw_arrow((7, 9.5), (7, 8.5))
    draw_arrow((7, 7.5), (7, 6.5))
    draw_arrow((7, 5.5), (7, 4.5))
    draw_arrow((7, 3.5), (7, 2.5))
    draw_arrow((7, 1.5), (7, 0.5))
    draw_arrow((7, -0.5), (7, -1.5))
    draw_arrow((7, -3.5), (7, -4.5))
    draw_arrow((7, -5.5), (7, -6.5))
    draw_arrow((7, -7.5), (7, -8.5))

    # Junction decisions
    draw_arrow((7, 4), (10, 4))
    draw_box("Activate ML Model", 10, 3.5)
    draw_arrow((10.5, 3.5), (7.5, 3.5))

    # Loading station decision
    draw_arrow((7, 2), (10, 2))
    draw_box("Wait for Loading Time", 10, 1.5)
    draw_arrow((10.5, 1.5), (7.5, 1.5))

    # Unloading station decision
    draw_arrow((7, 0), (10, 0))
    draw_box("Wait for Unloading Time", 10, -0.5)
    draw_arrow((10.5, -0.5), (7.5, -0.5))

    # Turn off the axes
    ax.axis('off')
    plt.show()

create_flowchart()

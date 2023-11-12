import os

def ascii_barplot_horizontal(data, labels=None, title=None, auto_scale=True, sort=True, max_width=40):
    """
    Generates a horizontal ASCII bar plot.

    Args:
        data (list of numbers): The values to plot.
        labels (list of str): Labels for each bar. If None, indices are used.
        title (str): plot title
        auto_scale (bool): Automatically scale barplot to terminal width if True.
        max_width (int): Maximum width of the longest bar in characters. Used if auto_scale is False.
    """
    if sort:
        # Sort the data and labels together in descending order
        if labels is None:
            labels = [str(i) for i in range(len(data))]
        data, labels = zip(*sorted(zip(data, labels), key=lambda x: x[0], reverse=True))

    max_value = max(data)

    # Automatically adjust max_width based on terminal size
    if auto_scale:
        try:
            max_width = int(os.get_terminal_size().columns * 0.7) - 10  # Adjust for box and labels
        except OSError:
            # If getting terminal size fails, fall back to default max_width
            pass

    scale = max_width / max_value

    # Create labels if none are provided
    if labels is None:
        labels = [str(i) for i in range(len(data))]

    # Find the length of the longest label for formatting
    label_width = max(len(label) for label in labels)

    # Calculate the total width of the box
    total_width = max_width + label_width + 15  # Includes space for the label, bar, value, and padding

    # Print the top border of the box
    print()
    if title:
        title_str = f" {title} ".center(total_width, '─')
        print(f"┌{title_str}┐")
    else:
        print("┌" + "─" * total_width + "┐")

    # Iterate over each data point to print its corresponding bar
    for label, value in zip(labels, data):
        scaled_value = int(value * scale)  # Scale the value
        bar = '█' * scaled_value
        value_str = f"{str(value)[:9]:>9}" if isinstance(value, int) else f"{value:9.3f}"  # Format the value with 2 decimal places
        print(f"│ {label:>{label_width}} : {bar: <{max_width}} {value_str} │")

    # Print the bottom border of the box
    print("└" + "─" * total_width + "┘")

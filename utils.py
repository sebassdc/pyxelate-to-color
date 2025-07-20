import numpy as np
import matplotlib.pyplot as plt
import time
import functools

from dataclasses import dataclass

from pyxelate import Pyx, Pal


def timing_decorator(func):
    """Decorator to measure and print execution time of methods."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds

        # Get class name if it's a method
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            # print(f"⏱️  {class_name}.{func.__name__}: {execution_time:.2f}ms")
        else:
            pass
            # print(f"⏱️  {func.__name__}: {execution_time:.2f}ms")

        return result
    return wrapper


@dataclass
class CellIndex:
    x: int
    y: int

@dataclass(frozen=True)
class PixelateOptions:
    downsample_by: int = 20
    upscale: int = 100
    palette: int = 8


class PixelArtProcessor:
    """Handles the main pixelation functionality."""

    def __init__(self, options: PixelateOptions):
        self.options = options
        self.pyx = None

    @timing_decorator
    def pixelate_image(self, image: np.ndarray) -> tuple[np.ndarray, Pyx]:
        """Pixelate an image using the configured options."""
        # Debug: Print the options being used in pixelation
        print(f"🔍 DEBUG - PixelArtProcessor.pixelate_image:")
        print(f"  - Image shape: {image.shape}")
        print(f"  - factor (downsample_by): {self.options.downsample_by}")
        print(f"  - palette: {self.options.palette}")
        print(f"  - upscale: {self.options.upscale}")

        self.pyx = Pyx(
            factor=self.options.downsample_by,
            palette=self.options.palette,
            upscale=self.options.upscale,
        )

        # Debug: Print what was actually set in Pyx
        print(f"🔍 DEBUG - Pyx object created with:")
        print(f"  - factor: {self.pyx.factor}")
        print(f"  - palette: {self.pyx.palette}")
        print(f"  - upscale: {self.pyx.upscale}")

        # fit an image, allow Pyxelate to learn the color palette
        self.pyx.fit(image)

        # transform image to pixel art using the learned color palette
        new_image = self.pyx.transform(image)

        # Debug: Print output image info
        print(f"🔍 DEBUG - Pixelation completed:")
        print(f"  - Original image shape: {image.shape}")
        print(f"  - Pixelated image shape: {new_image.shape}")

        return new_image, self.pyx


class GridRenderer:
    """Handles grid drawing and cell operations."""

    def __init__(self, cell_size: int, font_scale_factor: float = .6):
        self.cell_size = cell_size
        self.font_scale_factor = font_scale_factor

    def get_mask(self, cell_index: CellIndex, border_width=0) -> tuple[slice, slice]:
        """Get the mask for a specific cell."""
        start_mask_x = cell_index.x * self.cell_size
        end_mask_x = (cell_index.x + 1) * self.cell_size

        start_mask_y = cell_index.y * self.cell_size
        end_mask_y = (cell_index.y + 1) * self.cell_size

        mask = (
            slice(start_mask_y + border_width, end_mask_y - border_width),
            slice(start_mask_x + border_width, end_mask_x - border_width),
        )
        return mask

    @timing_decorator
    def get_cell_color(self, cell_index: CellIndex, image):
        """Get the color of a specific cell."""
        image_zone = image[self.get_mask(cell_index, border_width=1)]
        # Get the center pixel of the cell zone dynamically
        center_y, center_x = image_zone.shape[0] // 2, image_zone.shape[1] // 2
        return image_zone[center_y, center_x]


    @timing_decorator
    def create_draw_grid(self, image: np.ndarray, pyx: Pyx) -> np.ndarray:
        """Optimized version that renders all text at once instead of per-cell."""
        transformed_image = image.copy()
        # Calculate separate grid sizes for width and height to support non-square images
        grid_height = int(transformed_image.shape[0] / self.cell_size)
        grid_width = int(transformed_image.shape[1] / self.cell_size)
        total_cells = grid_width * grid_height
        print(f"🚀 OPTIMIZED - Grid size: {grid_width}x{grid_height} = {total_cells} cells")

        color_mapper = ColorMapper(pyx.colors.reshape(-1, 3))

        # First pass: draw all borders and backgrounds
        border_start = time.perf_counter()
        BLACK = np.array([0, 0, 0])
        WHITE = np.array([255, 255, 255])

        # Use separate ranges for width and height
        for i in range(grid_width):
            for j in range(grid_height):
                ci = CellIndex(x=i, y=j)
                # Draw border and background
                transformed_image[self.get_mask(ci)] = BLACK
                transformed_image[self.get_mask(ci, border_width=1)] = WHITE

        border_time = (time.perf_counter() - border_start) * 1000
        print(f"⏱️  Border/background drawing: {border_time:.2f}ms")

        # Second pass: collect all text positions and characters
        text_start = time.perf_counter()
        text_positions = []
        characters = []

        for i in range(grid_width):
            for j in range(grid_height):
                ci = CellIndex(x=i, y=j)
                cell_color = self.get_cell_color(ci, image)
                color_index = color_mapper.get_color_index(cell_color)

                # Calculate normalized position - fix dimension usage
                margin = (1 - self.font_scale_factor)/2
                font_x = self._normalize_font_position(ci.x, margin, image.shape[1])  # Use width for x
                font_y = self._normalize_font_position(ci.y, margin, image.shape[0])  # Use height for y

                # Y-Axis is inverted for fig
                font_y = 1 - font_y

                text_positions.append((font_x, font_y))
                characters.append(str(color_index))

        # Render all text at once with explicit figure settings
        img_height, img_width = transformed_image.shape[:2]
        # Set figure size based on image dimensions to maintain consistent scaling
        fig_width = img_width / 100  # Convert pixels to inches (assuming 100 DPI)
        fig_height = img_height / 100

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
        fig.figimage(transformed_image, resize=False)  # Don't resize, use exact size

        # Calculate font size relative to actual figure dimensions
        # Use a smaller scale factor for web environments
        effective_font_scale = self.font_scale_factor * 0.7  # Reduce by 30%
        calculated_fontsize = self.cell_size * effective_font_scale

        for (x, y), char in zip(text_positions, characters):
            fig.text(
                x, y, char,
                fontsize=calculated_fontsize,
                va="top",
                ma="center",  # Use ha instead of ma (ma is deprecated)
                color='black'
            )

        fig.canvas.draw()
        annotated_img = np.asarray(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        text_time = (time.perf_counter() - text_start) * 1000
        total_optimized_time = border_time + text_time

        print(f"⏱️  Text collection & rendering: {text_time:.2f}ms")
        print(f"⏱️  Total optimized time: {total_optimized_time:.2f}ms")
        print(f"🎯 Performance: {total_optimized_time/total_cells:.2f}ms per cell")
        print(f"📏 Font size used: {calculated_fontsize}")

        return annotated_img[:, :, :3]

    def _normalize_font_position(self, pos: int, margin: float, size) -> float:
        """Normalize font position within a cell."""
        factor = self.cell_size / size
        offset = factor * margin
        return (factor * pos) + offset


class ColorMapper:
    """Handles color mapping operations."""

    @timing_decorator
    def __init__(self, colors: np.ndarray):
        self.colors = colors
        self.color_map = self._create_color_map()

    @timing_decorator
    def get_color_index(self, color: np.ndarray) -> int:
        """Get the index of a color in the color map."""
        color_hash = self._hash_color(color)
        return self.color_map[color_hash]

    def _hash_color(self, color: np.ndarray) -> str:
        """Create a hash string from a color array."""
        return "".join(str(val) for val in color)

    def _create_color_map(self) -> dict:
        """Create a mapping from color hashes to indices."""
        result = {}
        for i, color in enumerate(self.colors):
            color_hash = self._hash_color(color)
            result[color_hash] = i
        return result


# Legacy function wrappers for backward compatibility
@timing_decorator
def pixelate_image(image: np.ndarray, options: PixelateOptions) -> tuple[np.ndarray, Pyx]:
    """Legacy wrapper for backward compatibility."""
    processor = PixelArtProcessor(options)
    return processor.pixelate_image(image)


@timing_decorator
def create_draw_grid(image: np.ndarray, cell_size: int, pyx: Pyx) -> np.ndarray:
    """Legacy wrapper for backward compatibility."""
    renderer = GridRenderer(cell_size)
    return renderer.create_draw_grid(image, pyx)


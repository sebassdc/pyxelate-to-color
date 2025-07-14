import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    mo.md("# Welcome to marimo! 🌊🍃")
    return mo, np, plt


@app.cell
def _():
    from skimage import io
    from pyxelate import Pyx, Pal
    return Pyx, io


@app.cell
def _(io):
    # load image with 'skimage.io.imread()'
    image = io.imread("examples/cute-panda.webp")  

    return (image,)


@app.cell
def _(Pyx, image, mo, plt):
    # 1) Instantiate Pyx transformer
    downsample_by = 20
    upscale = downsample_by * 5
    palette= 8
    pyx = Pyx(
        factor=downsample_by,
        palette=palette,
        upscale=upscale
    )
    # 2) fit an image, allow Pyxelate to learn the color palette
    pyx.fit(image)

    # 3) transform image to pixel art using the learned color palette
    new_image = pyx.transform(image)

    # save new image with 'skimage.io.imsave()'
    # io.imsave("pixel.png", new_image)
    plt.imshow(new_image)
    mo.hstack([
        downsample_by,
        palette,
        plt.show()
    ])


    return downsample_by, new_image, palette, pyx, upscale


@app.cell
def _(upscale):

    #cell_size = downsample_by

    cell_size = upscale
    from dataclasses import dataclass
    from typing import Tuple

    @dataclass
    class CellIndex:
        x: int
        y: int

    def get_mask(cell_index: CellIndex, border_width=0) -> Tuple[slice, slice]:
        start_mask_x = cell_index.x * cell_size
        end_mask_x = (cell_index.x + 1) * cell_size

        start_mask_y = cell_index.y * cell_size
        end_mask_y = (cell_index.y + 1) * cell_size

        mask = (
            slice(start_mask_y + border_width, end_mask_y - border_width),
            slice(start_mask_x + border_width, end_mask_x - border_width),
        )
        return mask
    return CellIndex, cell_size, get_mask


@app.cell
def _(CellIndex, cell_size, get_mask, np, plt):
    def normalize_font_position(pos: int, margin: float, size):
        factor = cell_size / size
        offset = factor * margin 
        return (factor * pos) + offset


    def draw_char_on_cell(character, image, font_position: CellIndex):
        BLACK = np.array([0, 0, 0])
        WHITE = np.array([255, 255, 255])
        border_color = BLACK
        bg_color = WHITE

        border_width = 1
        image[get_mask(font_position)] = border_color
        image[get_mask(font_position, border_width=border_width)] = bg_color
        # value between 0 and 1
        font_scale_factor = .6
        margin = (1 - font_scale_factor)/2
        font_position_coord_x = normalize_font_position(
            font_position.x,
            margin,
            size=image.shape[0],
        )
        font_position_coord_y = normalize_font_position(
            font_position.y,
            margin,
            size=image.shape[1]
        )
        fig = plt.figure()
        fig.figimage(image, resize=True)
        fig.text(
            font_position_coord_x,
            1-font_position_coord_y,
            character,
            fontsize=cell_size * font_scale_factor,
            va="top",
            ma="center"
        )
        fig.canvas.draw()
        annotated_img = np.asarray(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        return annotated_img[:, :, :3]
    return (draw_char_on_cell,)


@app.cell
def _(pyx):
    def hash_color(color):
        res = ""
        for val in color:
            res += str(val)
        return res
    def create_color_map(colors):
        result = {}
        for i, color in enumerate(colors):
            print(type(color))
            color_hash = hash_color(color)
            result[color_hash] = i
        return result

    color_map = create_color_map(pyx.colors.reshape(-1, 3))
    color_map
    return color_map, hash_color


@app.cell
def _(CellIndex, get_mask):
    def get_cell_color(cell_index: CellIndex, image):
        image_zone = image[get_mask(cell_index)]
        return(image_zone[3][3])
    return (get_cell_color,)


@app.cell
def _(pyx):
    pyx.colors
    return


@app.cell
def _(
    CellIndex,
    cell_size,
    color_map,
    draw_char_on_cell,
    get_cell_color,
    hash_color,
    new_image,
):
    transformed_image = new_image.copy()
    grid_size = int(transformed_image.shape[0] / cell_size)
    grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            ci = CellIndex(x=i, y=j)
            cell_color = get_cell_color(ci, new_image)
            color_index = color_map[hash_color(cell_color)]
            transformed_image = draw_char_on_cell(str(color_index), transformed_image, ci)



    return (transformed_image,)


@app.cell
def _(plt, transformed_image):
    plt.imshow(transformed_image)
    plt.show()
    return


@app.cell
def _(downsample_by, io, palette, transformed_image):
    io.imsave(f"down_{downsample_by}_pallete_{palette}.png", transformed_image)
    return


@app.cell
def _(downsample_by, io, new_image, palette):
    io.imsave(f"down_{downsample_by}_pallete_{palette}_origin.png", new_image)
    return


@app.cell
def _(transformed_image):
    transformed_image
    return


@app.cell
def _(pyx):
    pyx.colors
    return


if __name__ == "__main__":
    app.run()

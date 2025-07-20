import marimo

app = marimo.App()

with app.setup:

    import marimo as mo
    import matplotlib.pyplot as plt
    import utils

    mo.md("# Testing grounds 🌊🍃")

    from skimage import io
    help(utils)




@app.cell
def _():
    image = io.imread("examples/cute-panda.webp")

    plt.imshow(image)
    plt.show()

@app.cell
def _():
    upscale = 100
    pixelate_options = utils.PixelateOptions(
        downsample_by=20,
        upscale=upscale,
        palette=8,
    )

    cell_size = upscale

    pyxelated_image, pyx = utils.pixelate_image(
        image=image,
        options=pixelate_options
    )
    plt.imshow(pyxelated_image)
    plt.show()



@app.cell
def _():
    result_image = utils.create_draw_grid(
        pyxelated_image,
        cell_size=cell_size,
        pyx=pyx,
    )

    plt.imshow(result_image)
    plt.show()


    #
    # io.imsave(f"down_{downsample_by}_pallete_{palette}.png", transformed_image)
    #
    # io.imsave(f"down_{downsample_by}_pallete_{palette}_origin.png", new_image)
    #
    # transformed_image
    #
    # pyx.colors
    #

if __name__ == "__main__":
    app.run()

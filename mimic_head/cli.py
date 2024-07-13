import click
import gradio as gr

from .mimic_head import MimicHeadSDK


def func():
    print("hello")


@click.group()
@click.version_option()
def cli():
    """
    command
    """


@cli.command()
@click.option("-d", "--device", default=None, help="network inference device")
@click.option("-h", "--host", default="0.0.0.0", help="gradio server running host")
@click.option("-p", "--port", default=7860, help="gradio server running port")
def run(device, host, port):
    sdk = MimicHeadSDK()

    webcam_interface = gr.Interface(
        sdk.process,
        inputs=[
            gr.Image(label="Source Image"),
            gr.Image(sources=["webcam"], streaming=True, label="Webcam"),
        ],
        outputs=[
            gr.Image(),
        ],
        live=True,
    )
    video_interface = gr.Interface(
        sdk.process_video,
        inputs=[
            gr.Image(label="Source Image"),
            gr.Video(sources=["upload"], label="Driving Video"),
        ],
        outputs=[
            gr.Video(autoplay=True),
        ],
        live=True,
    )
    img_interface = gr.Interface(
        fn=sdk.process,
        inputs=[
            gr.Image(label="Source Image"),
            gr.Image(sources=["upload"], label="Driving Image"),
        ],
        outputs=[
            gr.Image(),
        ],
        live=True,
    )

    with gr.Blocks() as ui:
        gr.Markdown(
            "<center><h1>mimic_head: Unofficial One-click Version of LivePortrait, with Webcam Support </h1></center>"
        )
        gr.Markdown("<center>deploy: pip install mimic_head && mimic_head run</center>")
        gr.Markdown(
            "<center>credit: adapted from <a href=https://github.com/KwaiVGI/LivePortrait>LivePortrait</a></center>"
        )
        gr.Markdown(
            "<center>source code: <a href=https://github.com/vra/mimic_head>mimic_head</a></center>"
        )

        demo = gr.TabbedInterface(
            [
                webcam_interface,
                video_interface,
                img_interface,
            ],
            tab_names=[
                "Webcam Demo",
                "Video Demo",
                "Image Demo",
            ],
        )

    ui.launch(server_name=host, server_port=port)

import asyncio
import torch
from mmdet.apis import init_detector, async_inference_detector
from mmdet.utils.contextmanagers import concurrent

async def main():
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'
    device = 'cuda:0'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    streamqueue = asyncio.Queue()
    streamqueue_size = 3

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device=device))

    img = 'demo/demo.jpg'

    async with concurrent(streamqueue):
        result = await async_inference_detector(model, img)

    model.show_result(img, result)
    model.show_result(img, result, out_file='demo/result.jpg')

if __name__ == '__main__':
    asyncio.run(main())

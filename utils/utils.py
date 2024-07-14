import cv2


def create_video(source, fps=60, output_name='output'):
    out = cv2.VideoWriter(
        output_name + '.mp4',
        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
        fps,
        (source[0].shape[1], source[0].shape[0])
    )
    for i in range(len(source)):
        out.write(source[i])
    out.release()

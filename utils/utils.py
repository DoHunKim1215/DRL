import cv2
import gymnasium as gym


def get_make_env_fn(**kwargs):
    def make_env_fn(env_name, unwrapped=False, inner_wrappers=None, outer_wrappers=None):
        env = gym.make(env_name, render_mode='rgb_array')
        env = env.unwrapped if unwrapped else env

        if inner_wrappers:
            for wrapper in inner_wrappers:
                env = wrapper(env)

        if outer_wrappers:
            for wrapper in outer_wrappers:
                env = wrapper(env)

        return env

    return make_env_fn, kwargs


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

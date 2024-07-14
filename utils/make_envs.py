from model.actor_critic.multiprocess_env import MultiprocessEnv


def make_envs_fn(mef, mea, s, n):
    return MultiprocessEnv(mef, mea, s, n)
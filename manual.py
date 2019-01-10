import time
import argparse
import gym
import envs

def main():
    parser = argparse.ArgumentParser("minigrid manual control")
    parser.add_argument("-e", "--env-name", default="MiniGrid-SimpleEnv-5x5-v0")
    args = parser.parse_args()

    # Load environment
    env = gym.make(args.env_name)
    env.reset()

    renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            env.reset()
            return

        if keyName == 'ESCAPE':
            exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.forward

        elif keyName == 'SPACE':
            action = env.actions.toggle
        elif keyName == 'PAGE_UP':
            action = env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = env.actions.drop

        elif keyName == 'RETURN':
            action = env.actions.done

        else:
            print("unknown key %s" % keyName)
            return

        obs, reward, done, info = env.step(action)

        print('step=%s, reward=%.2f' % (env.step_count, reward))

        if done:
            print('done!')
            env.reset()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break

if __name__ == "__main__":
    main()

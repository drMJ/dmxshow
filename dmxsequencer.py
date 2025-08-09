import pyenttec
import numpy as np
import time
import random
import yaml
import argparse


def clamp(value, min_value, max_value):
    """Clamp the value between min_value and max_value."""
    return max(min(value, max_value), min_value)

def rand_sweep(channels, on=255, off=0):
    # track iterator fn turning one and only one channel on, the rest  off
    # the channel is chosen at random from the immediate neighbors of the current channel
    # yields a cue definition for each step
    current_channel_id = random.randint(0, len(channels) - 1)
    while True:
        keyframe = {channels[i]: on if i == current_channel_id else off for i in range(len(channels))}
        current_channel_id = clamp(current_channel_id + random.choice([-1, 1]), 0, len(channels) - 1)
        yield keyframe

def transition(transition_type, duration, current):
    # Return the appropriate transition function based on the type
    if transition_type == 'none':
        return 1
    elif transition_type == 'step':
        return 0 if current < duration/2 else 1
    elif transition_type == 'ramp':
         return current / duration
    elif transition_type == 'cos':
        return (1 - np.cos(np.pi * current / duration))/2
    else:
        return eval(transition_type, globals(), locals())

class Fixture:
    def __init__(self, data, address=1):
        for key, value in data.items():
            if isinstance(value, dict):
                value = Fixture(value, address)
            else:
                value += address-1
            setattr(self, key, value)

class DMXSequencer:
    def __init__(self, show, dmx_port='COM5', size=512):
        fixture_defs = show.get('fixtures', [])
        self.kinds = show.get('kinds', {})
        # instantiate the fixtures based on the kinds
        self.fixtures = {}
        for fixture_name, fixture_def in fixture_defs.items():
            self.fixtures[fixture_name] = Fixture(self.kinds[fixture_def['type']], fixture_def['address'])

        self.locals = self.fixtures.copy()

        self.scenes = show.get('scenes', [])
        self.sequence = show.get('sequence', [])
        self.timestep = 0.01
        self.dmx = pyenttec.DMXConnection(dmx_port, univ_size=size)

    def run(self):
        try:
            # while True:
                self.play_scene(self.scenes[0]['one_random_sweep'])
        except KeyboardInterrupt:
            self.dmx.close()

    def play_scene(self, scene_data):
        # instantiate a scene as a collection of iterators, one for each track, that transition through the keyframes based on time and timestep
        tracks = [self.make_track(track_data, scene_data['keyframes']) for track_name, track_data in scene_data['tracks'].items()]
        for t in np.arange(0, scene_data['duration'], self.timestep):
            for track in tracks:
                next(track, None)
            self.dmx.render()
            time.sleep(self.timestep)


    def make_track(self, track_data, keyframes):
        # Create a track as an iterator that transitions through the keyframes based on time and timestep
        self.locals['keyframes'] = keyframes
        while True:
            if isinstance(track_data, str):
                track_data = eval(track_data, locals=self.locals)  # Convert string representation to actual data
            for cue in track_data:
                if isinstance(cue, str):
                    (keyframe, duration, transition_type) = eval(cue, locals=self.locals)
                else:
                    (keyframe, duration, transition_type) = self.make_cue(cue, keyframes)

                # keyframe_def can be one of: string name of a predefined keyframe, a list of predefined key names, a dict of channel-value pairs or a string code that returns a dict
                while not isinstance(keyframe, dict):
                    if isinstance(keyframe, str) and keyframe in keyframes:
                        keyframe = self.make_value(keyframes[keyframe])
                    else:
                        keyframe = self.make_value(keyframe)

                # fix a target for each channel
                keyframe_instance = {}
                for channel, target_value in keyframe.items():
                    # if either is a list, sample from it
                    channel = self.make_value(channel)
                    target_value = self.make_value(target_value)
                    keyframe_instance[channel] = target_value
                #apply the transition function to each channel in the keyframe
                start_values = self.dmx.dmx_frame.tolist()
                for t in np.arange(0, duration, self.timestep):
                    for channel, target_value in keyframe_instance.items():
                        channel = int(channel)
                        start_value =start_values[0+channel-1] # todo: use the fixture base address instead of 0
                        current_value = int(transition(transition_type, duration, t) * (target_value - start_value) + start_value)
                        self.dmx[0+channel-1] = current_value
                    yield

    def make_cue(self, cue, keyframes):
        keyframe = cue[0]
        if not isinstance(keyframe, str) or keyframe not in keyframes:
            keyframe = self.make_value(keyframe)
        duration = self.make_value(cue[1])
        transition_type = cue[2] if len(cue) > 2 else 'none'
        
        return (keyframe, duration, transition_type)
    
    def make_value(self, val):
        if isinstance(val, str):
            return eval(val, locals=self.locals)  # Convert string representation to actual data
        elif isinstance(val, list):
            return random.choice(val)
        return val
            


def run():
    # if no cmdline args, load from a known JSON file
    parser = argparse.ArgumentParser(description='Run a DMX sequencer.')

    parser.add_argument('-p', '--port', type=str, default='32404', help='UDP port to use for OSC communication')
    parser.add_argument('-d', '--dmx', type=str, default='COM5', help='DMX port to use')
    parser.add_argument('-s', '--size', type=int, default=512, help='Total number of DMX channels, 24-512')
    parser.add_argument('-f', '--file', type=str, default='show.yaml', help='Path to the scene definition YAML file')
    args = parser.parse_args()
    with open(args.file, 'r') as f:
        show =  yaml.load(f, Loader=yaml.FullLoader)
    seq = DMXSequencer(show, dmx_port=args.dmx, size=args.size)
    seq.run()
    
if __name__ == "__main__":
    run()



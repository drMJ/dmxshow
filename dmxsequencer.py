import threading
import pyenttec
import numpy as np
import time
import random
import yaml
import argparse
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer


def clamp(value, min_value, max_value):
    """Clamp the value between min_value and max_value."""
    return max(min(value, max_value), min_value)

def rand_sweep(channels, on=255, off=0):
    """
    Generate a random sweep pattern across channels.
    
    Creates an infinite iterator that turns one channel on at a time while keeping
    all others off. The active channel randomly moves to an adjacent neighbor,
    creating a sweeping effect.
    
    Args:
        channels: List of channel identifiers to sweep across
        on (int): Value to set for the active channel (default: 255)
        off (int): Value to set for inactive channels (default: 0)
    
    Yields:
        dict: Keyframe definition mapping each channel to its value (on/off)
    """
    # track iterator fn turning one and only one channel on, the rest  off
    # the channel is chosen at random from the immediate neighbors of the current channel
    # yields a cue definition for each step
    current_channel_id = random.randint(0, len(channels) - 1)
    while True:
        keyframe = {channels[i]: on if i == current_channel_id else off for i in range(len(channels))}
        current_channel_id = clamp(current_channel_id + random.choice([-1, 1]), 0, len(channels) - 1)
        yield keyframe

def transition(transition_type, duration, current):
    """
    Calculate transition value based on type, duration, and current time.
    
    Supports various transition types for smooth animations between keyframes.
    
    Args:
        transition_type (str): Type of transition ('none', 'step', 'ramp', 'cos', or custom expression)
        duration (float): Total duration of the transition
        current (float): Current time within the transition
    
    Returns:
        float: Transition value between 0 and 1, representing progress through the transition
    """
    # Return the appropriate transition function based on the type
    if current >= duration or transition_type == 'none':
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
    """
    Represents a DMX fixture with its channel mapping. 
    Used for resolving named channel references (e.g. 'par1.red') to channel values.
    """
    def __init__(self, name, definition, types):
        """
        Creates a fixture object by processing the fixture definition data and
        adjusting channel numbers based on the fixture's DMX address. 
        Allows sub-fixtures, e.g. an RGBW light within a multi-head fixture. 
        
        Args:
            name (str): Name of the fixture
            definition (dict): Fixture definition containing fixture type and address
            types (dict): Dictionary of fixture types and their channel mappings
        """
        self.name = name
        self.address = definition.get('address', None)
        # if the definition has a type, it is an indirection to a fixture type
        data = types[definition['type']]  if 'type' in definition else definition
        self.macros = data.get('macros', {})
        for key, value in data['channels'].items():
            if isinstance(value, dict):
                value = Fixture(key, value, types)
            else:
                value += self.address-1
            setattr(self, key, value)
        if self.address is None:
            self.address = min([channel if isinstance(channel, int) else getattr(self, name).address for name, channel in data['channels'].items()])

class DMXController:
    def __init__(self, port, univ_size=24):
        if port:
            self.dmx = pyenttec.DMXConnection(port, univ_size=univ_size)
            self.get_frame = self.dmx.dmx_frame.tolist
            self.render = self.dmx.render
            self.close = self.dmx.close
        else:
            self.dmx = [0]*univ_size
            self.get_frame = lambda: self.dmx.copy()
            self.render = lambda: print(self.dmx)
            self.close = lambda: None

    def __setitem__(self, channel, value):
        self.dmx[channel] = value


class DMXSequencer:
    def __init__(self, show, dmx_port='COM5', osc_port=34201):
        """
        Initialize the DMX sequencer with show data and communication settings.
        
        Sets up DMX output, OSC server for remote control, and prepares fixtures
        and scenes from the show definition.
        
        Args:
            show (dict): Show definition containing fixtures and scenes
            dmx_port (str): Serial port for DMX output (default: 'COM5')
            osc_port (int): UDP port for OSC communication (default: 34201)
        """
        self.timestep = show.get('timestep', 0.01)  # Timestep for DMX update (default: 0.01 seconds)
        fixture_defs = show.get('fixtures', [])
        self.fixture_types = show.get('fixture_types', {})
        # instantiate the fixtures based on the fixture_types
        self.fixtures = {}
        size = show.get('universe_size', 512)  # Default DMX universe size
        if fixture_defs:
            for fixture_name, fixture_def in fixture_defs.items():
                self.fixtures[fixture_name] = Fixture(fixture_name, fixture_def, self.fixture_types)
        
        self.locals = {**self.fixtures.copy(), **show.get('macros', {})}
        self.scenes = show.get('scenes', [])
        self.sequence = show.get('sequence', [])
        self.active_group = list(self.scenes.keys())[0]
        self.dmx = DMXController(dmx_port, univ_size=size)
        self.osc_dispatcher = Dispatcher()
        self.osc_dispatcher.set_default_handler(self.osc_handler)
        self.osc_server = BlockingOSCUDPServer(("127.0.0.1", osc_port), self.osc_dispatcher)
        self.osc_thread = threading.Thread(target=self.osc_server.serve_forever, daemon=True)
        self.osc_thread.start()

    def osc_handler(self, addr, *args):
        # handles osc messages on a background thread
        if addr == "/settings/active_group":
            self.active_group = args[0]

    def run(self):
        """
        Main execution loop for the DMX sequencer.
        
        Continuously selects and plays scenes from the active group.
        Each scene runs for its specified duration, with tracks executing
        in parallel. Responds to group changes via OSC and handles graceful
        shutdown on keyboard interrupt.
        """
        try:
            while True:
                group = self.active_group
                scene_name = random.choice(list(self.scenes[group].keys()))
                scene_data = self.scenes[group][scene_name]
                keyframes = scene_data.get('keyframes', {})
                # a scene is a collection of iterators, one for each track, that transition through the keyframes based on time and timestep
                print(f"[{time.perf_counter()}] Playing {group}/{scene_name} for {scene_data['duration']} seconds")
                tracks = [self.make_track(track_data, keyframes) for track_name, track_data in scene_data['tracks'].items()]
                duration = self.make_value(scene_data['duration'])
                for t in np.arange(0, duration, self.timestep):
                    for track in tracks:
                        next(track, None)
                    self.dmx.render()
                    time.sleep(self.timestep)
                    if self.active_group != group:
                        break
        except KeyboardInterrupt:
            self.dmx.close()


    def make_track(self, track_data, keyframes):
        """
        Create a track iterator that transitions through cues over time.
        
        Converts track definitions into an infinite iterator that yields DMX values
        for each timestep. Handles cue evaluation, keyframe resolution, and smooth
        transitions between states.
        
        Args:
            track_data: Track definition (string expression or list of cues)
            keyframes (dict): Named keyframe definitions in the scene
        
        Yields:
            None: Side effect of updating DMX channel values during iteration
        """
        # Create a track as an iterator that transitions through the keyframes based on time and timestep
        self.locals['keyframes'] = keyframes
        while True:
            if isinstance(track_data, str):
                track_data = eval(track_data, globals(), self.locals)  # Convert string representation to actual data
            for cue in track_data:
                if isinstance(cue, str):
                    (keyframe, duration, transition_type) = eval(cue, globals(), self.locals)
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
                    channel = self.make_value(channel)
                    target_value = self.make_value(target_value, channel.macros if isinstance(channel, Fixture) else None)
                    if isinstance(channel, Fixture) and isinstance(target_value, dict): 
                        target_value = {k:self.make_value(v) for k, v in target_value.items()}
                    keyframe_instance[channel] = target_value
                #apply the transition function to each channel in the keyframe
                duration = self.make_value(duration)
                start_values = self.dmx.get_frame()
                for t in np.arange(self.timestep, duration + self.timestep, self.timestep):
                    for channel, target_value in keyframe_instance.items():
                        self.apply_transition(transition_type, channel, target_value, start_values, duration, t)
                    yield

    def apply_transition(self, transition_type, channel, target_value, start_values, duration, current):
        # if target value is a multi-channel macro, apply the transition to each channel
        if not isinstance(target_value, dict): # not a multi-value macro
            target_value = {1: target_value}
        base_channel = channel.address if isinstance(channel, Fixture) else int(channel)
        # apply the transition function to each channel 
        for channel, value in target_value.items():
            target_channel = base_channel + channel - 1
            start_value = start_values[target_channel - 1]
            current_value = int(transition(transition_type, duration, current) * (value - start_value) + start_value)
            self.dmx[target_channel-1] = current_value
            

    def make_cue(self, cue, keyframes):
        """
        Process a cue definition into its component parts.
        
        Extracts keyframe, duration, and transition type from a cue definition,
        applying default values and resolving dynamic expressions as needed.
        
        Args:
            cue (list): Cue definition [keyframe, duration, transition_type]
            keyframes (dict): Available keyframe definitions
        
        Returns:
            tuple: (keyframe, duration, transition_type) ready for execution
        """
        keyframe = cue[0]
        if not isinstance(keyframe, str) or keyframe not in keyframes:
            keyframe = self.make_value(keyframe)
        duration = self.make_value(cue[1])
        transition_type = cue[2] if len(cue) > 2 else 'none'
        
        return (keyframe, duration, transition_type)
    
    def make_value(self, val, locals=None):
        """
        Resolve a value that may be dynamic or randomized.
        
        Processes different value types to return concrete values:
        - Strings are evaluated as Python expressions
        - Lists have a random element selected
        - Other values are returned as-is
        
        Args:
            val: Value to resolve (str, list, or literal value)
        
        Returns:
            Resolved concrete value
        """
        if isinstance(val, str):
            return eval(val, globals(), {**self.locals, **(locals or {})})
        elif isinstance(val, list):
            return random.choice(val)
        return val
            


def run():
    """
    Main entry point for the DMX sequencer application.
    
    Parses command line arguments, loads the show definition from YAML,
    creates a DMXSequencer instance, and starts the main execution loop.
    
    Command line options:
        -p, --port: OSC UDP port (default: 34201)
        -d, --dmx: DMX serial port (default: COM5)
        -f, --file: Show definition YAML file (default: show.yaml)
    """
    # if no cmdline args, load from a known JSON file
    parser = argparse.ArgumentParser(description='Run a DMX sequencer.')

    parser.add_argument('-p', '--port', type=int, default=34201, help='UDP port to use for OSC communication')
    parser.add_argument('-d', '--dmx', type=str, default='COM5', help='DMX port to use')
    parser.add_argument('-f', '--file', type=str, default='show.yaml', help='Path to the scene definition YAML file')
    args = parser.parse_args()
    with open(args.file, 'r') as f:
        show =  yaml.load(f, Loader=yaml.FullLoader)
    seq = DMXSequencer(show, osc_port=args.port, dmx_port=args.dmx)
    seq.run()

def test(filename, **kwargs):
    with open(filename, 'r') as f:
        show =  yaml.load(f, Loader=yaml.FullLoader)
    seq = DMXSequencer(show, **kwargs)
    seq.run()
    
if __name__ == "__main__":
    # run()
    test("show.yaml", osc_port=34201, dmx_port=None)


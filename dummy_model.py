class DummyAgent:
    def __init__(self):
        self.tick_counter = 0
        # A simple sequence of actions to test the car's behavior
        self.action_sequence = ['ACCELERATE'] * 100 + ['STEER_RIGHT'] * 45 + ['ACCELERATE'] * 200

    def act(self, state):
        """
        Returns a single action based on a simple, predefined sequence.
        """
        action = self.action_sequence[self.tick_counter % len(self.action_sequence)]
        self.tick_counter += 1
        return action

    def load(self, path):
        """Dummy load method, does nothing."""
        print(f"DummyAgent: load method called with {path}, but no model to load.")
        pass

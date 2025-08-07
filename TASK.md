# TASK

Create a lane swithing agent/controller using DQN.

This agents task is to either switch to the left or right lane, or do nothing (stay in its own lane and let the cruise control do its job, keeping pace with the car infornt of it/ behind it whithout colliding with it (this cruise controller is woriking quite good, is well tested and you can assume that its working perfectly as long as the car doesnt change lane)).

A lane change in itself is done by using either of the helper functions:

initate_change_lane_left()
initate_change_lane_right()

These lane change functions have been carefully implemeted and tuned to ensure that the car within a short timespan in practice turns off its cruise control (i.e. it neither accelerates nor decelerates (along the x-direction)) and executes a smooth lane change (along the y-direction), ending up in the center of the desired lane.

Your task now is to train a DQN agent to perform lane changes, such that we maximize the distance travelled and can drive as fast as possible without crashing into any cars in adjacent lanes. 

Even though you train a DQN agent, you should as a safety measure also add a cool-down period of at least N=3 seconds after the agent has made a lane change before it can make another lane change. 


It is important that we desing the agent and reward function in such a way that the agent makes wise decisions of lane changes, and does them in a manner that will maximize the distance travelled and the speed of the car. 

Also: since there are only 5 lanes and our car spawns in the middle, punish heavily if the agent tries to change lane and hits the wall (will happen if you turn left in the leftmost lane or turn right in the rightmost lane).

EXTRA:
This means we might have to take into consideration the fact that some of the cars in adjacent lanes that spawn in front of the car will likely be slower than our own car and that cars that spawn behind our car will likely be faster than our own car. 
So we might want to ensure the car changes lanes to infront of the slower cars that we pass by, and change lane into the lane behind the cars that spawn behind, surpass us with a faster speed.

To ensure that our agent is not a simple time-invariant controller and should take into consideration of the fact that adjacent cars might increase or decrease their own speed, we want to 
This means that we should set up a way to keep track of and feed into our model a selection of the last N sensor values. We aim to maximize the amount of information the model recieves while minimizing the feature vector size. Thus. Specifically lets do it using geometric backoff, so lets say i include:
[k]
[k-1]
[k-2]
[k-4]
[k-8]
[k-16]
[k-32]
where k are the sensor values at time step/tick k, k-1, k-2, k-4, k-8, k-16, k-32. (temporal stacking / "frame stacking" or "state stacking")


TO SUMMARIZE:

These are the inputs to the agent:
self.sensor_names = [
            'left_side', 'left_side_front', 'left_front', 'front_left_front',
            'front', 'front_right_front', 'right_front', 'right_side_front',
            'right_side', 'right_side_back', 'right_back', 'back_right_back',
            'back', 'back_left_back', 'left_back', 'left_side_back'
        ]
(with temporal stacking on all of these)

And the outputs is one of three executions to be made

1. lane change left
2. lane change right
3. do nothing

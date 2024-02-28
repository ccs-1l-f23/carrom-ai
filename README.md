# carrom-ai

Reinforcement learning agent to play the South-Asian tabletop board game _Carrom_ (similar to billiards). CCS 1L project authored by Rohil Shah. See video and summary below for more details.


https://github.com/ccs-1l-f23/carrom-ai/assets/47135708/88a6fbe5-c614-4a1d-8ac5-b9ac953953ef

**Auto-generated Video Summary:**

1. **Understanding Carrom:**
   - Brief overview: A two-player tabletop game involving flicking a striker to pocket coins.
2. **Project Structure:**
   - **Simulation:**
     - Leveraged an [open-source simulator](https://github.com/samiranrl/Carrom_rl/tree/master) from IIT Bombay.
     - Written in Python, using Pymunk for 2D physics and Pygame for graphics.
     - Customized the open-source code to suit my project's needs.

   - **Hand-Coded Algorithms:**
     - Implemented various strategies, including a random-coin player and a center-of-mass player.
     - Aimed for a progression in complexity to facilitate performance measurement.

   - **Reinforcement Learning:**
     - Initial steps involved standardizing the simulation into an "environment."
     - Utilized Proximal Policy Optimization (PPO) as the reinforcement learning algorithm.

3. **Training Progress:**
   - See videos demonstrating the simulation and hand-coded algorithms.
   - See the win-rate graph of the reinforcement learning algorithm (PPO) against a randomly shooting player.

4. **Next Steps:**
   - **Simulation Enhancement:**
     - There is a need for optimization to improve speed and iteration time.
     - Potential adjustments to the physics to better mimic real-life carrom conditions.

   - **Hand-Coded Agents Improvement:**
     - Could develop more intelligent algorithms with advanced strategies like pocket targeting and defensive maneuvers.

   - **Reinforcement Learning Opportunities:**
     - PPO algorithm could be optimized.
     - Can try other reinforcement learning algorithms such as Q-learning or model-based approaches.
     - Genetic algorithms like NEAT are potential alternatives for achieving positive results.

## Running the environment
1. Ensure that you have Python 3.9 installed.
2. Install the dependencies using `python -m pip install -r requirements.txt`
3. Run two random agents against each other using `python3 main.py`

## carrom-ai
Gymnasium-based environment

## carrom_env
PettingZoo-based environment

## auto
Initial translation from https://github.com/samiranrl/Carrom_rl/blob/master/2_player_server/

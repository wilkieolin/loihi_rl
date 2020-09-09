ProcessNodes & Reinforcement Learning
=====================================

This module contains code for both reinforcement learning tasks on Loihi and the underlying 'ProcessNodes' framework which was used to build the underlying spiking networks.

ProcessNodes focuses on using a computational graph framework allow stereotyped computations with given shapes to be abstracted to nodes. Connectivity and compartments then can be automatically generated and re-generated as the task at hand changes. Current nodes and connectivity methods are listed in *primitives.py*. Examples of how to use nodes are in *node_examples*.

Full examples of networks built using hierarchies of nodes to complete reinforcement learning tasks are in the other subfolders. *Bandit* showcases a solution to the multi-arm bandit problem. *Maze* builds on this to show an agent learning a navigation task. *Blackjack* is the final example, and demonstrates on-chip learning of the card game Blackjack. 
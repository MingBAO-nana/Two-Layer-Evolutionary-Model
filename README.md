# Two-Layer Evolutionary Model Code
## 1 Introduction to the two-Layer Evolutionary Model
　　This research addresses the evolutionary problems of complex systems by constructing a general two-layer evolutionary game model, which consists of the element evolution network and the group game network.
* ___The element evolution network___ presents the correlation and dynamic evolution of driving and physical elements, reflecting the system's overall development.
* ___The group game network___, a complex network with a specific structure composed of the system's participants, allows participants to continuously optimize their strategies through gaming.

　　In our model, we abstract some system elements as strategies participants can choose and establish interaction and feedback between the two layers. We predict the outcome of strategy choice by updating the group's state, simulating changes in physical element proportions under driving element influences, and reflecting system evolution trends.  
　　Due to the finite rationality of participants, the optimal equilibrium of the game cannot be found initially. Therefore, participant strategies must be modified and improved through numerous game iterations. Therefore, the code for the model’s evolutionary process is provided here on GitHub.  
## 2 Code Operation Instructions
### 2.1 Project Overview
* ___(1) "1-Algorithm": Main Program.___
* ___(2) Input Files:___
  * _"Element strategy values and initial market share"_: Represents the initial parameter values of physical elements in the given scenario.
  * _"Market share of component"_: A file that is read during the process, recording the evolutionary state value progressively.
  * _"Technology"_, _"Demand"_, _"Component"_: These files contain sets of driving and physical elements that determine the node coordinates in the element evolution network.
  * _"Technology to component"_, _"Demand to component"_, _"Component to component"_: These files describe the relationships between elements, determining the edges in the element evolution network.
* ___(3) Additional Files:___
  * _"Description of scenario elements"_: Describes the meaning of various elements in the scenario.
  * _"actual data"_: Represents the actual market share data of physical elements in the scenario, which can be used to determine the initial evolution values and for subsequent parameter calibration.
  * _"600_15_1_Evolutionary state value"_, _"600_15_1_Evolutionary trend"_: These are example files that show the output of the program when different parameters are set. Specifically, these files represent the evolutionary state values and trends of a scenario where the group game network consists of 600 nodes with a connection strength of 15 during the first experiment.
### 2.2 Operating Environment
　　This program runs in a `Python 3.7` environment. The required libraries include `networkx`, `matplotlib`, `pandas`, `numpy`, and `tqdm`.

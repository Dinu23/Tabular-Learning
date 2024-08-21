# Tabular Learning: Reinforcement Learning Algorithms

## Overview

This repository contains implementations of various tabular reinforcement learning algorithms for solving sequential decision problems. The focus is on value-based methods, where the State-Action function \(Q(s, a)\) is estimated and maintained in a tabular format, commonly known as a Q-table. These algorithms are tested on grid-based environments to compare their performance under different settings and exploration strategies.

## Features

- **Dynamic Programming**: Implementations of policy evaluation and policy iteration methods.
- **Model-Free Algorithms**: 
  - **SARSA**: On-policy temporal difference learning.
  - **Q-Learning**: Off-policy temporal difference learning.
  - **Monte-Carlo Methods**: Learning from complete episodes.
- **Exploration Strategies**: 
  - ϵ-greedy
  - Boltzmann exploration
  - Upper Confidence Bound (UCB)
  

### Prerequisites

Ensure you have Python 3.x installed along with the necessary libraries:

```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Dinu23/Tabular-Learning.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Tabular-Learning
   ```

### Running the Algorithms

You can run the algorithms directly through the provided scripts

Example command to run Q-Learning:
```bash
python Q_learning.py
```


## Contact

For any questions or inquiries, please reach out to:

- **Name**: Dinu Catalin-Viorel
- **Email**: viorel.dinu00@gmail.com

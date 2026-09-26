# RL Agents

![Preview](./img/agents.gif)


## Status

| Status | Meaning |
|---------|---------|
| 0 | Running |
| 1 | Started |
| 2 | Win |
| 3 | Lose |
| 4 | Draw |
| 5 | Truncated |

- **0 (Running)**: The process is currently active and in progress.
- **1 (Started)**: The process has been initiated and is preparing to run.
- **2 (Win)**: The process completed successfully with a winning outcome.
- **3 (Lose)**: The process completed with a losing or unsuccessful outcome.
- **4 (Draw)**: The process completed without a winner or loser, resulting in a tie.
- **5 (Truncated)**: The episode was stopped before reaching a natural terminal state due to a step limit.
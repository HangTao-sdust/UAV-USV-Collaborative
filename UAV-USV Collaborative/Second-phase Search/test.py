
import subprocess

for i in range(100):
    subprocess.run(["python", ".\\path_find_k_mean_greedy.py", "--seed", f"{i}"])
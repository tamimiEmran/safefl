import subprocess
import time
import os

# Ensure results directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

# Common parameters for all protocols
common_params = [
    "--dataset", "MNIST",
    "--niter", "200",
    "--test_every", "10",
    "--nbyz", "0",
    "--nruns", "1",
    "--seed", "42",
    "--nworkers", "50",
    "--batch_size", "8"
    #"--byz_type", "trim_attack"
    ]

# Define protocol-specific parameters
protocols = {
        "flame": [
            "--aggregation", "flame",
            "--flame_epsilon", "3000",
            "--flame_delta", "0.001"
        ],
        "divide_and_conquer": [
            "--aggregation", "divide_and_conquer",
            "--dnc_niters", "5",
            "--dnc_c", "1",
            "--dnc_b", "2000"
        ],
        "heirichalFL": [
            "--aggregation", "heirichalFL",
            "--n_groups", "10",
            "--assumed_mal_prct", "0.0"
        ],
        "fltrust": [
            "--aggregation", "fltrust",
            "--server_pc", "100"
        ],
        "fedavg": [
            "--aggregation", "fedavg"
        ]

    }

def run_protocol(name, params):
    print(f"\n{'='*80}")
    print(f"Running {name} protocol")
    print(f"{'='*80}")

    # Build command
    cmd = ["python", "main.py"] + common_params + params

    # Print command for reference
    print("Executing command:")
    print(" ".join(cmd))

    # Track time
    start_time = time.time()

    # Run the process
    result = subprocess.run(cmd)

    # Check if the process was successful
    if result.returncode == 0:
        print(f"{name} protocol completed successfully")
    else:
        print(f"{name} protocol failed with return code {result.returncode}")

    # Print execution time
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")


# make heirichalFL the first protocol to run
protocols = {k: protocols[k] for k in ["heirichalFL", "fedavg", "flame", "divide_and_conquer", "fltrust"]}
# Run each protocol
for protocol_name, protocol_params in protocols.items():
    run_protocol(protocol_name, protocol_params)

print("\nAll protocols completed")
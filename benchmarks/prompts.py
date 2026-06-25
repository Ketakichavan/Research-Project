SIMPLE_PROMPTS = [
    "Summarize the benefits of edge computing",
    "What is latency in networking?",
    "Explain what an IoT device is"
]

COMPLEX_PROMPTS = [
    "Design a fault-tolerant edge computing architecture for an autonomous vehicle that handles sensor failures, network partitions, and real-time decisions simultaneously.",
    "An edge network has 10 nodes with varying compute capacity: nodes 1-3 have 8GB RAM, nodes 4-7 have 16GB RAM, nodes 8-10 have 32GB RAM. Design an optimal task scheduling algorithm that minimizes latency while maximizing resource utilization."
]

IMAGE_PROMPTS = [
    {
        "text": "Describe what you see in this network topology diagram and identify potential bottlenecks.",
        "image_path": "/home/jovyan/multi_agent_benchmark/benchmarks/images/network_topology.png"
    },
    {
        "text": "Analyze this system architecture diagram and suggest optimizations for edge computing.",
        "image_path": "/home/jovyan/multi_agent_benchmark/benchmarks/images/system_arch.png"
    },
    {
        "text": "What does this performance graph show? Identify any anomalies or trends.",
        "image_path": "/home/jovyan/multi_agent_benchmark/benchmarks/images/performance_graph.png"
    },
]
#!/bin/bash
# Usage: ./impairments.sh [apply_latency|apply_loss|apply_bandwidth|reset]

INTERFACE="lo"

apply_latency() {
    sudo tc qdisc add dev $INTERFACE root netem delay ${1:-100}ms
    echo "Applied ${1:-100}ms delay on $INTERFACE"
}

apply_loss() {
    sudo tc qdisc add dev $INTERFACE root netem loss ${1:-5}%
    echo "Applied ${1:-5}% packet loss on $INTERFACE"
}

apply_bandwidth() {
    sudo tc qdisc add dev $INTERFACE root tbf rate ${1:-1}mbit burst 32kbit latency 400ms
    echo "Applied ${1:-1}mbit bandwidth cap on $INTERFACE"
}

reset() {
    sudo tc qdisc del dev $INTERFACE root 2>/dev/null
    echo "Network reset to baseline"
}

$1 $2

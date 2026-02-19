# Neuro-Symbolic 5G/6G Orchestrator

## Overview
This project implements an Intent-Based Networking (IBN) orchestration system for 5G/6G networks, combining MARL (Multi-Agent Reinforcement Learning) with Symbolic Logic (LLM-based reasoning).

## Key Features
- **Tiered Architecture**: Legacy (Tier 1), Static AI (Tier 2), Neuro-Symbolic Orchestrator (Tier 3).
- **Safety Critical**: Hysteresis-tuned handovers and strict latency guarantees (<1ms for URLLC).
- **Energy Efficiency**: Physics-aware reward modeling (saving ~7.5% energy).
- **Real LLM Integration**: Supports OpenAI and Local LLM via TeleLLM interface.

## Running
1.  **Train Agents**: `python run/train.py`
2.  **Benchmark**: `python run/benchmark_scenarios.py`

## Artifacts
- Checkpoints: `models/mec_policy.pth`, `models/ho_policy.pth`
- Results: `data/scenarios/`

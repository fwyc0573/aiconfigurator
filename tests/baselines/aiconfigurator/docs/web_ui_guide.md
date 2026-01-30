# Web UI Guide

## Modification History

| Date       | Summary of Changes |
|------------|-------------------|
| 2026-01-09 | Initial guide for Web UI configuration and access |

## 1. Overview

The `aiconfigurator` Web UI provides an interactive interface for:
*   Visualizing Pareto frontiers (Throughput vs. Latency).
*   Comparing Aggregated vs. Disaggregated performance.
*   Configuring advanced parameters without editing YAML files.

It is built using **Gradio** and must be launched from the command line.

## 2. Launching the Web UI

### Standard Launch
To start the server on the default port (7860):

```bash
# Ensure environment is active
source .venv_aiconfigurator/bin/activate

# Launch
aiconfigurator webapp
```

### Custom Host and Port
You can bind to a specific interface or change the port using CLI arguments:

```bash
# Example: Listen on all interfaces, port 8080
aiconfigurator webapp --server_name 0.0.0.0 --server_port 8080
```

### Additional Flags
*   `--debug`: Enable verbose logging.
*   `--enable_profiling`: Show the profiling tab.
*   `--experimental`: Enable experimental features.

## 3. Remote Access Guide

Since the application is running on a remote headless server, you **cannot** open a browser on the server itself. You must use **SSH Port Forwarding** (Tunneling) to access the UI from your local machine.

### Step-by-Step Instructions

#### 1. Start the Server (Remote)
On the remote server, start the webapp binding to localhost (safer):
```bash
aiconfigurator webapp --server_name 127.0.0.1 --server_port 7860
```

#### 2. Create SSH Tunnel (Local)
On your **local machine** (laptop/desktop), open a terminal and run:

```bash
# Syntax: ssh -L [LocalPort]:127.0.0.1:[RemotePort] [User]@[RemoteHost]
ssh -L 7860:127.0.0.1:7860 ycfeng@frontier-server
```
*Replace `ycfeng@frontier-server` with your actual SSH connection string.*

#### 3. Access in Browser (Local)
Open your web browser (Chrome/Safari) on your local machine and navigate to:
**http://127.0.0.1:7860**

### Troubleshooting

| Issue | Solution |
|-------|----------|
| **Connection Refused** | Ensure the server is running and the port matches the one in your SSH command. |
| **Port in Use** | If 7860 is taken, use a different local port: `ssh -L 8080:127.0.0.1:7860 ...` then visit `localhost:8080`. |
| **Blank Page** | Check the remote server logs for python errors. Ensure `gradio` is installed correctly. |
